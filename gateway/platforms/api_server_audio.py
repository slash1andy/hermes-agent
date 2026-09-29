"""``POST /v1/audio/transcriptions`` (+ ``/p/<profile>/`` mirror) — authenticated,
profile-scoped, file-only multipart speech-to-text receiver reusing the existing STT engine
(``tools.transcription_tools``).

This is a bounded prerequisite slice, not a listener integration and not OpenAI Audio API
compatibility: it accepts exactly one ``file`` multipart field and returns ``{"text": str}``.
There is no ``url``/``model``/``provider`` selector, no ``response_format``, no streaming, and no
other OpenAI audio endpoint. A future slice wires an actual transport listener on top of this
receiver; landing this alone does not restore voice replies end-to-end.

Auth is stricter here than the shared OpenAI-compatible routes: those allow an unauthenticated
default-profile listener (test/manual wiring); this endpoint always requires a configured,
verified ``API_SERVER_KEY`` — it must never execute anonymously.
"""

from __future__ import annotations

import asyncio
import email
import email.policy
import hermes_constants
import logging
import os
import re
import tempfile
from contextlib import suppress
from email.message import Message
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - aiohttp is a hard dependency of api_server.py
    web = None  # type: ignore[assignment]

import gateway.proxy_admission as proxy_admission
import tools.transcription_common as transcription_common
import tools.transcription_tools as transcription_tools
from gateway.proxy_admission import ApiAgentAdmissionDenied

logger = logging.getLogger("gateway.platforms.api_server")

FIELD_NAME = "file"
# Wire cap = the existing native remote-audio file cap (tools.transcription_common.MAX_FILE_SIZE)
# plus bounded room for multipart boundaries/headers around the one file field. This never
# changes the unrelated global request-size cap (MAX_REQUEST_BYTES in api_server.py).
_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_AUDIO_UPLOAD_WIRE_BYTES = transcription_common.MAX_FILE_SIZE + _MULTIPART_OVERHEAD_BYTES

# Deliberately no leading/trailing slack: a leaf filename is matched EXACTLY against the raw
# decoded text (never stripped/normalized first), so padding, control characters and path
# separators all fail this fullmatch on their own.
_SAFE_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_DISPOSITION_UNIQUE_PARAMS = ("name", "filename")


class _STTDisabledError(Exception):
    """Raised by the worker when the resolved profile's ``stt.enabled`` is false — checked fresh
    at threaded-processing entry and again before any local fallback, never trusted from an
    earlier read."""


def _api_server_module():
    """Lazy import of the facade module — avoids a circular import at module load time
    (api_server.py imports this module lazily from inside a handler method)."""
    from gateway.platforms import api_server as _api_server
    return _api_server


def _bad_request(message: str, *, code: str = "invalid_request") -> "web.Response":
    return _api_server_module()._error_response(message, 400, code=code)


def _fixed_error(message: str, status: int, *, code: str, err_type: str = "invalid_request_error") -> "web.Response":
    return _api_server_module()._error_response(message, status, code=code, err_type=err_type)


def _policy_denied_response(exc: ApiAgentAdmissionDenied) -> "web.Response":
    return _api_server_module()._error_response(
        str(exc), 503 if exc.malformed else 403, code="gateway.proxy_required")


def _safe_audio_extension(filename: Optional[str]) -> Optional[str]:
    """Return a supported native audio extension for a strictly-validated leaf filename, or
    ``None``. The filename is checked EXACTLY as decoded (no stripping/normalizing): path
    separators, traversal, URL schemes, percent-encoded nesting and any padding/control
    character all fail. It is display-only metadata and is NEVER joined into a storage path."""
    if not filename:
        return None
    if "/" in filename or "\\" in filename or ".." in filename or "://" in filename or "%" in filename:
        return None
    if _SAFE_FILENAME_RE.fullmatch(filename) is None:
        return None
    ext = Path(filename).suffix.lower()
    return ext if ext in transcription_common.SUPPORTED_FORMATS else None


def _has_any_defect(message: Message) -> bool:
    """True when the message or any nested part/header recorded a structural parse defect
    (missing boundary, malformed header, ...). ``email.policy.default`` still parses a defective
    body permissively instead of raising, so defects must be checked explicitly."""
    return any(
        part.defects or any(getattr(header, "defects", ()) for _, header in part.items())
        for part in message.walk()
    )


def _disposition_fields(part: Message) -> Optional[Tuple[str, Optional[str]]]:
    """Return ``(name, filename)`` for one strictly-valid ``form-data`` part, or ``None`` when the
    part is anything other than exactly one well-formed ``Content-Disposition: form-data`` header
    with at most one ``name``/``filename`` param each (a duplicated header or duplicated param is
    rejected, never silently resolved to the first occurrence)."""
    headers = part.get_all("Content-Disposition") or []
    if len(headers) != 1:
        return None
    if part.get_content_disposition() != "form-data":
        return None
    if part.get("Content-Transfer-Encoding") is not None:
        # multipart/form-data parts never carry a transfer encoding; presence of one is an
        # encoded-body smuggling attempt, rejected outright rather than decoded.
        return None
    seen: set = set()
    name: Optional[str] = None
    filename: Optional[str] = None
    for key, value in part.get_params(header="Content-Disposition") or []:
        key_lower = key.lower()
        if key_lower not in _DISPOSITION_UNIQUE_PARAMS:
            continue
        if key_lower in seen:
            return None
        seen.add(key_lower)
        if key_lower == "name":
            name = value
        else:
            filename = value
    return name or "", filename


def _parse_strict_multipart(
    content_type: str, body: bytes
) -> Optional[List[Tuple[str, Optional[str], bytes]]]:
    """Parse a fully-captured multipart/form-data body via the stdlib email parser (``policy.
    default``): bounded, in-memory, and never re-enters aiohttp's own streaming multipart reader
    (the wire bytes are already captured by ``_read_limited_request_body``). Returns one
    ``(field_name, filename_or_None, raw_bytes)`` tuple per part, or ``None`` when the body is not
    EXACTLY a well-formed top-level ``multipart/form-data`` payload with no defects — including a
    missing closing boundary, a non-``form-data`` child part, a nested multipart child part, or any
    part with a duplicated ``Content-Disposition`` header/param or a transfer encoding. Nothing is
    ever silently skipped: any structural violation rejects the whole body."""
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode(
        "ascii", errors="replace")
    try:
        message = email.message_from_bytes(header + body, policy=email.policy.default)
    except Exception:
        return None
    if _has_any_defect(message):
        return None
    if message.get_content_type() != "multipart/form-data":
        return None
    if not message.is_multipart():
        return None
    children = message.get_payload()
    if not isinstance(children, list) or not children:
        return None
    parts: List[Tuple[str, Optional[str], bytes]] = []
    for part in children:
        if not isinstance(part, Message) or part.is_multipart():
            return None
        fields = _disposition_fields(part)
        if fields is None:
            return None
        name, filename = fields
        payload = part.get_payload(decode=True)
        parts.append((name, filename, payload if isinstance(payload, bytes) else b""))
    return parts


async def _read_bounded_multipart_file(
    request: "web.Request",
) -> Tuple[Optional[bytes], Optional[str], Optional["web.Response"]]:
    """Read + strictly validate the request body. Returns ``(file_bytes, extension, None)`` on
    success, else ``(None, None, error_response)``. Exactly one ``file`` field, a supported
    native audio extension, non-empty content, within the existing remote-audio size cap — any
    extra/duplicate field (including a duplicate ``file``) is rejected by the single-part rule."""
    content_type = request.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type.lower():
        return None, None, _bad_request("A multipart/form-data upload is required.")
    from gateway.platforms.whatsapp_cloud import _read_limited_request_body
    try:
        raw = await _read_limited_request_body(request, MAX_AUDIO_UPLOAD_WIRE_BYTES)
    except ValueError:
        return None, None, _fixed_error(
            "Upload exceeds the maximum allowed size.", 413, code="audio_too_large")
    except Exception:
        return None, None, _bad_request("Failed to read the request body.")

    parts = _parse_strict_multipart(content_type, raw)
    if parts is None:
        return None, None, _bad_request("Malformed multipart body.")
    if len(parts) != 1:
        return None, None, _bad_request(f"Exactly one '{FIELD_NAME}' field is required.")
    name, filename, payload = parts[0]
    if name != FIELD_NAME:
        return None, None, _bad_request(f"Exactly one '{FIELD_NAME}' field is required.")
    extension = _safe_audio_extension(filename)
    if extension is None:
        return None, None, _bad_request("Missing or unsupported audio filename.")
    if not payload:
        return None, None, _bad_request("Uploaded file is empty.")
    if len(payload) > transcription_common.MAX_FILE_SIZE:
        return None, None, _fixed_error(
            "Uploaded file exceeds the maximum size.", 413, code="audio_too_large")
    return payload, extension, None


def _worker_process_audio(payload: bytes, extension: str) -> Dict[str, Any]:
    """Synchronous worker (run via ``asyncio.to_thread``): owns the uploaded bytes' whole
    on-disk lifetime. Rechecks the process-root proxy policy AND the resolved profile's STT-
    enabled flag fresh at threaded-processing entry and again before any local fallback (the
    earlier reads in the handler are a narrow window, not a guarantee); writes the bytes under
    the OWNING profile's own ``cache/audio`` dir — never the client's filename joined into a
    path — calls the real STT seam, and unlinks the temp file in ``finally`` regardless of
    outcome so a cancelled/disconnected caller can never race the bytes away from STT while it
    is still reading them. Only an explicit ``{"success": False, ...}`` dict from the primary
    provider triggers the existing local-fallback seam; anything else (non-dict, missing/odd
    ``success``) is returned as-is for the caller to treat as a fixed failure."""
    proxy_admission.admit_local_api_agent_creation()
    if not transcription_tools.is_stt_enabled():
        raise _STTDisabledError()
    cache_dir = Path(hermes_constants.get_hermes_home()) / "cache" / "audio"
    hermes_constants.mkdir_under_hermes_home(cache_dir)
    fd, tmp_path = tempfile.mkstemp(dir=str(cache_dir), prefix="stt-upload-", suffix=extension)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        result = transcription_tools.transcribe_audio(tmp_path)
        if isinstance(result, dict) and result.get("success") is False:
            proxy_admission.admit_local_api_agent_creation()
            if not transcription_tools.is_stt_enabled():
                raise _STTDisabledError()
            result = transcription_tools.transcribe_audio_local_fallback(tmp_path)
        return result
    finally:
        with suppress(OSError):
            os.unlink(tmp_path)


def _finish_worker_task(adapter: Any, reservation: Dict[str, bool], task: "asyncio.Task") -> None:
    """Task done-callback: retrieve any exception (so a caller that abandoned the awaiting
    coroutine on disconnect never leaves an "exception was never retrieved" log) and release the
    pending-work reservation exactly once."""
    with suppress(BaseException):
        task.exception()
    _api_server_module()._release_pending_api_work(adapter, reservation)


async def handle_audio_transcriptions(adapter: Any, request: "web.Request") -> "web.Response":
    """Real handler for ``POST /v1/audio/transcriptions``; ``api_server.py`` delegates to this
    thinly, mirroring the ``api_server_cron_delivery`` sibling pattern."""
    _api_server = _api_server_module()

    if not adapter._expected_api_key():
        # Anonymous default-profile access is allowed on the shared OpenAI-compatible routes
        # (test/manual wiring); this endpoint must never execute without a verified key.
        return adapter._auth_failed_response()
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    draining = adapter._draining_response()
    if draining is not None:
        return draining
    try:
        proxy_admission.admit_local_api_agent_creation()
    except ApiAgentAdmissionDenied as exc:
        return _policy_denied_response(exc)

    denial_response: Optional["web.Response"] = None
    result: Optional[Dict[str, Any]] = None
    # Reserved from before the first await (the bounded body read) through worker completion —
    # a request in flight here must stay visible to shutdown drain the whole time, not only once
    # parsing finishes.
    with _api_server._reserve_pending_api_work(adapter) as reservation:
        payload, extension, err = await _read_bounded_multipart_file(request)
        if err is not None:
            return err

        task = asyncio.ensure_future(asyncio.to_thread(_worker_process_audio, payload, extension))
        # The done callback owns the reservation once detached: a cancelled/disconnected caller
        # must not free the slot (or delete the upload) while the worker thread — which
        # ``asyncio.shield`` insulates from this coroutine's own cancellation — is still running.
        reservation["detached"] = True
        task.add_done_callback(lambda done: _finish_worker_task(adapter, reservation, done))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except ApiAgentAdmissionDenied as exc:
            denial_response = _policy_denied_response(exc)
        except _STTDisabledError:
            denial_response = _fixed_error("stt is disabled for this profile.", 503, code="stt_disabled")
        except Exception:
            # Fixed label only: no exception text, traceback, or upload path ever crosses the
            # HTTP boundary (a provider failure's error text can carry a leaked credential).
            logger.error("audio transcription worker failed")
            denial_response = _fixed_error(
                "Audio transcription failed.", 502, code="audio_transcription_failed",
                err_type="server_error")

    if denial_response is not None:
        return denial_response
    if not isinstance(result, dict) or result.get("success") is not True:
        return _fixed_error(
            "Audio transcription failed.", 502, code="audio_transcription_failed", err_type="server_error")
    transcript = result.get("transcript")
    if not isinstance(transcript, str):
        return _fixed_error(
            "Audio transcription failed.", 502, code="audio_transcription_failed", err_type="server_error")
    return web.json_response({"text": transcript})
