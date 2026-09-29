"""Remote (native-proxy) voice-transcription client for GatewayRunner.

Split out of ``gateway/run_inbound.py``: the HTTP client and its per-clip enrichment loop for the
configured native STT receiver, kept separate from the local-provider path and the shared policy
helpers (which remain on ``GatewayInboundMixin`` in ``gateway/run_inbound.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import List, Optional, Tuple

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


def _recheck_remote_stt_admission(self, expected_proxy_url: str) -> None:
    """Fresh recheck immediately before dispatching to the remote receiver: a malformed/revoked
    root policy or a proxy URL that changed mid-loop refuses this request rather than silently
    using a stale snapshot or auto-rerouting to whatever now resolves."""
    from gateway.proxy_admission import ProxyPolicyError
    deny_local, proxy_url = self._resolve_stt_dispatch_policy()
    if not deny_local or proxy_url != expected_proxy_url:
        raise ProxyPolicyError(
            "the proxy policy or URL changed mid-transcription — refusing this remote request."
        )


async def _transcribe_one_clip_via_proxy(
    self, path: str, proxy_url: str, proxy_key: str
) -> Tuple[Optional[str], str]:
    """``(transcript_or_None, note)`` for one clip via the configured native remote receiver.
    Single attempt: no automatic HTTP retry and no local fallback — a failed/uncertain remote
    request stays failed for this clip."""
    from gateway.proxy_admission import ProxyPolicyError
    from gateway.run import _GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS
    from tools.transcription_common import DEFAULT_STT_TIMEOUT, MAX_FILE_SIZE, SUPPORTED_FORMATS

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        return None, self._PROXY_STT_FAILURE_NOTE

    def _read_bounded() -> Optional[bytes]:
        try:
            with open(path, "rb") as handle:
                data = handle.read(MAX_FILE_SIZE + 1)
        except OSError:
            return None
        return None if len(data) > MAX_FILE_SIZE else data

    data = await asyncio.to_thread(_read_bounded)
    if data is None:
        return None, self._PROXY_STT_FAILURE_NOTE

    try:
        _recheck_remote_stt_admission(self, proxy_url)
    except ProxyPolicyError:
        return None, self._PROXY_STT_FAILURE_NOTE

    try:
        from aiohttp import ClientSession as _AioClientSession, ClientTimeout, FormData
    except ImportError:
        return None, self._PROXY_STT_FAILURE_NOTE

    form = FormData()
    form.add_field("file", data, filename=f"clip{ext}", content_type="application/octet-stream")
    headers = {"Authorization": f"Bearer {proxy_key}"}
    try:
        async with _AioClientSession(timeout=ClientTimeout(total=DEFAULT_STT_TIMEOUT)) as session:
            async with session.post(
                f"{proxy_url}/v1/audio/transcriptions", data=form, headers=headers,
                allow_redirects=False,
            ) as resp:
                if resp.status != 200:
                    return None, self._PROXY_STT_FAILURE_NOTE
                try:
                    body = await resp.content.readexactly(_GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS + 1)
                except asyncio.IncompleteReadError as exc:
                    body = exc.partial
                if len(body) > _GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS:
                    return None, self._PROXY_STT_FAILURE_NOTE
                result = json.loads(body)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.info("Voice transcription via proxy failed for a clip: %s", type(exc).__name__)
        return None, self._PROXY_STT_FAILURE_NOTE

    transcript = result.get("text") if isinstance(result, dict) else None
    if not isinstance(transcript, str):
        return None, self._PROXY_STT_FAILURE_NOTE
    if not transcript.strip():
        return None, (
            "[The user sent a voice message but it came through "
            "empty or inaudible — speech-to-text returned no "
            "words. Do not guess at the content; ask the user "
            "to resend or type it out.]"
        )
    return transcript, f'"{transcript}"'


async def _enrich_message_with_transcription_via_proxy(
    self, user_text: str, audio_paths: List[str], proxy_url: str
) -> tuple[str, List[str]]:
    """Remote counterpart of ``_enrich_message_with_transcription``'s local loop: every clip
    goes through the configured native receiver, never local STT."""
    from agent.secret_scope import UnscopedSecretError, get_secret
    from gateway.proxy_admission import ProxyPolicyError
    try:
        try:
            proxy_key = (get_secret("GATEWAY_PROXY_KEY") or "").strip()
        except UnscopedSecretError:
            proxy_key = os.getenv("GATEWAY_PROXY_KEY", "").strip()
    except Exception:
        return self._prepend_media_prefix(self._PROXY_STT_FAILURE_NOTE, user_text), []
    if not proxy_key:
        return self._prepend_media_prefix(self._PROXY_STT_FAILURE_NOTE, user_text), []

    enriched_parts = []
    successful_transcripts: List[str] = []
    for path in audio_paths:
        try:
            _recheck_remote_stt_admission(self, proxy_url)
        except ProxyPolicyError:
            enriched_parts.append(self._PROXY_STT_FAILURE_NOTE)
            continue
        transcript, note = await _transcribe_one_clip_via_proxy(self, path, proxy_url, proxy_key)
        if transcript is not None:
            successful_transcripts.append(transcript)
        enriched_parts.append(note)
    if enriched_parts:
        user_text = self._prepend_media_prefix("\n\n".join(enriched_parts), user_text)
    return user_text, successful_transcripts
