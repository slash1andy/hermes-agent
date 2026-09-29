"""Handler for ``POST /p/{profile}/cron/deliveries`` — the authenticated isolated cron delivery
ingress extension. ``gateway/platforms/api_server.py`` registers the route and delegates the whole
request here; this module owns parsing, authentication, authorization and the durable-enqueue
receipt. See ``tests/gateway/test_api_server_isolated_cron_delivery.py`` for the full contract and
``cron/isolated_delivery.py`` for the validation/authorization primitives shared with the
delivery-time transport guard.

Scope of this slice: a bounded plain-text envelope, a dedicated per-profile credential, and a
durable enqueue that the native scheduler drains through the real live-adapter/standalone lanes.
No media, no arbitrary job fields, no synchronous send.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from aiohttp import web

from cron import isolated_delivery as _isolated

logger = logging.getLogger(__name__)

MAX_WIRE_BYTES = 64 * 1024
MAX_CONTENT_BYTES = 16 * 1024
MIN_DELIVERY_KEY_LEN = 32

_ALLOWED_FIELDS = frozenset({"execution_id", "platform", "chat_id", "thread_id", "content"})
_REQUIRED_FIELDS = frozenset({"execution_id", "platform", "chat_id", "content"})
_RESERVED_PLATFORM_TOKENS = frozenset({"bot-chat", "local", "origin", "all"})

# Allowlist-only, no path/control chars. fullmatch (not match+^$) so a trailing "\n" — which "$"
# alone would let slip past — is rejected.
_EXECUTION_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,%d}" % _isolated.EXECUTION_ID_MAX)
_TARGET_ID_RE = re.compile(r"[^\s\x00-\x1f\x7f:,]{1,%d}" % _isolated.TARGET_ID_MAX)


def _reject(status: int, message: str) -> "web.Response":
    """Bare, secret-free error envelope — never echoes request content or internals."""
    return web.json_response({"error": message}, status=status)


def _expected_delivery_key() -> str:
    """The dedicated per-profile ``CRON_DELIVERY_KEY`` for the currently scoped profile, or ``""``
    when absent/unusable. Never ``API_SERVER_KEY``, never an environment/root fallback — reads
    through the fail-closed profile secret scope exactly like ``_expected_api_key``'s named-profile
    branch."""
    try:
        from agent.secret_scope import get_secret
        from hermes_cli.auth import has_usable_secret
        key = get_secret("CRON_DELIVERY_KEY", "") or ""
    except Exception:
        return ""
    return key if has_usable_secret(key, min_length=MIN_DELIVERY_KEY_LEN) else ""


def _authenticate(request: "web.Request") -> bool:
    import hmac
    expected = _expected_delivery_key()
    if not expected:
        return False
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header[7:].strip()
    try:
        return hmac.compare_digest(token.encode(), expected.encode())
    except Exception:
        return False


def _validate_body(payload: Any) -> tuple[Optional[dict], Optional[str]]:
    """Strict allowlist validation -> ``(clean_fields, None)`` or ``(None, error)``."""
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object"
    unknown = set(payload) - _ALLOWED_FIELDS
    if unknown:
        return None, "Unknown field(s) in request body"
    missing = _REQUIRED_FIELDS - set(payload)
    if missing:
        return None, "Missing required field(s)"

    execution_id = payload.get("execution_id")
    if not isinstance(execution_id, str) or not _EXECUTION_ID_RE.fullmatch(execution_id):
        return None, "Invalid execution_id"

    # Platform is taken literally: no .strip()/.lower() normalization, which would let a
    # whitespace- or case-varied token slip past the reserved-token check while still resolving to
    # a real platform downstream.
    platform = payload.get("platform")
    if (not isinstance(platform, str) or not platform
            or platform != platform.lower() or platform != platform.strip()):
        return None, "Invalid platform"
    if platform in _RESERVED_PLATFORM_TOKENS:
        return None, "Invalid platform"
    from cron.scheduler_delivery import _is_known_delivery_platform
    if not _is_known_delivery_platform(platform):
        return None, "Unknown delivery platform"
    from gateway.config import Platform
    try:
        Platform(platform)
    except (ValueError, KeyError):
        return None, "Unknown delivery platform"

    chat_id = payload.get("chat_id")
    if not isinstance(chat_id, str) or not _TARGET_ID_RE.fullmatch(chat_id):
        return None, "Invalid chat_id"

    thread_id = payload.get("thread_id")
    if thread_id is not None:
        if not isinstance(thread_id, str) or not _TARGET_ID_RE.fullmatch(thread_id):
            return None, "Invalid thread_id"

    content = payload.get("content")
    if not isinstance(content, str):
        return None, "Invalid content"
    try:
        content_bytes = content.encode("utf-8")
    except UnicodeEncodeError:
        return None, "Invalid content encoding"
    if len(content_bytes) > MAX_CONTENT_BYTES:
        return None, "content too large"

    # Case-sensitive substring matching (_has_media_directives) misses lowercase/mixed-case and
    # Unicode-casefolded variants of the MEDIA: tag (e.g. "media:", "MeDiA:", "medİa:"). The native
    # tag regexes are re.IGNORECASE and match those variants correctly — used directly (bare
    # .search(), never the stat-performing _extensionless_media_matches/validate_media_delivery_path
    # helpers) so rejection never touches the filesystem.
    from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE, MEDIA_EXTENSIONLESS_TAG_RE
    if (MEDIA_TAG_CLEANUP_RE.search(content) or MEDIA_EXTENSIONLESS_TAG_RE.search(content)
            or "[[audio_as_voice]]" in content or "[[as_document]]" in content):
        return None, "Media directives are not accepted on this endpoint"

    return {
        "execution_id": execution_id, "platform": platform, "chat_id": chat_id,
        "thread_id": thread_id, "content": content,
    }, None


async def handle_cron_delivery(request: "web.Request") -> "web.Response":
    """``POST /p/{profile}/cron/deliveries``. Requires a named profile prefix + that profile's own
    ``CRON_DELIVERY_KEY``; the caller's target is authorized against the primary gateway's own
    ``profile_routes``, never trusted from the request. Durable enqueue only — delivery happens on
    the next native scheduler drain."""
    from gateway.platforms.api_server import _api_request_profile
    profile = _api_request_profile.get()
    if not profile:
        return _reject(403, "This endpoint requires a named /p/<profile>/ prefix")

    if not _authenticate(request):
        return _reject(401, "Invalid or missing CRON_DELIVERY_KEY")

    if profile == "default":
        # A valid dedicated key on the root/launch profile still can't authorize root sends —
        # the endpoint only serves NAMED satellite profiles.
        return _reject(403, "This endpoint does not serve the default profile")

    # readexactly (not read()) accumulates to EOF/the cap instead of returning a partial chunk.
    from gateway.platforms.whatsapp_cloud import _read_limited_request_body
    try:
        raw = await _read_limited_request_body(request, MAX_WIRE_BYTES)
    except ValueError:
        return _reject(400, "Request body too large")
    except Exception:
        return _reject(400, "Failed to read request body")
    try:
        payload = json.loads(raw)
    except Exception:
        return _reject(400, "Invalid JSON in request body")

    fields, error = _validate_body(payload)
    if error is not None:
        return _reject(400, error)

    from gateway.config import Platform
    platform_enum = Platform(fields["platform"])
    if not _isolated.authorize_target(platform_enum, fields["chat_id"], fields["thread_id"]):
        return _reject(403, "Target is not authorized for this profile")

    from cron import delivery_queue
    envelope = _isolated.build_envelope(
        fields["execution_id"], fields["platform"], fields["chat_id"], fields["thread_id"], profile)
    fingerprint = _isolated.request_fingerprint(
        profile=profile, execution_id=fields["execution_id"], platform=fields["platform"],
        chat_id=fields["chat_id"], thread_id=fields["thread_id"], content=fields["content"])
    import asyncio
    try:
        result = await asyncio.to_thread(
            delivery_queue.enqueue, fields["execution_id"], envelope, fields["content"],
            request_fingerprint=fingerprint)
    except Exception:
        # Fixed-label only: logger.exception would attach the raw exception (and, via its args,
        # potentially payload content) to the log record. No traceback, no exception text.
        logger.error("isolated cron delivery enqueue failed")
        return _reject(503, "Delivery queue temporarily unavailable")

    status = result.get("status")
    if status == "conflict":
        return _reject(409, "execution_id already used with a different payload/target")
    receipt = {"execution_id": result["execution_id"], "status": status}
    if status in {"pending", "delivering"}:
        return web.json_response(receipt, status=202)
    return web.json_response(receipt, status=200)
