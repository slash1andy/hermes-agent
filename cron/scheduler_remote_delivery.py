"""The isolated executor's HTTP producer: admits one job's outbound text per resolved target
through a configured remote profile's authenticated ``POST /p/{profile}/cron/deliveries`` endpoint
(``gateway/platforms/api_server_cron_delivery.py``) instead of any local adapter.

Split out of ``cron.scheduler_delivery`` because it owns its own HTTP client (stdlib only, no
redirects, no proxies) and config/secret resolution, rather than the target-resolution/formatting
concerns that module already owns. See ``native-isolated-producer-contract.md`` for the full
invariant set this implements.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import ssl
from typing import Any, Optional
from urllib.parse import urlsplit

logger = logging.getLogger("cron.scheduler")

_TIMEOUT_SECONDS = 10.0
_MAX_RESPONSE_BYTES = 64 * 1024
# Receiver's own pending/terminal vocabulary (cron/delivery_queue.py's `_TERMINAL` + its two
# non-terminal statuses) — the only strings this producer ever trusts out of a receipt body.
_PENDING_STATUSES = frozenset({"pending", "delivering"})
_TERMINAL_STATUSES = frozenset({"delivered", "failed", "unknown", "suppressed"})
# HTTP statuses the receiver returns for a PROVEN pre-admission rejection (never a claim that
# something already admitted then failed) — the only statuses "failed" is allowed to follow
# without a matching receipt body.
_PRE_ADMISSION_REJECTION_STATUSES = frozenset({400, 401, 403, 404})


def configured_delivery_gateway(cfg: Any) -> tuple[str, Optional[str]]:
    """``("disabled", None)`` | ``("fail_closed", None)`` | ``("configured", url)`` from the
    effective ``cron.delivery_gateway_url``. An unreadable/unparsable config.yaml (served as a
    :class:`hermes_cli.config_read_errors.FailedConfigRead`), a non-dict root, or a malformed
    ``cron`` section never resolves to "disabled" — the caller cannot tell that fallback from a
    genuinely empty setting, so it fails closed instead of silently sending nothing through this
    producer while looking healthy. A non-empty value that is not a plain ``http(s)://`` URL is
    equally malformed, never treated as absent."""
    from hermes_cli.config_read_errors import FailedConfigRead

    if isinstance(cfg, FailedConfigRead) or not isinstance(cfg, dict):
        return "fail_closed", None
    cron_cfg = cfg.get("cron", {})
    if not isinstance(cron_cfg, dict):
        return "fail_closed", None
    raw = cron_cfg.get("delivery_gateway_url")
    if raw in (None, ""):
        return "disabled", None
    if not isinstance(raw, str):
        return "fail_closed", None
    if any(char.isspace() or ord(char) <= 31 or ord(char) == 127 for char in raw):
        return "fail_closed", None
    try:
        parts = urlsplit(raw)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return "fail_closed", None
        if parts.username is not None or parts.password is not None:
            return "fail_closed", None
        if "?" in raw or "#" in raw:
            return "fail_closed", None

        netloc = parts.netloc
        if netloc.startswith("["):
            closing = netloc.find("]")
            suffix = netloc[closing + 1:] if closing >= 0 else "invalid"
            if suffix and not suffix.startswith(":"):
                return "fail_closed", None
            port_text = suffix[1:] if suffix else None
        else:
            port_text = netloc.rsplit(":", 1)[1] if ":" in netloc else None
        if port_text == "":
            return "fail_closed", None
        if parts.port is not None and not 1 <= parts.port <= 65535:
            return "fail_closed", None

        path_parts = parts.path.split("/")
        if len(path_parts) != 5 or path_parts[0:2] != ["", "p"] or path_parts[3:] != ["cron", "deliveries"]:
            return "fail_closed", None
        from hermes_cli.profiles import validate_profile_name
        try:
            validate_profile_name(path_parts[2])
        except ValueError:
            return "fail_closed", None
    except (ValueError, TypeError):
        return "fail_closed", None
    return "configured", raw


def configured_delivery_secret() -> Optional[str]:
    """The dedicated ``CRON_DELIVERY_KEY`` for the currently scoped (executor) profile, or ``None``
    when absent/unusable. Scope-bound via ``agent.secret_scope.get_secret`` — never an ambient or
    root fallback, and never ``API_SERVER_KEY``."""
    from agent.secret_scope import get_secret
    from cron.isolated_delivery import MIN_DELIVERY_KEY_LEN
    from hermes_cli.auth import has_usable_secret

    key = get_secret("CRON_DELIVERY_KEY", "") or ""
    return key if has_usable_secret(key, min_length=MIN_DELIVERY_KEY_LEN) else None


def operation_id(execution_id: str, lane: str, slot: int) -> str:
    """Stable, bounded SHA256 operation identity from the durable execution id + success/failure
    lane + resolved target slot — never target/content, so reordering or a content change can never
    silently mint a different id for the same logical admission."""
    canonical = f"{execution_id}:{lane}:{slot}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _post(url: str, secret: str, payload: dict) -> tuple[Optional[int], Optional[dict], bool]:
    """One bounded POST: redirects are never followed (``http.client`` does not auto-follow),
    ambient proxies are never consulted (``http.client`` reads none), the response is read up to a
    fixed cap, and there is no retry of any kind. Returns ``(status, parsed_body, reached_receiver)``
    — ``reached_receiver=False`` means a transport failure (connect/timeout/TLS); the raw exception
    is never logged, only this bare boolean."""
    try:
        parts = urlsplit(url)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {secret}",
            "Content-Length": str(len(body)),
        }
        if parts.scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                parts.hostname, parts.port or 443, timeout=_TIMEOUT_SECONDS,
                context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(
                parts.hostname, parts.port or 80, timeout=_TIMEOUT_SECONDS)
        try:
            path = parts.path or "/"
            if parts.query:
                path = f"{path}?{parts.query}"
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                return resp.status, None, True
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except Exception:
                parsed = None
            return resp.status, parsed if isinstance(parsed, dict) else None, True
        finally:
            conn.close()
    except Exception:
        return None, None, False


def _classify_receipt(status: Optional[int], body: Optional[dict], expected_id: str) -> str:
    """The receiver's own outcome for THIS operation id, or ``"unknown"``/``"failed"`` when the
    receipt cannot prove it. Every membership/attribute check is preceded by a type check — a
    malformed field (a list where a string is expected, a non-dict body) must fall through to
    ``"unknown"`` rather than raise.

    - A proven pre-admission rejection (400/401/403/404) is ``"failed"`` — no body needed, the
      receiver never wrote a queue row.
    - Otherwise the id must match AND the receipt's ``status`` must be a string AND the HTTP
      status must be the one the receiver actually pairs it with (202 only for pending/delivering,
      200 only for one of the receiver's own terminal statuses) — the exact string is returned
      unchanged (``"delivered"``/``"failed"``/``"unknown"``/``"suppressed"``), never re-labelled.
    - Anything else (wrong id, wrong HTTP/status pairing, an arbitrary/malformed status, a lost or
      unparsable body) is ``"unknown"`` — never a fallback/resend trigger, never raised.
    """
    if status in _PRE_ADMISSION_REJECTION_STATUSES:
        return "failed"
    if not isinstance(status, int) or not isinstance(body, dict):
        return "unknown"
    if body.get("execution_id") != expected_id:
        return "unknown"
    receipt_status = body.get("status")
    if not isinstance(receipt_status, str):
        return "unknown"
    if status == 202:
        return "queued" if receipt_status in _PENDING_STATUSES else "unknown"
    if status == 200:
        return receipt_status if receipt_status in _TERMINAL_STATUSES else "unknown"
    return "unknown"


def deliver_target(
    job: dict, target: dict, content: str, *, url: str, secret: str, lane: str, slot: int,
) -> tuple[str, Optional[str]]:
    """Admit one resolved target's content through the configured gateway: exactly one POST, no
    retry. Returns ``(outcome, error)`` — ``outcome`` is the receiver's own disposition
    (``"queued"``, ``"delivered"``, ``"suppressed"``, ``"unknown"`` or ``"failed"``); ``error`` is
    ``None`` for ``"queued"``/``"delivered"``/``"suppressed"`` (a legitimate, non-error terminal or
    pending state) and a fixed, safe, nonempty label for ``"unknown"``/``"failed"`` — never the raw
    URL, request/response body, secret or transport exception text."""
    execution_id = job.get("execution_id")
    if not execution_id:
        # Missing execution identity fails closed rather than inventing one from target/content.
        return "failed", "missing execution identity for configured delivery"
    op_id = operation_id(str(execution_id), lane, slot)
    payload: dict = {
        "execution_id": op_id, "platform": target["platform"], "chat_id": str(target["chat_id"]),
        "content": content,
    }
    thread_id = target.get("thread_id")
    if thread_id:
        payload["thread_id"] = str(thread_id)
    status, body, reached_receiver = _post(url, secret, payload)
    if not reached_receiver:
        # A lost response does NOT prove the endpoint was never reached (it may have admitted the
        # request and the reply was lost) — never word this as "did not reach"/"never admitted".
        return "unknown", f"configured delivery to {target['platform']} outcome is unconfirmed"
    outcome = _classify_receipt(status, body, op_id)
    if outcome in ("queued", "delivered", "suppressed"):
        return outcome, None
    if outcome == "failed":
        return "failed", f"configured delivery to {target['platform']} was rejected"
    return "unknown", f"configured delivery to {target['platform']} outcome is unconfirmed"
