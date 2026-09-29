"""Shared validation/authorization for the authenticated isolated cron delivery HTTP extension
(``POST /p/{profile}/cron/deliveries``).

Used by both the HTTP ingress handler (``gateway/platforms/api_server_cron_delivery.py``) and the
delivery-time transport guard (``cron/scheduler_delivery.py::_resolve_target_transport``) so the
two stay in lockstep: an envelope built here carries the same marker the transport guard checks
before it would otherwise fall through to the standalone lane on a revoked/unmatched route.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

# Internal marker on the transient job dict, holding the OWNING profile name (not a bare bool) —
# never settable from the HTTP body. ``_resolve_target_transport`` compares it against the current
# scoped home so a job admitted/processed under the wrong profile's tick denies even when that
# profile's own routes would otherwise match the same chat.
MARKER = "_isolated_http_delivery"

EXECUTION_ID_MAX = 128
TARGET_ID_MAX = 256


def is_marked(job: dict) -> bool:
    return bool(job.get(MARKER))


def owning_profile_matches_current_home(job: dict) -> bool:
    """True iff the envelope's owning profile is the profile currently scoped (same check
    ``profile_routes`` matching uses) — catches a marked job somehow drained under a different
    profile's tick."""
    owner = job.get(MARKER)
    if not isinstance(owner, str) or not owner:
        return False
    from hermes_cli.profiles import profile_matches_home
    try:
        return bool(profile_matches_home(owner))
    except Exception:
        return False


def build_envelope(execution_id: str, platform: str, chat_id: str, thread_id: Optional[str], profile: str) -> dict:
    """The minimal INTERNAL job envelope persisted/drained for one isolated delivery. Carries no
    script/session/callback fields — only what ``_resolve_delivery_targets``' ``platform:chat_id[:
    thread_id]`` grammar needs, plus the owning profile for the consumer-side check above."""
    deliver = f"{platform}:{chat_id}:{thread_id}" if thread_id else f"{platform}:{chat_id}"
    return {"id": execution_id, "deliver": deliver, MARKER: profile}


def request_fingerprint(
    *, profile: str, execution_id: str, platform: str, chat_id: str,
    thread_id: Optional[str], content: str,
) -> str:
    """Hash of the validated envelope, bound to the profile; no secret material included. Used to
    tell an idempotent replay (same fingerprint) from a same-id conflict (payload/target/profile
    changed) in ``cron.delivery_queue``."""
    canonical = json.dumps(
        {"profile": profile, "execution_id": execution_id, "platform": platform,
         "chat_id": chat_id, "thread_id": thread_id, "content": content},
        sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def authorize_target(platform, chat_id: str, thread_id: Optional[str]) -> bool:
    """True iff the primary gateway's CURRENT ``profile_routes`` grant the scoped profile this
    exact target, via the same ``SharedRouteAdapters`` gate ``_resolve_target_transport`` uses —
    reused (not re-derived) so ingress-time and delivery-time checks can't drift. Reads config
    fresh every call (no caching); owned-adapter profiles' own per-chat allowlist has no defined
    reader yet, so that case denies rather than guesses.
    """
    from cron.scheduler_preflight import SharedRouteAdapters, _primary_profile_routes_for_current_home

    routes = _primary_profile_routes_for_current_home()
    probe = SharedRouteAdapters({platform: object()}, routes)
    return probe.get(platform, {"chat_id": chat_id, "thread_id": thread_id}) is not None
