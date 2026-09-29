"""Required-proxy admission for ordinary gateway turns and background tasks.

``gateway.proxy_required`` on THIS process's own launch config — its process-root
``config.yaml`` (never a served profile's), read via ``get_routing_process_hermes_home`` — decides
whether an ordinary turn or ``/bg`` task may fall back to local execution when no usable proxy URL is configured.
Missing key: current default (``False``, not required) explicitly allows an ordinary turn to run
locally when no proxy URL resolves. An explicit ``proxy_required: false`` does the same. Any
explicit non-boolean value, a ``gateway`` section present but not a mapping, or
a root config that fails to parse or isn't a mapping: fails closed — refusing BOTH the local and the
proxy dispatch for that turn, not merely widening to "required" — so a malformed root policy can
never silently permit local execution. The URL that is ultimately dispatched to (when a proxy is
used) still comes from the resolving profile's own scope (``GATEWAY_PROXY_URL`` / ``gateway.proxy_url``);
only the require/refuse POLICY itself is pinned to the process root.

Configuration guard, not a proxy/URL validator — it only answers whether this ordinary turn or
``/bg`` task may skip the remote proxy path; the exact proxy URL resolution and dispatch stay in
``gateway/run_turn.py::_get_proxy_url`` / ``_run_agent_via_proxy``.

``admit_local_maintenance`` shares the same ``gateway_proxy_required`` read for OUT-OF-TURN session
maintenance (auto hygiene compression, manual ``/compress``): callers recheck it across admission/
preparation and local construction or cached-compaction boundaries.
"""

from __future__ import annotations

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.config import require_readable_config_before_write
from hermes_cli.config_effective import _effective
from hermes_constants import get_routing_process_hermes_home

# Named in every denial message so a caller/log can grep the exact policy that blocked the turn.
PROXY_REQUIRED_KEY = "gateway.proxy_required"


class ProxyPolicyError(Exception):
    """Raised when the process-root proxy policy cannot be confirmed safe (fail closed)."""


class ApiAgentAdmissionDenied(Exception):
    """Raised by ``admit_local_api_agent_creation`` when the process-root proxy policy refuses
    local execution of a native API agent turn. Distinct from ``ProxyPolicyError`` (which is only
    the config-read primitive) and never ``_ProviderAuthResolutionError`` — that type's caller
    contract renders a plain completed-looking text reply, which would silently launder a refused
    turn as success. ``malformed`` distinguishes an unresolved/invalid root policy (503) from an
    explicit ``proxy_required: true`` (403); the message is a fixed, safe diagnostic sentence with
    no config or exception content."""

    def __init__(self, message: str, *, malformed: bool = False) -> None:
        super().__init__(message)
        self.malformed = malformed


def admit_local_maintenance() -> None:
    """Recheck ``gateway.proxy_required`` at out-of-turn admission/preparation and local
    construction or cached-compaction boundaries. Raises ``ProxyPolicyError`` for BOTH an
    explicit ``proxy_required: true`` and an unresolved/malformed root policy — maintenance has no
    API-specific 403-vs-503 story, so both fail closed the same way, uniformly. Callers place
    checks at the preparation and local execution boundaries so a late flip is caught before it
    runs a local model."""
    if gateway_proxy_required():
        raise ProxyPolicyError(
            f"{PROXY_REQUIRED_KEY} disables local out-of-turn maintenance for this process."
        )


def admit_local_api_agent_creation() -> None:
    """Recheck ``gateway.proxy_required`` before native API agent creation."""
    try:
        proxy_required = gateway_proxy_required()
    except ProxyPolicyError as exc:
        raise ApiAgentAdmissionDenied(
            f"{PROXY_REQUIRED_KEY} policy is unresolved; refusing this API agent request.",
            malformed=True) from exc
    if proxy_required:
        raise ApiAgentAdmissionDenied(
            f"local API agent execution is disabled by {PROXY_REQUIRED_KEY}.")


def gateway_proxy_required() -> bool:
    """Read the process-root policy, isolated from any served profile scope.

    Invalid or unreadable policy raises ``ProxyPolicyError`` so callers fail closed.
    """
    config_path = get_routing_process_hermes_home() / "config.yaml"
    token = set_secret_scope(None)
    try:
        try:
            raw = require_readable_config_before_write(config_path)
        except Exception as exc:
            raise ProxyPolicyError(
                "the process config could not be confirmed as a valid mapping, so "
                f"{PROXY_REQUIRED_KEY} cannot be confirmed — failing closed"
            ) from exc
        try:
            effective = _effective(raw)
        except Exception as exc:
            raise ProxyPolicyError(
                "the process config could not be parsed, so "
                f"{PROXY_REQUIRED_KEY} cannot be confirmed — failing closed"
            ) from exc
    finally:
        reset_secret_scope(token)

    if "gateway" not in effective:
        return False  # Absent entirely: current default (not required) is preserved.
    gateway_section = effective["gateway"]
    if not isinstance(gateway_section, dict):
        raise ProxyPolicyError(
            f"the process config's 'gateway' section is not a mapping, so {PROXY_REQUIRED_KEY} "
            "cannot be confirmed — failing closed"
        )
    if "proxy_required" not in gateway_section:
        return False
    value = gateway_section["proxy_required"]
    if value is True:
        return True
    if value is False:
        return False
    raise ProxyPolicyError(
        f"{PROXY_REQUIRED_KEY} is set but is not a plain boolean — failing closed"
    )
