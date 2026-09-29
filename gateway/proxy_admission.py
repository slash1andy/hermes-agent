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


def gateway_proxy_required() -> bool:
    """``True`` when this process's own launch config declares ``gateway.proxy_required: true``.

    Reads via ``get_routing_process_hermes_home()`` (the pinned launch identity, else the live
    process env — never a served profile's ContextVar override). The active task may have a
    served profile's secret scope installed (e.g. mid turn); a root-owned ``${VAR}`` ref must
    resolve against THIS process's own authority, not that scope, so the scope is cleared here for
    the read and restored immediately after — the same ``set_secret_scope``/``reset_secret_scope``
    primitive every other scoped read uses, no ``os.environ`` mutation.

    Raises ``ProxyPolicyError`` (never the raw parse/config exception, which may quote file
    content) when the root config can't be parsed, the root or ``gateway`` shape isn't a mapping,
    or ``proxy_required`` is present but not a plain bool — callers must treat that as "policy
    unconfirmed, refuse this turn entirely" (fail closed to denying BOTH local and proxy dispatch),
    not "absent" and not merely "required".
    """
    config_path = get_routing_process_hermes_home() / "config.yaml"
    token = set_secret_scope(None)
    try:
        # ``require_readable_config_before_write`` is the strict primitive that fails closed on an
        # unreadable/unparseable/non-mapping root instead of collapsing it to ``{}``. Its RETURNED
        # mapping is what gets expanded/overlaid below — never a second open of config_path — so
        # there is no window between "validated" and "read for real" where a racing writer could
        # replace the file with non-mapping content that a fresh re-open would silently coerce away.
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
