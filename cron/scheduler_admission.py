"""Cron execution admission for the shared transport process (cron/AGENTS.md).

``cron.execution_enabled`` on THIS process's own launch config — its launch profile's
``config.yaml`` (which may itself be a named profile), never a profile it merely SERVES under
multiplexing — decides whether it may run local cron work (script subprocess, AIAgent
construction, tick dispatch, external-worker handoff). Missing key: current default (enabled). Any
explicit value other than the boolean ``True``, an ``execution_enabled`` key present but not a
plain bool, a ``cron`` section present but not a mapping, or a config that fails to parse: denies
(fail closed).

Configuration guard, not an OS-level sandbox or a complete isolated-delivery runtime — it only
decides whether local cron execution may start; the delivery queue drains independently of it.
"""

from __future__ import annotations

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.config_effective import load_user_config_effective
from hermes_constants import get_routing_process_hermes_home

# Named in every denial message so a caller/log can grep the exact policy that blocked the run.
EXECUTION_ENABLED_KEY = "cron.execution_enabled"


def cron_execution_denied_reason() -> "str | None":
    """``None`` when this process may execute cron work locally; otherwise a safe, non-secret
    error naming the policy (never the parse exception's own text, which may quote file content).

    Reads via ``get_routing_process_hermes_home()`` (the pinned launch identity, else the live
    process env — never a served profile's ContextVar override). The active task may have a
    served profile's secret scope installed (e.g. mid cron-tick); a root-owned ``${VAR}`` ref must
    resolve against THIS process's own authority, not that scope, so the scope is cleared here for
    the read and restored immediately after — the same ``set_secret_scope``/``reset_secret_scope``
    primitive every other scoped read uses, no ``os.environ`` mutation.
    """
    config_path = get_routing_process_hermes_home() / "config.yaml"
    token = set_secret_scope(None)
    try:
        effective = load_user_config_effective(config_path, fail_closed=True)
    except Exception:
        return (
            f"Cron execution denied: the process config could not be parsed, so "
            f"{EXECUTION_ENABLED_KEY} cannot be confirmed true — failing closed.")
    finally:
        reset_secret_scope(token)

    if not isinstance(effective, dict) or "cron" not in effective:
        return None  # Absent entirely: current default (enabled) is preserved.
    cron_section = effective["cron"]
    if not isinstance(cron_section, dict):
        return (
            f"Cron execution denied: the process config's 'cron' section is not a mapping, so "
            f"{EXECUTION_ENABLED_KEY} cannot be confirmed true — failing closed.")
    if "execution_enabled" not in cron_section:
        return None
    if cron_section["execution_enabled"] is True:
        return None
    return (
        f"Cron execution denied: {EXECUTION_ENABLED_KEY} is not explicitly true on the process "
        f"config — this transport process does not run local cron work.")
