"""Required proxy admission (``gateway.proxy_required``) — bounded coverage for the
``gateway/proxy_admission.py::gateway_proxy_required()`` reader and its
ordinary-turn admission point in ``gateway/run_turn.py::_run_agent_inner``
(see ``required-proxy-admission-contract.md``).

Every test drives the REAL ``GatewayRunner._run_agent`` -> ``_run_agent_inner`` entry point end to
end: the real config loader (``_load_gateway_config``/``load_user_config_effective``), the real
profile-scope binder (``_profile_scope_for_source`` -> ``_profile_runtime_scope``), and the real
``_get_proxy_url`` resolver all run unmodified. Two labelled stop sentinels stand in for the two
possible dispatch destinations so a positive/negative case can be proven without paying for a full
local turn or a real network call:

- ``_run_agent_build_turn_context`` (local turn preparation — the seam immediately before
  ``AIAgent``/provider/tool/CLI construction) — patched to record a call and raise
  ``_LocalPrepReached`` at once.
- ``_run_agent_via_proxy`` (remote dispatch) — patched the same way, raising
  ``_ProxyDispatchReached``, so a denial case can prove it made zero proxy attempts too ("zero
  effects"), not just zero local ones.

The tests keep the admission boundary real while replacing only the two dispatch destinations with
stop sentinels, so policy failures cannot be mistaken for a successful local or proxy turn.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import yaml

import hermes_constants
from agent import secret_scope as ss
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource

# The contract names this exact process-root key in every denial message.
_DENIAL_MARKER = "gateway.proxy_required"

_UNSET = object()  # "no gateway: key at all" vs. an explicit gateway_section=None (present, null)


class _LocalPrepReached(Exception):
    """Stop sentinel: ``_run_agent_build_turn_context`` was reached (local turn preparation)."""


class _ProxyDispatchReached(Exception):
    """Stop sentinel: ``_run_agent_via_proxy`` was reached (remote dispatch attempted)."""


class _MuteAdapter(BasePlatformAdapter):
    """Minimal real adapter — only what ``_run_agent_display_settings``/``_delivery_adapter_for``
    touch before either stop sentinel fires. A ``send`` call would mean a sentinel failed to stop
    dispatch early, so it is a hard failure, not a silent no-op."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        raise AssertionError("no send expected: a stop sentinel should have fired first")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


@pytest.fixture(autouse=True)
def _reset_multiplex_and_pin():
    ss.set_multiplex_active(False)
    hermes_constants.pin_process_hermes_home(None)
    yield
    ss.set_multiplex_active(False)
    hermes_constants.pin_process_hermes_home(None)


def _stop_sentinels(monkeypatch):
    """Patch both dispatch-entry seams; return (local_calls, proxy_calls) call-count lists."""
    from gateway.run import GatewayRunner

    local_calls: List[bool] = []
    proxy_calls: List[bool] = []

    def _stub_local(self, *args, **kwargs):
        local_calls.append(True)
        raise _LocalPrepReached()

    async def _stub_proxy(self, *args, **kwargs):
        proxy_calls.append(True)
        raise _ProxyDispatchReached()

    monkeypatch.setattr(GatewayRunner, "_run_agent_build_turn_context", _stub_local)
    monkeypatch.setattr(GatewayRunner, "_run_agent_via_proxy", _stub_proxy)
    return local_calls, proxy_calls


def _process_root_home() -> Path:
    """The conftest-redirected ``HERMES_HOME`` this process's OWN launch config lives at."""
    return hermes_constants.get_process_hermes_home()


def _write_config(home: Path, *, raw_yaml: "str | None" = None, gateway_section: Any = _UNSET) -> None:
    config_path = home / "config.yaml"
    if raw_yaml is not None:
        config_path.write_text(raw_yaml, encoding="utf-8")
        return
    if gateway_section is _UNSET:
        config_path.unlink(missing_ok=True)
        return
    config_path.write_text(yaml.safe_dump({"gateway": gateway_section}), encoding="utf-8")


def _write_root_config(monkeypatch, *, raw_yaml: "str | None" = None, gateway_section: Any = _UNSET) -> Path:
    """Write config.yaml at the process-root home and mirror it onto ``gateway.run``'s active home
    (module-level ``_hermes_home`` is frozen at import time, before this test's redirected
    ``HERMES_HOME`` existed — see ``tests/gateway/test_run_progress_topics.py``'s identical seam)."""
    home = _process_root_home()
    _write_config(home, raw_yaml=raw_yaml, gateway_section=gateway_section)
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    return home


def _make_runner() -> Any:
    from gateway.run import GatewayRunner

    adapter = _MuteAdapter()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False, group_sessions_per_user=False, stt_enabled=False,
    )
    return runner


def _source(**overrides) -> SessionSource:
    kwargs = dict(platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="dm", thread_id=None)
    kwargs.update(overrides)
    return SessionSource(**kwargs)


async def _drive(runner, source, **turn_kwargs) -> Dict[str, Any]:
    return await runner._run_agent(
        message="hi", context_prompt="", history=[], source=source,
        session_id="sess-required-proxy", session_key="agent:main:telegram:dm:chat-1",
        **turn_kwargs,
    )


# ── (1) absent / explicit-false root policy: ordinary local turns are unaffected ────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gateway_section",
    [_UNSET, {"proxy_required": False}],
    ids=["absent", "explicit-false"],
)
async def test_absent_or_false_policy_reaches_local_preparation(monkeypatch, gateway_section):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _write_root_config(monkeypatch, gateway_section=gateway_section)
    local_calls, proxy_calls = _stop_sentinels(monkeypatch)
    runner = _make_runner()

    with pytest.raises(_LocalPrepReached):
        await _drive(runner, _source())

    assert local_calls == [True]
    assert proxy_calls == []


# ── (2) required + missing/empty URL: denied before ANY dispatch, fixed safe diagnostic ─────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gateway_section",
    [{"proxy_required": True}, {"proxy_required": True, "proxy_url": ""}],
    ids=["missing-url", "explicit-empty-url"],
)
async def test_required_missing_or_empty_url_denied_before_dispatch(monkeypatch, gateway_section):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _write_root_config(monkeypatch, gateway_section=gateway_section)
    local_calls, proxy_calls = _stop_sentinels(monkeypatch)
    runner = _make_runner()

    result = await _drive(runner, _source())

    assert local_calls == []
    assert proxy_calls == []
    assert result["api_calls"] == 0
    assert result["messages"] == []
    assert _DENIAL_MARKER in result["final_response"]


# ── (3) corrupt / non-mapping root or gateway shape, bad boolean type: fail closed regardless ───
# ── of an otherwise-valid remote URL, with zero effects and no leaked raw content/exception text ─


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_yaml,gateway_section",
    [
        ("- just\n- a\n- list\n", None),
        (None, "oops-not-a-mapping"),
        (None, None),
        (None, {"proxy_required": "true", "proxy_url": "http://127.0.0.1:9/x"}),
        (None, {"proxy_required": 1, "proxy_url": "http://127.0.0.1:9/x"}),
        ("gateway: [unterminated\n", None),
    ],
    ids=[
        "root-not-a-mapping",
        "gateway-not-a-mapping",
        "gateway-explicit-null",
        "proxy-required-string-true",
        "proxy-required-int-one",
        "broken-yaml",
    ],
)
async def test_malformed_root_policy_fails_closed_despite_valid_url(monkeypatch, raw_yaml, gateway_section):
    # A real, otherwise-usable proxy URL via the process env — proves the denial is about the
    # POLICY shape, not about the URL being unusable.
    monkeypatch.setenv("GATEWAY_PROXY_URL", "http://127.0.0.1:9/synthetic-unused")
    _write_root_config(monkeypatch, raw_yaml=raw_yaml, gateway_section=gateway_section)
    local_calls, proxy_calls = _stop_sentinels(monkeypatch)
    runner = _make_runner()

    result = await _drive(runner, _source())

    assert local_calls == []
    assert proxy_calls == []
    assert result["api_calls"] == 0
    assert _DENIAL_MARKER in result["final_response"]
    # Fixed, safe diagnostic only — never the raw parser exception text or file content.
    assert "unterminated" not in result["final_response"]
    assert "oops-not-a-mapping" not in result["final_response"]


# ── (4) real routing transition: a served SIBLING profile can neither tighten nor loosen the ────
# ── process-ROOT policy — root's own value governs every turn, A -> B -> A, scopes restore ──────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "root_required,sibling_required,expect_local_reached",
    [
        (True, False, False),  # sibling tries to weaken a required root: still denied
        (False, True, True),   # sibling tries to tighten a permissive root: root still governs
    ],
    ids=["root-required-defeats-sibling-false", "root-permissive-despite-sibling-true"],
)
async def test_sibling_profile_cannot_override_process_root_policy_a_b_a(
    monkeypatch, tmp_path, root_required, sibling_required, expect_local_reached,
):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    root = tmp_path / "root"
    root.mkdir(parents=True)
    sibling = root / "profiles" / "sibling"
    sibling.mkdir(parents=True)
    _write_config(root, gateway_section={"proxy_required": root_required})
    _write_config(sibling, gateway_section={"proxy_required": sibling_required})

    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", root)
    hermes_constants.pin_process_hermes_home(str(root))
    ss.set_multiplex_active(True)

    local_calls, proxy_calls = _stop_sentinels(monkeypatch)
    runner = _make_runner()
    runner.config.multiplex_profiles = True
    source_a = _source(profile="default")
    source_b = _source(profile="sibling")

    async def _step(source, live_home):
        local_calls.clear()
        # The hosted profile is also the live environment while its turn is running.  The pinned
        # root remains the policy owner; the runtime scope must restore this value on each return.
        monkeypatch.setenv("HERMES_HOME", str(live_home))
        if expect_local_reached:
            with pytest.raises(_LocalPrepReached):
                await _drive(runner, source)
            assert local_calls == [True]
        else:
            result = await _drive(runner, source)
            assert local_calls == []
            assert _DENIAL_MARKER in result["final_response"]

    await _step(source_a, root)
    await _step(source_b, sibling)
    await _step(source_a, root)

    assert proxy_calls == []
    assert os.environ["HERMES_HOME"] == str(root)
    # Restoration: an unscoped read of the root config afterward is unchanged (no mutation, no
    # leaked override) — the same file this whole cycle's decisions were supposed to key on.
    assert yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))["gateway"]["proxy_required"] == root_required


# ── (5) required root still denies malformed served-profile URLs, without leaking resolver errors ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "proxy_url",
    [123, ["http://127.0.0.1:9/synthetic-list-url"]],
    ids=["integer-url", "list-url"],
)
async def test_required_root_denies_real_served_profile_invalid_url(monkeypatch, tmp_path, proxy_url):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    root = tmp_path / "root"
    root.mkdir(parents=True)
    sibling = root / "profiles" / "sibling"
    sibling.mkdir(parents=True)
    _write_config(root, gateway_section={"proxy_required": True})
    _write_config(sibling, gateway_section={"proxy_url": proxy_url})
    monkeypatch.setenv("HERMES_HOME", str(sibling))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", root)
    hermes_constants.pin_process_hermes_home(str(root))
    ss.set_multiplex_active(True)

    local_calls, proxy_calls = _stop_sentinels(monkeypatch)
    runner = _make_runner()
    runner.config.multiplex_profiles = True

    result = await _drive(runner, _source(profile="sibling"))

    assert local_calls == []
    assert proxy_calls == []
    assert result["api_calls"] == 0
    assert result["final_response"] == (
        "⚠️ gateway.proxy_required could not be resolved safely — refusing this turn."
    )
    assert "AttributeError" not in result["final_response"]


@pytest.mark.asyncio
async def test_required_root_denies_strict_validation_config_race(monkeypatch):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    root = _write_root_config(monkeypatch, gateway_section={"proxy_required": True})
    import gateway.proxy_admission as proxy_admission

    real_validator = proxy_admission.require_readable_config_before_write
    fault_runs = []

    def _validate_then_one_shot_fixture_fault(config_path):
        validated = real_validator(config_path)
        # EXPLICIT ONE-SHOT FIXTURE FAULT: race the real read with nonempty list content.
        if not fault_runs:
            fault_runs.append(True)
            config_path.write_text(yaml.safe_dump(["synthetic-race-list-marker"]), encoding="utf-8")
        return validated

    monkeypatch.setattr(
        proxy_admission,
        "require_readable_config_before_write",
        _validate_then_one_shot_fixture_fault,
    )
    local_calls, proxy_calls = _stop_sentinels(monkeypatch)
    runner = _make_runner()

    result = await _drive(runner, _source())

    assert fault_runs == [True]
    assert local_calls == []
    assert proxy_calls == []
    assert result["api_calls"] == 0
    assert _DENIAL_MARKER in result["final_response"]
    assert "synthetic-race-list-marker" not in result["final_response"]
    assert yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) == [
        "synthetic-race-list-marker"
    ]


@pytest.mark.asyncio
async def test_required_root_denies_proxy_url_fault_without_raw_exception(monkeypatch):
    _write_root_config(monkeypatch, gateway_section={"proxy_required": True})
    local_calls, proxy_calls = _stop_sentinels(monkeypatch)
    runner = _make_runner()
    marker = "synthetic-private-marker"

    def _faulting_proxy_url():
        raise RuntimeError(marker)

    runner._get_proxy_url = _faulting_proxy_url
    result = await _drive(runner, _source())

    assert local_calls == []
    assert proxy_calls == []
    assert result["api_calls"] == 0
    assert result["final_response"] == (
        "⚠️ gateway.proxy_required could not be resolved safely — refusing this turn."
    )
    assert marker not in result["final_response"]


@pytest.mark.asyncio
async def test_required_proxy_native_entry_denies_second_resolution_config_fault(monkeypatch):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    root = _write_root_config(
        monkeypatch,
        gateway_section={
            "proxy_required": True,
            "proxy_url": "http://127.0.0.1:9/synthetic-unused",
        },
    )
    import aiohttp
    from gateway.run import GatewayRunner

    real_proxy = next(
        base.__dict__["_run_agent_via_proxy"]
        for base in GatewayRunner.__mro__
        if "_run_agent_via_proxy" in base.__dict__
    )
    local_calls, proxy_calls = _stop_sentinels(monkeypatch)
    # Keep the real native proxy entry; only local preparation remains a stop sentinel.
    monkeypatch.setattr(GatewayRunner, "_run_agent_via_proxy", real_proxy)
    http_calls = []

    def _unexpected_http_creation(*args, **kwargs):
        http_calls.append(True)
        raise AssertionError("HTTP creation must not be reached after the second resolver fault")

    monkeypatch.setattr(aiohttp, "ClientSession", _unexpected_http_creation)
    runner = _make_runner()
    resolver_calls = []
    second_resolution_fault = []
    real_resolver = runner._get_proxy_url

    def _resolve_with_one_shot_config_fault():
        resolver_calls.append(len(resolver_calls) + 1)
        if len(resolver_calls) == 1:
            resolved = real_resolver()
            # Real-file mutation after the first real resolution: only this synthetic URL changes.
            _write_config(root, gateway_section={"proxy_required": True, "proxy_url": 17})
            return resolved
        try:
            return real_resolver()
        except Exception as exc:
            second_resolution_fault.append(exc)
            raise

    runner._get_proxy_url = _resolve_with_one_shot_config_fault
    result = await _drive(runner, _source())

    assert resolver_calls == [1, 2]
    assert second_resolution_fault
    assert local_calls == []
    assert proxy_calls == []
    assert http_calls == []
    assert result["api_calls"] == 0
    assert result["messages"] == []
    assert result["final_response"] == (
        "⚠️ gateway.proxy_required could not be resolved safely — refusing this turn."
    )
    assert "synthetic" not in result["final_response"]
    assert "AttributeError" not in result["final_response"]


# ── (6) the direct reader owns the root authority, then restores the served scope exactly ───────


def test_gateway_proxy_required_reads_root_marker_and_restores_served_scope(monkeypatch, tmp_path):
    import gateway.proxy_admission as proxy_admission
    from gateway.proxy_admission import ProxyPolicyError, gateway_proxy_required

    root = tmp_path / "root"
    root.mkdir()
    _write_config(
        root,
        raw_yaml=(
            "gateway:\n"
            "  proxy_required: true\n"
            "  marker: ${_PROXY_TEST_SENTINEL}\n"
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("_PROXY_TEST_SENTINEL", "root-value")
    hermes_constants.pin_process_hermes_home(str(root))
    ss.set_multiplex_active(True)

    captured = []
    real_effective = proxy_admission._effective

    def _capture_effective(raw):
        effective = real_effective(raw)
        captured.append((effective["gateway"]["marker"], effective["gateway"]["proxy_required"]))
        return effective

    monkeypatch.setattr(proxy_admission, "_effective", _capture_effective)
    outer_token = ss.set_secret_scope({"_PROXY_TEST_SENTINEL": "profile-value"})
    try:
        assert gateway_proxy_required() is True
        # These assertions deliberately run before resetting the outer served-profile scope.
        assert captured[0][0] == "root-value"
        assert captured[0][1] is True
        assert ss.get_secret("_PROXY_TEST_SENTINEL") == "profile-value"

        def _throwing_effective(raw):
            raise RuntimeError("synthetic-private-marker")

        monkeypatch.setattr(proxy_admission, "_effective", _throwing_effective)
        with pytest.raises(ProxyPolicyError):
            gateway_proxy_required()
        assert ss.get_secret("_PROXY_TEST_SENTINEL") == "profile-value"
    finally:
        ss.reset_secret_scope(outer_token)
