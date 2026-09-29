"""Required-proxy admission for out-of-turn session maintenance (auto hygiene, manual /compress).

Mirrors the stop-sentinel / real-root-reader style of
``tests/gateway/test_required_proxy_admission.py``: the process-root policy reader
(``gateway/proxy_admission.py::gateway_proxy_required``) and the real ``GatewayRunner`` maintenance
methods run unmodified; only the final local ``AIAgent`` construction / provider-model resolution
is a labelled fake seam.
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import hermes_constants
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.proxy_admission import ProxyPolicyError
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource

_DENIAL_MARKER = "gateway.proxy_required"
_UNSET = object()
_REQUIRED_VARIANTS = [
    {"proxy_required": True},
    {"proxy_required": True, "proxy_url": "http://127.0.0.1:9/x"},
    "oops-not-a-mapping",
]
_REQUIRED_IDS = ["required-no-url", "required-with-url", "malformed-root"]
_PERMISSIVE_VARIANTS = [_UNSET, {"proxy_required": False}]
_PERMISSIVE_IDS = ["absent", "explicit-false"]


def _write_root_config(monkeypatch, *, gateway_section=_UNSET, extra=None):
    home = hermes_constants.get_process_hermes_home()
    cfg = dict(extra or {})
    if gateway_section is not _UNSET:
        cfg["gateway"] = gateway_section
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    return home


def _install_fake_run_agent(monkeypatch, agent_cls):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


class _CountingHygieneAgent:
    instances = 0

    def __init__(self, **kwargs):
        type(self).instances += 1
        self.session_id = kwargs.get("session_id", "x")
        self._session_db = kwargs.get("session_db")
        self._last_compaction_in_place = False
        self.context_compressor = SimpleNamespace(
            bind_session_state=MagicMock(), _last_compress_aborted=False, _last_aux_model_failure_model=None,
        )
        self.shutdown_memory_provider = MagicMock()
        self.close = MagicMock()

    def _compress_context(self, messages, *_a, **_k):
        self.session_id = f"{self.session_id}_compressed"
        return ([{"role": "assistant", "content": "compressed"}], None)


def _overlimit_history(n=20):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}" * 20, "timestamp": f"t{i}"}
        for i in range(n)
    ]


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _make_hygiene_runner(monkeypatch, agent_cls, history):
    from hermes_state import AsyncSessionDB, SessionDB
    import gateway.run as gateway_run

    home = hermes_constants.get_process_hermes_home()
    db = SessionDB(db_path=home / "state.db")
    session_id = "sess-maint"
    db.create_session(session_id, "telegram")
    _install_fake_run_agent(monkeypatch, agent_cls)

    adapter = _Adapter()
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")})
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    session_entry = SessionEntry(
        session_key="agent:main:telegram:dm:c1", session_id=session_id,
        created_at=datetime.now(), updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm",
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.get_model_override.return_value = None
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = AsyncSessionDB(db)
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_k: 100)
    event = MessageEvent(
        text="hello", source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1"),
        message_id="1",
    )
    return runner, session_entry, event, db


async def _drive_hygiene(runner, session_entry, event, history):
    return await runner._hmwa_run_session_hygiene(
        event, event.source, session_entry, session_entry.session_key, history, "quick-key", 1,
    )


# ── auto hygiene: required-true (with/without a configured URL) and a malformed root all fail ──
# ── closed, returning the EXACT input history object with zero local agent construction ─────────

@pytest.mark.asyncio
@pytest.mark.parametrize("gateway_section", _REQUIRED_VARIANTS, ids=_REQUIRED_IDS)
async def test_auto_hygiene_denies_and_preserves_full_history(monkeypatch, gateway_section):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _CountingHygieneAgent.instances = 0
    history = _overlimit_history()
    runner, session_entry, event, db = _make_hygiene_runner(monkeypatch, _CountingHygieneAgent, history)
    _write_root_config(
        monkeypatch, gateway_section=gateway_section,
        extra={"compression": {"enabled": True, "hygiene_hard_message_limit": 10}},
    )
    try:
        result = await _drive_hygiene(runner, session_entry, event, history)
        assert result is history, "denial must return the exact original history, never a bound/truncated copy"
        assert _CountingHygieneAgent.instances == 0, "required proxy must block local hygiene agent construction"
        runner.session_store.rewrite_transcript.assert_not_called()
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway_section", _PERMISSIVE_VARIANTS, ids=_PERMISSIVE_IDS)
async def test_auto_hygiene_reaches_real_handlers_when_not_required(monkeypatch, gateway_section):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _CountingHygieneAgent.instances = 0
    history = _overlimit_history()
    runner, session_entry, event, db = _make_hygiene_runner(monkeypatch, _CountingHygieneAgent, history)
    _write_root_config(
        monkeypatch, gateway_section=gateway_section,
        extra={"compression": {"enabled": True, "hygiene_hard_message_limit": 10}},
    )
    try:
        result = await _drive_hygiene(runner, session_entry, event, history)
        assert _CountingHygieneAgent.instances == 1, "a permissive policy must still reach real local construction"
        assert result is not history
    finally:
        db.close()


@pytest.mark.asyncio
async def test_auto_hygiene_denies_on_late_flip_before_construction(monkeypatch):
    """The root config flips permissive -> required between the settings read and local
    construction (a race, not a static value); the guard must re-check, not trust a stale answer."""
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _CountingHygieneAgent.instances = 0
    history = _overlimit_history()
    runner, session_entry, event, db = _make_hygiene_runner(monkeypatch, _CountingHygieneAgent, history)
    home = _write_root_config(
        monkeypatch, gateway_section={"proxy_required": False},
        extra={"compression": {"enabled": True, "hygiene_hard_message_limit": 10}},
    )
    import gateway.proxy_admission as proxy_admission
    real_validator = proxy_admission.require_readable_config_before_write
    reads: list = []

    def _flip_after_first_read(config_path):
        validated = real_validator(config_path)
        reads.append(True)
        if len(reads) == 1:
            data = dict(validated) if isinstance(validated, dict) else {}
            data["gateway"] = {"proxy_required": True}
            data.setdefault("compression", {"enabled": True, "hygiene_hard_message_limit": 10})
            (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        return validated

    monkeypatch.setattr(proxy_admission, "require_readable_config_before_write", _flip_after_first_read)
    try:
        result = await _drive_hygiene(runner, session_entry, event, history)
        assert result is history, "a late flip must still block local construction and return the full history"
        assert _CountingHygieneAgent.instances == 0
        runner.session_store.rewrite_transcript.assert_not_called()
        assert len(reads) >= 2, "a guard checked only once at the top cannot catch a mid-turn flip"
    finally:
        db.close()


# ── manual /compress: same policy, a fixed refusal instead of a silent fail-through ──────────────

def _manual_history():
    return [
        {"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"}, {"role": "assistant", "content": "four"},
    ]


def _make_manual_gw(monkeypatch):
    from gateway.run import GatewayRunner
    gw = GatewayRunner.__new__(GatewayRunner)
    history = _manual_history()
    entry = MagicMock(session_id="sid", session_key="agent:main:telegram:dm:c1")
    gw.session_store = MagicMock()
    gw.session_store.get_model_override.return_value = None
    gw._async_session_store = MagicMock()
    gw._async_session_store._store = gw.session_store
    gw._async_session_store.get_or_create_session = AsyncMock(return_value=entry)
    gw._async_session_store.load_transcript = AsyncMock(return_value=history)
    gw._async_session_store.update_session = AsyncMock(return_value=None)
    gw._async_session_store.rewrite_transcript = AsyncMock(return_value=True)
    gw._async_session_store._save = AsyncMock(return_value=None)
    gw._session_db = SimpleNamespace(_db=object(), get_session=AsyncMock(return_value=None))

    async def _run_in_executor_with_context(fn):
        return await asyncio.get_running_loop().run_in_executor(None, fn)

    gw._run_in_executor_with_context = _run_in_executor_with_context
    gw._session_key_for_source = lambda source: entry.session_key
    gw._sync_telegram_topic_binding = lambda *_a, **_k: None
    gw._evict_cached_agent = MagicMock()
    gw._cleanup_agent_resources_off_loop = AsyncMock(return_value=None)
    event = MagicMock()
    event.source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", thread_id=None)
    return gw, event, history


class _ManualCompressionAgent:
    constructed = 0
    summaries = 0

    def __init__(self, **kwargs):
        type(self).constructed += 1
        self.session_id = kwargs.get("session_id", "sid") + "_compressed"
        self.tools = None
        self._cached_system_prompt = ""
        self._context_window = self.context_window = self.max_context_tokens = 100
        self._compression_skipped_due_to_lock = None
        self._last_compaction_in_place = False
        self.context_compressor = SimpleNamespace(
            has_content_to_compress=lambda _messages: True,
            _last_compress_aborted=False,
            _last_aux_model_failure_model=None,
            _last_compress_refused_would_grow=False,
            _last_summary_fallback_used=False,
            _last_summary_error=None,
        )
        self.shutdown_memory_provider = MagicMock()
        self.close = MagicMock()

    def _compress_context(self, messages, *_args, **_kwargs):
        type(self).summaries += 1
        return ([{"role": "assistant", "content": "manual summary"}], "")


def _install_manual_agent(monkeypatch):
    _ManualCompressionAgent.constructed = 0
    _ManualCompressionAgent.summaries = 0
    _install_fake_run_agent(monkeypatch, _ManualCompressionAgent)


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway_section", _REQUIRED_VARIANTS, ids=_REQUIRED_IDS)
async def test_manual_compress_denies_before_run_manual_compression(monkeypatch, gateway_section):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _write_root_config(monkeypatch, gateway_section=gateway_section)
    _install_manual_agent(monkeypatch)
    gw, event, _history = _make_manual_gw(monkeypatch)
    event.get_command_args.return_value = ""

    resolve_calls = []

    def _resolve_session_agent_runtime(**_kwargs):
        resolve_calls.append(True)
        return "test/model", {"api_key": "fake-key", "provider": "test"}

    gw._resolve_session_agent_runtime = _resolve_session_agent_runtime
    reply = await gw._handle_compress_command_inner(event)
    assert resolve_calls == [], "required proxy must stop before provider resolution"
    assert _ManualCompressionAgent.constructed == 0
    assert _ManualCompressionAgent.summaries == 0
    assert not getattr(gw._async_session_store.rewrite_transcript, "await_count", 0)
    assert not getattr(gw._async_session_store.update_session, "await_count", 0)
    assert _DENIAL_MARKER in reply


@pytest.mark.asyncio
async def test_manual_compress_preview_remains_valid_under_required_proxy(monkeypatch):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _write_root_config(monkeypatch, gateway_section={"proxy_required": True})
    _install_manual_agent(monkeypatch)
    gw, event, _history = _make_manual_gw(monkeypatch)
    event.get_command_args.return_value = "--preview"
    reply = await gw._handle_compress_command_inner(event)
    assert _ManualCompressionAgent.constructed == 0
    assert _ManualCompressionAgent.summaries == 0
    assert not getattr(gw._async_session_store.rewrite_transcript, "await_count", 0)
    assert not getattr(gw._async_session_store.update_session, "await_count", 0)
    assert "Preview" in reply


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway_section", _PERMISSIVE_VARIANTS, ids=_PERMISSIVE_IDS)
async def test_manual_compress_reaches_real_handlers_when_not_required(monkeypatch, gateway_section):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _write_root_config(monkeypatch, gateway_section=gateway_section)
    _install_manual_agent(monkeypatch)
    gw, event, _history = _make_manual_gw(monkeypatch)
    event.get_command_args.return_value = ""
    gw._resolve_session_agent_runtime = lambda **_k: ("test/model", {"api_key": "fake-key", "provider": "test"})
    gw._resolve_session_reasoning_config = lambda **_k: None

    await gw._handle_compress_command_inner(event)
    assert _ManualCompressionAgent.summaries == 1, "the real handler must invoke the deterministic fake summary"
    gw._async_session_store.update_session.assert_awaited()


@pytest.mark.asyncio
async def test_auto_hygiene_late_flip_with_disabled_compression_preserves_identity(monkeypatch):
    """Fixture keeps the native settings read, but disables compression so the late policy flip
    must still be caught before the no-compression payload path can bound the transcript."""
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _CountingHygieneAgent.instances = 0
    history = _overlimit_history()
    runner, session_entry, event, db = _make_hygiene_runner(monkeypatch, _CountingHygieneAgent, history)
    home = _write_root_config(
        monkeypatch, gateway_section={"proxy_required": False},
        extra={"compression": {"enabled": False, "hygiene_hard_message_limit": 10}},
    )
    import gateway.proxy_admission as proxy_admission
    import gateway.run_turn as run_turn
    real_validator = proxy_admission.require_readable_config_before_write
    reads = []
    bound_calls = []
    real_bound = run_turn.GatewayTurnMixin._bound_hygiene_payload

    def _flip_after_first_read(config_path):
        validated = real_validator(config_path)
        reads.append(True)
        if len(reads) == 1:
            data = dict(validated)
            data["gateway"] = {"proxy_required": True}
            (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        return validated

    def _bound_spy(*args, **kwargs):
        bound_calls.append(True)
        return real_bound(*args, **kwargs)

    monkeypatch.setattr(proxy_admission, "require_readable_config_before_write", _flip_after_first_read)
    monkeypatch.setattr(run_turn.GatewayTurnMixin, "_bound_hygiene_payload", staticmethod(_bound_spy))
    try:
        result = await _drive_hygiene(runner, session_entry, event, history)
        assert result is history
        assert bound_calls == []
        assert _CountingHygieneAgent.instances == 0
        runner.session_store.rewrite_transcript.assert_not_called()
        assert len(reads) >= 2
    finally:
        db.close()


@pytest.mark.asyncio
async def test_auto_hygiene_late_flip_on_real_no_compress_plan_preserves_identity(monkeypatch):
    """A late root flip after the real plan still blocks the post-plan maintenance boundary."""
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _CountingHygieneAgent.instances = 0
    history = _overlimit_history()
    runner, session_entry, event, db = _make_hygiene_runner(monkeypatch, _CountingHygieneAgent, history)
    home = _write_root_config(
        monkeypatch, gateway_section={"proxy_required": False},
        extra={"compression": {"enabled": True, "hygiene_hard_message_limit": 10}},
    )
    import gateway.proxy_admission as proxy_admission
    real_validator = proxy_admission.require_readable_config_before_write
    reads = []
    real_plan = runner._hmwa_hygiene_plan
    plan_reached = False

    def _policy_read_spy(config_path):
        validated = real_validator(config_path)
        reads.append(True)
        return validated

    async def _real_plan_then_cooldown(*args, **kwargs):
        nonlocal plan_reached
        plan = await real_plan(*args, **kwargs)
        plan.needs_compress = False
        data = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        data["gateway"] = {"proxy_required": True}
        (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        plan_reached = True
        return plan

    monkeypatch.setattr(proxy_admission, "require_readable_config_before_write", _policy_read_spy)
    monkeypatch.setattr(runner, "_hmwa_hygiene_plan", _real_plan_then_cooldown)
    try:
        result = await _drive_hygiene(runner, session_entry, event, history)
        assert result is history
        assert plan_reached
        assert _CountingHygieneAgent.instances == 0
        runner.session_store.rewrite_transcript.assert_not_called()
        assert not getattr(runner.session_store.append_to_transcript, "call_count", 0)
        assert not getattr(runner.session_store.update_session, "call_count", 0)
        assert len(reads) >= 3, "the root policy must be rechecked after the real no-compression plan"
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("root_required", "served_required", "should_construct"),
    [(True, False, False), (False, True, True)],
    ids=["root-required-served-permissive", "root-permissive-served-required"],
)
async def test_maintenance_policy_is_root_pinned_across_served_profile_scope(
    monkeypatch, tmp_path, root_required, served_required, should_construct,
):
    """The served profile's native runtime scope cannot replace the pinned process-root policy."""
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    root = tmp_path / "root"
    served = tmp_path / "served"
    root.mkdir(); served.mkdir()
    (served / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"proxy_required": served_required}}), encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    hermes_constants.pin_process_hermes_home(str(root))
    _CountingHygieneAgent.instances = 0
    history = _overlimit_history()
    try:
        _write_root_config(
            monkeypatch, gateway_section={"proxy_required": root_required},
            extra={"compression": {"enabled": True, "hygiene_hard_message_limit": 10}},
        )
        runner, session_entry, event, db = _make_hygiene_runner(monkeypatch, _CountingHygieneAgent, history)
        try:
            from gateway.run import _profile_runtime_scope
            with _profile_runtime_scope(served, prepared_secret_scope={}):
                result = await _drive_hygiene(runner, session_entry, event, history)
            assert (result is not history) is should_construct
            assert _CountingHygieneAgent.instances == int(should_construct)
            if not should_construct:
                runner.session_store.rewrite_transcript.assert_not_called()
        finally:
            db.close()
    finally:
        hermes_constants.pin_process_hermes_home(None)


@pytest.mark.asyncio
async def test_manual_compress_late_flip_blocks_construction_after_provider_resolution(monkeypatch):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _write_root_config(monkeypatch, gateway_section={"proxy_required": False})
    _install_manual_agent(monkeypatch)
    gw, event, history = _make_manual_gw(monkeypatch)
    event.get_command_args.return_value = ""
    import gateway.proxy_admission as proxy_admission
    home = hermes_constants.get_process_hermes_home()
    real_validator = proxy_admission.require_readable_config_before_write
    reads = []

    def _flip_after_first_read(config_path):
        validated = real_validator(config_path)
        reads.append(True)
        if len(reads) == 1:
            (home / "config.yaml").write_text(
                yaml.safe_dump({"gateway": {"proxy_required": True}}), encoding="utf-8",
            )
        return validated

    resolve_calls = []
    gw._resolve_session_agent_runtime = lambda **kwargs: (
        resolve_calls.append(True) or ("test/model", {"api_key": "fake-key", "provider": "test"})
    )
    monkeypatch.setattr(proxy_admission, "require_readable_config_before_write", _flip_after_first_read)
    reply = await gw._handle_compress_command_inner(event)
    assert len(reads) >= 2
    assert resolve_calls == [True]
    assert _ManualCompressionAgent.constructed == 0
    assert _ManualCompressionAgent.summaries == 0
    assert history == _manual_history()
    assert not getattr(gw._async_session_store.rewrite_transcript, "await_count", 0)
    assert not getattr(gw._async_session_store.update_session, "await_count", 0)
    assert _DENIAL_MARKER in reply


@pytest.mark.asyncio
async def test_required_root_blocks_actual_hygiene_codex_compaction_boundary(monkeypatch):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _write_root_config(monkeypatch, gateway_section={"proxy_required": True})
    history = _overlimit_history()
    runner, session_entry, _event, db = _make_hygiene_runner(monkeypatch, _CountingHygieneAgent, history)
    native_calls = []
    runner._agent_cache = {
        session_entry.session_key: (
            SimpleNamespace(
                _codex_session=SimpleNamespace(compact_thread=lambda: native_calls.append(True)),
                context_compressor=SimpleNamespace(compression_count=0),
            ),
            0.0,
        ),
    }
    hs = SimpleNamespace(
        data={"compression": {"codex_app_server_auto": "native"}},
        total_ceiling_seconds=30.0,
    )
    plan = SimpleNamespace(approx_tokens=345_000)
    try:
        with pytest.raises(ProxyPolicyError, match=_DENIAL_MARKER):
            await runner._hmwa_hygiene_codex_compaction(
                hs, plan, history, session_entry, session_entry.session_key,
                {"api_mode": "codex_app_server"},
            )
        assert native_calls == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_manual_compress_cached_codex_late_flip_blocks_cached_call(monkeypatch):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _write_root_config(monkeypatch, gateway_section={"proxy_required": False})
    _install_manual_agent(monkeypatch)
    gw, event, history = _make_manual_gw(monkeypatch)
    event.get_command_args.return_value = ""
    cached_calls = []
    cached = SimpleNamespace(
        _codex_session=object(),
        context_compressor=SimpleNamespace(compression_count=0),
        _compress_context=lambda *args, **kwargs: cached_calls.append(True),
    )
    gw._cached_agent_for = lambda *args, **kwargs: cached
    gw._resolve_session_agent_runtime = lambda **kwargs: (
        "test/model", {"api_key": "fake-key", "provider": "test", "api_mode": "codex_app_server"}
    )
    import gateway.proxy_admission as proxy_admission
    home = hermes_constants.get_process_hermes_home()
    real_validator = proxy_admission.require_readable_config_before_write
    reads = []

    def _flip_after_first_read(config_path):
        validated = real_validator(config_path)
        reads.append(True)
        if len(reads) == 1:
            (home / "config.yaml").write_text(
                yaml.safe_dump({"gateway": {"proxy_required": True}}), encoding="utf-8",
            )
        return validated

    monkeypatch.setattr(proxy_admission, "require_readable_config_before_write", _flip_after_first_read)
    reply = await gw._handle_compress_command_inner(event)
    assert len(reads) >= 2
    assert cached_calls == []
    assert history == _manual_history()
    assert not getattr(gw._async_session_store.rewrite_transcript, "await_count", 0)
    assert not getattr(gw._async_session_store.update_session, "await_count", 0)
    assert _DENIAL_MARKER in reply


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["hygiene", "manual"], ids=["auto-builder", "manual-builder"])
async def test_required_root_blocks_actual_maintenance_builders_before_agent_construct(monkeypatch, builder):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    _write_root_config(monkeypatch, gateway_section={"proxy_required": True})
    _CountingHygieneAgent.instances = 0
    _install_manual_agent(monkeypatch)
    if builder == "hygiene":
        history = _overlimit_history()
        runner, session_entry, _event, db = _make_hygiene_runner(monkeypatch, _CountingHygieneAgent, history)
        try:
            with pytest.raises(ProxyPolicyError, match=_DENIAL_MARKER):
                await runner._hmwa_hygiene_build_agent("test/model", {"api_key": "fake"}, session_entry)
        finally:
            db.close()
        assert _CountingHygieneAgent.instances == 0
    else:
        gw, _event, _history = _make_manual_gw(monkeypatch)
        with pytest.raises(ProxyPolicyError, match=_DENIAL_MARKER):
            await gw._build_manual_compression_agent("sid", "test/model", {"api_key": "fake"})
        assert _ManualCompressionAgent.constructed == 0
