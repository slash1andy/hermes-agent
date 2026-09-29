"""Bot Chat delivery admission — independent review found the tick's queue-drain bypasses the
transport-process execution guard: ``scheduler_tick.py`` calls ``bot_chat_delivery.drain()`` before
the ``cron.execution_enabled`` check, and for an unowned delivery target, drain's own
``_deliver_to_bot_chat`` falls through to a genuine ``hermes chat -Q`` CLI turn under ANOTHER
profile's credentials (cron/scheduler_delivery.py) — not a pure outbound queue. A live-owner
handoff (message injection into an already-running session) is not local execution and must stay
available regardless of this process's policy.

These tests drive the real ``cron.bot_chat_delivery``/``cron.scheduler_delivery`` entry points. The
only mocks are the final CLI-spawn seam (``_run_bot_chat_turn``) and, for the live-owner case,
owner discovery/delivery (``tools.bot_live_delivery``) — no fabricated helper, no real model call.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import yaml


def _write_root_config(home, cron_section=None):
    home.mkdir(parents=True, exist_ok=True)
    if cron_section is not None:
        (home / "config.yaml").write_text(yaml.safe_dump({"cron": cron_section}), encoding="utf-8")


@pytest.fixture
def root_home(tmp_path, monkeypatch):
    home = tmp_path / "root"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _target_home(tmp_path, name="target"):
    from hermes_state import SessionDB

    home = tmp_path / "profiles" / name
    home.mkdir(parents=True)
    SessionDB(db_path=home / "state.db").close()  # real, empty, no owner registered
    return home


def _turn_stub(monkeypatch):
    import cron.scheduler_delivery as sched_delivery

    stub = Mock(return_value=Mock(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(sched_delivery, "_run_bot_chat_turn", stub)
    return stub


# ── (1) direct unowned delivery: denied must refuse before the CLI spawn ────────────────────────


def test_deliver_to_bot_chat_denied_never_spawns_cli_for_unowned_target(
        root_home, tmp_path, monkeypatch):
    from cron.scheduler_delivery import BOT_CHAT_EXECUTION_DENIED_MARKER, _deliver_to_bot_chat
    import tools.environments.local as local_env

    target = _target_home(tmp_path)
    _write_root_config(root_home, {"execution_enabled": False})
    turn = _turn_stub(monkeypatch)
    env_spy = Mock(side_effect=AssertionError("must not build a delivery-target env while denied"))
    monkeypatch.setattr(local_env, "served_profile_child_env", env_spy)

    error = _deliver_to_bot_chat({"id": "job-1", "name": "watchdog"}, "content", "", deferred=None)

    turn.assert_not_called()
    env_spy.assert_not_called()
    # Exact sentinel, not a substring/parsed message: _drain relies on identity to know nothing
    # was ever attempted (never infers "not started" from arbitrary error text).
    assert error == BOT_CHAT_EXECUTION_DENIED_MARKER


# ── (2) drain of a persisted unowned record: denied must not claim/terminalize it ────────────────


def test_bot_chat_drain_leaves_unowned_record_queued_when_denied_then_delivers_once_enabled(
        root_home, tmp_path, monkeypatch):
    import cron.bot_chat_delivery as bot_chat

    target = _target_home(tmp_path)
    job = {"id": "job-2", "name": "watchdog"}
    key = "2" * 64
    turn = _turn_stub(monkeypatch)

    _write_root_config(root_home, {"execution_enabled": False})
    record = bot_chat.defer(key, job, "original content", "", target)
    assert record["status"] == "queued"

    bot_chat.drain()

    turn.assert_not_called()
    denied_record = bot_chat.read_pending(key)
    # A policy refusal is not a delivery attempt's outcome: the record must stay queued/retryable,
    # never "claimed" (in flight) or "ambiguous" (an unknown-outcome terminal) — those states mean
    # something was actually attempted, which never happened here.
    assert denied_record["status"] == "queued"
    assert denied_record["content"] == "original content"

    _write_root_config(root_home, {"execution_enabled": True})
    bot_chat.drain()

    turn.assert_called_once()
    settled_record = bot_chat.read_pending(key)
    assert settled_record["status"] == "settled"


# ── (3) live-owner handoff is not local execution: must remain available when denied ────────────


def test_bot_chat_live_owner_handoff_remains_available_when_denied(
        root_home, tmp_path, monkeypatch):
    from cron.scheduler_delivery import _deliver_to_bot_chat
    import tools.bot_live_delivery as bot_live

    target = _target_home(tmp_path)
    _write_root_config(root_home, {"execution_enabled": False})
    turn = _turn_stub(monkeypatch)
    monkeypatch.setattr(bot_live, "find_canonical_live_owner", Mock(return_value={
        "profile_home": str(target), "session_id": "s1", "lease_id": "l1",
        "live_session_id": "live1"}))

    def _fake_deliver(home, owner, message, *, delivery_id=None, **kwargs):
        return {"message": message, "delivery_id": delivery_id,
                "notification_category": kwargs.get("notification_category", "result"),
                "status": "settled"}

    monkeypatch.setattr(bot_live, "deliver_to_live_owner", _fake_deliver)

    error = _deliver_to_bot_chat({"id": "job-3", "name": "watchdog"}, "content", "", deferred=None)

    turn.assert_not_called()  # handed off via the live owner's own session, no local agent/CLI
    assert error is None


# ── (4) live owner disappears between _drain's pre-claim check and the delivery attempt ──────────


def test_bot_chat_live_owner_disappearing_race_stays_queued_when_denied(
        root_home, tmp_path, monkeypatch):
    """_drain's cheap pre-check saw a live owner (so it let a denied process claim the record —
    live-owner handoffs are always allowed); by the time _deliver_to_bot_chat re-checks, the owner
    is gone. The deep CLI-fallback guard, not the pre-check, must be what closes this race."""
    import cron.bot_chat_delivery as bot_chat
    import tools.bot_live_delivery as bot_live

    target = _target_home(tmp_path)
    job = {"id": "job-4", "name": "watchdog"}
    key = "4" * 64
    turn = _turn_stub(monkeypatch)
    live_owner = {"profile_home": str(target), "session_id": "s1", "lease_id": "l1",
                  "live_session_id": "live1"}
    monkeypatch.setattr(bot_live, "find_canonical_owner", Mock(return_value=live_owner))
    # First call (_drain's pre-check): still live. Second call (_deliver_to_bot_chat's own
    # re-check): gone.
    monkeypatch.setattr(
        bot_live, "find_canonical_live_owner", Mock(side_effect=[live_owner, None]))

    _write_root_config(root_home, {"execution_enabled": False})
    record = bot_chat.defer(key, job, "original content", "", target)
    assert record["status"] == "queued"

    bot_chat.drain()

    turn.assert_not_called()
    denied_record = bot_chat.read_pending(key)
    assert denied_record["status"] == "queued"
    assert denied_record["content"] == "original content"

    monkeypatch.setattr(bot_live, "find_canonical_owner", Mock(return_value=None))
    monkeypatch.setattr(bot_live, "find_canonical_live_owner", Mock(return_value=None))
    _write_root_config(root_home, {"execution_enabled": True})
    bot_chat.drain()

    turn.assert_called_once()
    settled_record = bot_chat.read_pending(key)
    assert settled_record["status"] == "settled"
