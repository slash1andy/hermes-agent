"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    # decompose_task now routes through call_llm (see #35566) — mock it at
    # the source module so task config, extra_body, and retries stay out of
    # unit-test scope.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist. The decomposer uses
    profiles_mod.list_profiles() to build the roster + valid-set, and
    profiles_mod.profile_exists() to resolve orchestrator/default."""
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def test_decompose_with_fanout_creates_children(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"


def test_decompose_fanout_children_inherit_root_assignee_when_unrouted(kanban_home):
    """Unrouted children fall back to the ROOT task's assignee, not
    the decomposer's active profile (#114294). The active profile here is ``private``
    (an incognito profile with no credentials), so the old fallback spawned
    workers that deadlocked on capability blockers."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="zdr", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "made_up", "parents": []},
            {"title": "build", "body": "code it", "assignee": None, "parents": [0]},
        ],
    })

    # get_active_profile_name() is mocked to names[0] = "private" — the
    # global default chain would resolve there without kanban.default_assignee.
    patches = _patch_list_profiles(["private", "zdr"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0.assignee == "zdr"
    assert c1.assignee == "zdr"
    # Same class for the root: no ``orchestrator_profile`` must not hand the
    # orchestration card to the dispatcher's own (here: incognito) profile.
    assert root.assignee == "zdr"


def test_decompose_explicit_default_assignee_wins_over_root_assignee(kanban_home):
    """An explicitly configured ``kanban.default_assignee`` stays
    authoritative for unroutable children; the root task's assignee only
    fills in when no explicit default is set (explicit config → card
    assignee → active profile)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="engineer", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "made_up", "parents": []},
            {"title": "build", "body": "code it", "assignee": None, "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["engineer", "docs", "private"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"kanban": {"default_assignee": "docs"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0.assignee == "docs"
    assert c1.assignee == "docs"


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "made_up",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_load_routing_falls_back_to_defaults_when_config_unreadable(kanban_home, monkeypatch):
    """decompose_task promises ok=False on expected failures; a config read that raises (missing
    profile home, HomeInitializationError) must not escape _load_routing as an exception."""
    from hermes_cli import config as config_mod

    def _boom():
        raise FileNotFoundError("profile home is gone")

    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    routing = decomp._load_routing()
    assert routing.default_assignee == "default" and routing.auto_promote is True


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="x")  # ready, not triage

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False


# --- auto mode: genuine new intake only (gateway auto-decompose) ------------

_FANOUT = jsonlib.dumps({"fanout": True, "tasks": [{"title": "a"}, {"title": "b", "parents": [0]}]})
_SINGLE = jsonlib.dumps({"fanout": False, "title": "tight", "body": "spec"})


def _with_profiles(stack, names=("orchestrator",)):
    for p in _patch_list_profiles(list(names)):
        stack.enter_context(p)


def test_auto_selection_is_tenant_filtered_oldest_first_and_manual_unchanged(kanban_home):
    with kbc.connect() as conn:
        late = kb.create_task(conn, title="late", triage=True, tenant="a")
        tie_x = kb.create_task(conn, title="x", triage=True, tenant="a")
        tie_y = kb.create_task(conn, title="y", triage=True, tenant="a")
        other = kb.create_task(conn, title="other", triage=True, tenant="b")
        escalated = kb.create_task(conn, title="escalated", triage=True, tenant="a")
        with kb.write_txn(conn):
            kb._append_event(conn, escalated, "block_loop_detected", {"kind": "x"})
            for tid, created in ((late, 20), (tie_x, 10), (tie_y, 10), (other, 5), (escalated, 1)):
                conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (created, tid))
    assert decomp.list_triage_ids(auto=True, tenant="a") == [*sorted([tie_x, tie_y]), late]
    assert set(decomp.list_triage_ids(tenant="a")) == {late, tie_x, tie_y, escalated}


def test_auto_rejects_ineligible_task_before_any_model_call(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="escalated", triage=True)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "gave_up", {"error": "boom"})
    with _patch_aux_client(_FANOUT) as call_llm:
        outcome = decomp.decompose_task(tid, author="auto-decomposer", auto=True)
    assert outcome.ok is False
    call_llm.assert_not_called()


def test_manual_decompose_of_escalated_linked_root_still_fans_out(kanban_home):
    with kbc.connect() as conn:
        prerequisite = kb.create_task(conn, title="prerequisite")
        tid = kb.create_task(conn, title="root", triage=True, parents=[prerequisite])
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "block_loop_detected", {"kind": "x"})
    with ExitStack() as stack:
        _with_profiles(stack)
        stack.enter_context(_patch_aux_client(_FANOUT))
        outcome = decomp.decompose_task(tid, author="me")
    assert outcome.ok and outcome.fanout and len(outcome.child_ids) == 2


@pytest.mark.parametrize("payload", [_FANOUT, _SINGLE], ids=["fanout", "specify"])
def test_auto_eligibility_lost_during_model_call_is_rejected_at_write(kanban_home, payload):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough idea", triage=True, assignee="orchestrator")

    def escalate_then_reply(*args, **kwargs):
        with kbc.connect_closing() as conn, kb.write_txn(conn):
            kb._append_event(conn, tid, "block_loop_detected", {"kind": "x"})
        return _fake_aux_response(payload)

    with ExitStack() as stack:
        _with_profiles(stack)
        call_llm = stack.enter_context(patch("agent.auxiliary_client.call_llm", side_effect=escalate_then_reply))
        outcome = decomp.decompose_task(tid, author="auto-decomposer", auto=True)
    assert outcome.ok is False
    call_llm.assert_called_once()
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
