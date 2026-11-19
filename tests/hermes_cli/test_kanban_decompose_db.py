"""Tests for decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert comments
    assert any(ev.kind == "decomposed" for ev in events)


# --- auto-decompose eligibility: only genuine new intake -------------------

_ONE_CHILD = [{"title": "only step", "parents": []}]


def _auto(conn, tid):
    return decompose_triage_task(
        conn, tid, root_assignee="orch", children=_ONE_CHILD, intake_only=True,
    )


def test_fresh_triage_decomposes_under_intake_only(kanban_home):
    with kbc.connect() as conn:
        assert _auto(conn, _create_triage(conn))


def test_escalated_triage_rejected_under_intake_only(kanban_home):
    with kbc.connect() as conn:
        held = _create_triage(conn)
        gave_up = _create_triage(conn)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET block_kind = 'needs_input' WHERE id = ?", (held,))
            kb._append_event(conn, gave_up, "gave_up", {"error": "boom"})
        for tid in (held, gave_up):
            assert _auto(conn, tid) is None
            assert kb.get_task(conn, tid).status == "triage"


def test_retriaged_decomposed_root_is_not_refanned_under_intake_only(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        assert _auto(conn, tid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
        assert _auto(conn, tid) is None
        assert conn.execute("SELECT COUNT(*) FROM task_links WHERE child_id = ?", (tid,)).fetchone()[0] == 1


def test_eligibility_lost_before_mutation_is_rejected_in_both_transactions(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        with kb.write_txn(conn):  # escalation lands after selection, before the write
            kb._append_event(conn, tid, "block_loop_detected", {"kind": "x"})
        assert _auto(conn, tid) is None
        assert kb.specify_triage_task(conn, tid, title="t", intake_only=True) is False
        assert kb.get_task(conn, tid).status == "triage"


@pytest.mark.parametrize("as_parent", [False, True])
def test_link_added_before_mutation_is_rejected_in_both_transactions(kanban_home, as_parent):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        other = kb.create_task(conn, title="other")
        with kb.write_txn(conn):  # either link direction lands after selection
            kb._link(conn, *((tid, other) if as_parent else (other, tid)))
        assert _auto(conn, tid) is None
        assert kb.specify_triage_task(conn, tid, title="t", intake_only=True) is False
        assert kb.get_task(conn, tid).status == "triage"


def test_manual_fanout_of_linked_root_is_unrestricted(kanban_home):
    with kbc.connect() as conn:
        prerequisite = kb.create_task(conn, title="prerequisite")
        root = kb.create_task(conn, title="root", triage=True, parents=[prerequisite])
        assert _auto(conn, root) is None
        assert decompose_triage_task(conn, root, root_assignee="orch", children=_ONE_CHILD)


def test_human_held_work_untouched_under_intake_only(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="held", initial_status="blocked")
        assert _auto(conn, tid) is None
        assert kb.get_task(conn, tid).status == "blocked"
