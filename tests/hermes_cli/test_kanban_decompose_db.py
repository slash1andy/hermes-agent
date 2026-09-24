"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


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
    with kb.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kb.connect() as conn:
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


def test_decompose_returns_none_when_task_missing(kanban_home):
    with kb.connect() as conn:
        result = kb.decompose_triage_task(
            conn,
            "nonexistent",
            root_assignee="orch",
            children=[{"title": "x"}],
            author="me",
        )
    assert result is None


def test_decompose_returns_none_when_task_not_in_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="already a real task")  # not triage
        result = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "x"}],
            author="me",
        )
    assert result is None


def test_decompose_empty_children_returns_none(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        result = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[],
            author="me",
        )
    assert result is None


def test_decompose_rejects_self_parent(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="cannot list itself"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[{"title": "x", "parents": [0]}],
                author="me",
            )


def test_decompose_rejects_out_of_range_parent(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="not a valid index"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[{"title": "x", "parents": [5]}],
                author="me",
            )


def test_decompose_rejects_cyclic_parents(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="cyclic dependency"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[
                    {"title": "A", "parents": [1]},
                    {"title": "B", "parents": [0]},
                ],
                author="me",
            )


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)


def test_decompose_children_inherit_dir_workspace(kanban_home):
    """Fan-out children inherit the root's dir workspace, not scratch."""
    proj = "/home/teknium/myproject"
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="codegen root", assignee="worker",
            workspace_kind="dir", workspace_path=proj, triage=True,
        )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "part A"}, {"title": "part B", "parents": [0]}],
            author="decomposer",
        )
    assert child_ids and len(child_ids) == 2
    with kb.connect() as conn:
        for cid in child_ids:
            t = kb.get_task(conn, cid)
            assert t.workspace_kind == "dir"
            assert t.workspace_path == proj


def test_decompose_children_stay_scratch_when_root_scratch(kanban_home):
    """No regression: a scratch root still fans out into scratch children."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="scratch root", assignee="worker",
            workspace_kind="scratch", triage=True,
        )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "s1"}], author="decomposer",
        )
    with kb.connect() as conn:
        t = kb.get_task(conn, child_ids[0])
    assert t.workspace_kind == "scratch"
    assert t.workspace_path is None


def test_decompose_per_child_workspace_override(kanban_home):
    """An explicit per-child workspace beats inheritance."""
    proj = "/home/teknium/myproject"
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="root", assignee="worker",
            workspace_kind="dir", workspace_path=proj, triage=True,
        )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[
                {"title": "override", "workspace_kind": "dir",
                 "workspace_path": "/other/repo"},
                {"title": "inherit"},
            ],
            author="decomposer",
        )
    with kb.connect() as conn:
        over = kb.get_task(conn, child_ids[0])
        inh = kb.get_task(conn, child_ids[1])
    assert over.workspace_path == "/other/repo"
    assert inh.workspace_path == proj


# --- auto-decompose eligibility: only genuine new intake -------------------

_ONE_CHILD = [{"title": "only step", "parents": []}]


def _auto(conn, tid):
    return kb.decompose_triage_task(
        conn, tid, root_assignee="orch", children=_ONE_CHILD, intake_only=True,
    )


def test_fresh_triage_is_auto_eligible_and_decomposes(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        assert kb.list_auto_decompose_ids(conn) == [tid]
        assert _auto(conn, tid)


def test_retry_circuit_triage_is_excluded(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        # Escalation record left by the block-loop breaker.
        conn.execute("UPDATE tasks SET block_kind = 'needs_input' WHERE id = ?", (tid,))
        conn.commit()
        assert kb.list_auto_decompose_ids(conn) == []
        assert _auto(conn, tid) is None
        assert kb.get_task(conn, tid).status == "triage"

        # Generic retry-circuit (failure-limit) escalation: gave_up event.
        t2 = _create_triage(conn)
        kb._append_event(conn, t2, "gave_up", {"error": "boom"})
        conn.commit()
        assert t2 not in kb.list_auto_decompose_ids(conn)
        assert _auto(conn, t2) is None


def test_already_decomposed_root_is_excluded_no_duplicate_graph(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        assert _auto(conn, tid)
        # Root later escalates back to triage; repeated ticks must not refan.
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
        conn.commit()
        assert kb.list_auto_decompose_ids(conn) == []
        assert _auto(conn, tid) is None
        assert _auto(conn, tid) is None
        n = conn.execute(
            "SELECT COUNT(*) FROM task_links WHERE child_id = ?", (tid,)
        ).fetchone()[0]
        assert n == 1


def test_changed_task_rejected_at_mutation_boundary(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        assert kb.list_auto_decompose_ids(conn) == [tid]  # selected while eligible
        kb._append_event(conn, tid, "block_loop_detected", {"kind": "x"})
        conn.commit()  # new escalation lands before the mutation
        assert _auto(conn, tid) is None
        assert kb.specify_triage_task(conn, tid, title="t", intake_only=True) is False
        assert kb.get_task(conn, tid).status == "triage"


def test_linked_task_excluded_and_rejected_at_mutation_boundary(kanban_home):
    for as_parent in (False, True):
        with kb.connect() as conn:
            tid = _create_triage(conn)
            other = kb.create_task(conn, title="other")
            assert tid in kb.list_auto_decompose_ids(conn)  # selected while unlinked
            a, b = (tid, other) if as_parent else (other, tid)
            conn.execute(
                "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (a, b)
            )
            conn.commit()  # graph link lands before the mutation
            assert tid not in kb.list_auto_decompose_ids(conn)
            assert _auto(conn, tid) is None
            assert kb.specify_triage_task(conn, tid, title="t", intake_only=True) is False
            assert kb.get_task(conn, tid).status == "triage"


def test_human_held_work_untouched(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="held", initial_status="blocked")
        assert kb.list_auto_decompose_ids(conn) == []
        assert _auto(conn, tid) is None
        assert kb.get_task(conn, tid).status == "blocked"
