"""``unblock --expect-guard G --expect-related TASK:TOKEN``: a release decided on ANOTHER card's handoff applies only while that card is unchanged.

Real board DB under a temp HERMES_HOME. Drift is committed by a second connection after the caller read both guards; the transaction
case proves the related check, the unblock and the comment share one write transaction; CLI cases run the real ``hermes_cli.main`` process.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_guard as kbg
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli.kanban_db_guard import block_guard, parse_related, unblock_if_guard

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _guard(conn, tid) -> str:
    return block_guard(kb.get_task(conn, tid), kb.list_events(conn, tid), kb.list_runs(conn, tid))


def _pair(conn):
    """A parked review card and a finished implementation card whose last run carries the handoff metadata.

    Optional columns the typed projections (Task/Run dataclasses, ``show``) do not print are given distinctive values, so
    ``_rows`` can only stay equal when the SQL rows themselves did.
    """
    impl = kb.create_task(conn, title="impl", assignee="coder", created_by="orchestrator")
    claimed = kb.claim_task(conn, impl)
    assert kb.complete_task(conn, impl, summary="done", metadata={"head_sha": "a" * 40, "checks": {"unit": 0}},
                            expected_run_id=claimed.current_run_id)
    rev = kb.create_task(conn, title="review", assignee="reviewer", created_by="orchestrator", initial_status="blocked")
    conn.execute("UPDATE task_runs SET worker_started_at = 1700000123, max_runtime_seconds = 4242, last_heartbeat_at = 1700000456 "
                 "WHERE task_id = ?", (impl,))
    conn.execute("UPDATE tasks SET worker_started_at = 1700000789 WHERE id = ?", (rev,))
    conn.commit()
    return rev, impl


def _rows(conn, tid):
    """Every SQL column of every task, event, run and comment row of a card (not the typed Task/Run/Event projections)."""
    def all_rows(table, key):
        return [tuple(r) for r in conn.execute(f"SELECT * FROM {table} WHERE {key} = ? ORDER BY 1", (tid,))]
    return tuple(all_rows(t, k) for t, k in (("tasks", "id"), ("task_events", "task_id"), ("task_runs", "task_id"), ("task_comments", "task_id")))


def _cli(home, *args, extra_env=None):
    env = {"HOME": str(home.parent), "PATH": os.environ["PATH"], "HERMES_HOME": str(home), "PYTHONPATH": str(REPO), **(extra_env or {})}
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", *args], cwd=REPO, env=env,
                          capture_output=True, text=True, check=False)


def _show(home, tid) -> dict:
    return json.loads(_cli(home, "show", tid, "--json").stdout)


def _edit(conn, impl, **fields):
    assert kb.edit_task(conn, impl, result="r", **fields)


def _late_run(conn, impl):  # a late run row: no task or event change
    conn.execute("INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at, summary) "
                 "VALUES (?, 'coder', 'done', 'completed', 1, 2, 'late')", (impl,))
    conn.commit()


def _workspace_only(conn, impl):  # no event and no run row at all
    kbw.set_workspace_path(conn, impl, "/somewhere/else")


def _status_ready(conn, impl):
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (impl,))
    conn.commit()


_DRIFT = {
    "summary": lambda conn, impl: _edit(conn, impl, summary="a different handoff"),
    "metadata": lambda conn, impl: _edit(conn, impl, summary="done", metadata={"head_sha": "b" * 40, "checks": {"unit": 0}}),
    "checks": lambda conn, impl: _edit(conn, impl, summary="done", metadata={"head_sha": "a" * 40, "checks": {}}),
    "run": _late_run,
    "workspace": _workspace_only,
    "status": _status_ready,
}


def _drift(conn, impl, kind):
    _DRIFT[kind](conn, impl)


@pytest.mark.parametrize("kind", ["summary", "metadata", "checks", "run", "workspace", "status"])
def test_implementation_drift_after_both_reads_refuses_and_keeps_the_adversary_edit(kanban_home, kind):
    with kbc.connect_closing() as reader, kbc.connect_closing() as other:
        rev, impl = _pair(reader)
        g_rev, g_impl = _guard(reader, rev), _guard(reader, impl)
        _drift(other, impl, kind)
        assert _guard(other, impl) != g_impl  # every consumed field moves the token, including the event-free workspace/run edits
        review_before, impl_after = _rows(other, rev), _rows(other, impl)
        assert unblock_if_guard(reader, rev, g_rev, comment=("orchestrator", "UNBLOCK: x"), related=(impl, g_impl)) is False
        assert _rows(other, rev) == review_before and kb.get_task(other, rev).status == "blocked"
        assert _rows(other, impl) == impl_after  # the other writer's edit is neither rolled back nor rewritten


def test_unchanged_pair_releases_once_with_one_comment_and_never_touches_the_implementation(kanban_home):
    with kbc.connect_closing() as conn:
        rev, impl = _pair(conn)
        g_rev, g_impl = _guard(conn, rev), _guard(conn, impl)
        impl_before = _rows(conn, impl)
        assert unblock_if_guard(conn, rev, g_rev, comment=("orchestrator", "UNBLOCK: frozen"), related=(impl, g_impl)) is True
        assert kb.get_task(conn, rev).status == "ready"
        assert [c.body for c in kb.list_comments(conn, rev)] == ["UNBLOCK: frozen"]
        assert _rows(conn, impl) == impl_before
        assert unblock_if_guard(conn, rev, g_rev, comment=("orchestrator", "UNBLOCK: again"), related=(impl, g_impl)) is False  # repeat: nothing more
        assert [c.body for c in kb.list_comments(conn, rev)] == ["UNBLOCK: frozen"]


def test_a_stale_review_guard_refuses_even_with_a_current_related_guard(kanban_home):
    with kbc.connect_closing() as reader, kbc.connect_closing() as other:
        rev, impl = _pair(reader)
        g_rev, g_impl = _guard(reader, rev), _guard(reader, impl)
        kb.block_task(other, rev, reason="needs Andrew", kind="needs_input")
        before = _rows(other, rev)
        assert unblock_if_guard(reader, rev, g_rev, comment=("orchestrator", "UNBLOCK: x"), related=(impl, g_impl)) is False
        assert _rows(other, rev) == before


def test_missing_wrong_or_foreign_related_card_refuses(kanban_home):
    with kbc.connect_closing() as conn:
        rev, impl = _pair(conn)
        _, other_impl = _pair(conn)
        g_rev, g_impl, g_other = _guard(conn, rev), _guard(conn, impl), _guard(conn, other_impl)
        before = _rows(conn, rev)
        for related in (("t_missing0", g_impl), (other_impl, g_impl), (impl, g_other), (impl, "0" * 64)):
            assert unblock_if_guard(conn, rev, g_rev, related=related) is False, related
        assert _rows(conn, rev) == before


def test_the_token_names_its_card(kanban_home):
    """Only the id differs between these two guard inputs: same task fields, same supplied event rows, same supplied run rows."""
    with kbc.connect_closing() as conn:
        _, impl = _pair(conn)
        task, events, runs = kb.get_task(conn, impl), kb.list_events(conn, impl), kb.list_runs(conn, impl)
        assert events and runs
        assert block_guard(task, events, runs) == _guard(conn, impl)
        assert block_guard(dataclasses.replace(task, id="t_other_card"), events, runs) != block_guard(task, events, runs)


def test_the_full_row_snapshot_sees_columns_the_typed_projections_omit(kanban_home):
    with kbc.connect_closing() as conn:
        _, impl = _pair(conn)
        before, projected = _rows(conn, impl), kb.list_runs(conn, impl)
        conn.execute("UPDATE task_runs SET worker_started_at = 1 WHERE task_id = ?", (impl,))
        conn.commit()
        assert _rows(conn, impl) != before
        assert "worker_started_at" not in projected[0].__dict__ and kb.list_runs(conn, impl) == projected


def test_both_guard_reads_the_unblock_and_the_comment_share_one_write_transaction(kanban_home, monkeypatch):
    """The review guard read, the related guard read, the unblock and the comment all run between one BEGIN and its COMMIT on the real
    connection, and a second connection is locked out from the related card at each of the two guard reads."""
    with kbc.connect_closing() as reader:
        rev, impl = _pair(reader)
        g_rev, g_impl = _guard(reader, rev), _guard(reader, impl)
        db = str(kb.kanban_db_path())
        statements: list[str] = []
        reader.set_trace_callback(statements.append)
        real_guard_matches, reads, locked = kbg.guard_matches, [], []

        def observed_guard_matches(conn, task_id, expect):
            result = real_guard_matches(conn, task_id, expect)  # the real read, result unchanged
            reads.append((task_id, expect, result, conn.in_transaction, len(statements)))
            adversary = sqlite3.connect(db, timeout=0)
            try:
                adversary.execute("UPDATE tasks SET workspace_path = '/x' WHERE id = ?", (impl,))
                adversary.commit()
            except sqlite3.OperationalError as e:
                locked.append(str(e))
            finally:
                adversary.close()
            return result

        monkeypatch.setattr(kbg, "guard_matches", observed_guard_matches)
        assert unblock_if_guard(reader, rev, g_rev, comment=("orchestrator", "UNBLOCK: x"), related=(impl, g_impl)) is True
        monkeypatch.undo()
        reader.set_trace_callback(None)
        assert [(r[0], r[1], r[2], r[3]) for r in reads] == [(rev, g_rev, True, True), (impl, g_impl, True, True)]
        assert len(locked) == 2 and all("locked" in e for e in locked)  # the implementation card could not change at either read
        begin = next(i for i, s in enumerate(statements) if s.upper().startswith("BEGIN"))
        commit = max(i for i, s in enumerate(statements) if s.upper().startswith("COMMIT"))
        assert begin < reads[0][4] <= reads[1][4] < commit  # both reads were made on this connection inside BEGIN ... COMMIT
        after_second_read = statements[reads[1][4]:commit]
        assert any(s.startswith("UPDATE tasks") for s in after_second_read) and any("INSERT INTO task_comments" in s for s in after_second_read)
        assert kb.get_task(reader, impl).workspace_path in (None, "")  # the adversary's write never landed
        adversary = sqlite3.connect(db, timeout=0)  # serialization ends at commit: afterwards the card is writable again
        adversary.execute("UPDATE tasks SET workspace_path = '/after' WHERE id = ?", (impl,))
        adversary.commit()
        adversary.close()


def test_a_failure_after_the_related_check_rolls_the_release_back(kanban_home, monkeypatch):
    with kbc.connect_closing() as reader:
        rev, impl = _pair(reader)
        g_rev, g_impl = _guard(reader, rev), _guard(reader, impl)
        before = _rows(reader, rev)
        monkeypatch.setattr(kb, "add_comment", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            unblock_if_guard(reader, rev, g_rev, comment=("orchestrator", "UNBLOCK: x"), related=(impl, g_impl))
        assert _rows(reader, rev) == before


# --- CLI: the real process --------------------------------------------------------------------------------------------------

def test_cli_pair_release_then_repeat_and_workspace_only_drift(kanban_home):
    with kbc.connect_closing() as conn:
        rev, impl = _pair(conn)
    r, i = _show(kanban_home, rev), _show(kanban_home, impl)
    assert "--expect-related" in _cli(kanban_home, "unblock", "--help").stdout  # what the private gate probes
    with kbc.connect_closing() as conn:
        kbw.set_workspace_path(conn, impl, "/changed")  # no event, no run row
    stale = _cli(kanban_home, "unblock", rev, "--expect-guard", r["block_guard"], "--expect-related", f"{impl}:{i['block_guard']}", "--reason", "go")
    assert stale.returncode == 1 and "nothing written" in stale.stderr
    assert _show(kanban_home, impl)["block_guard"] != i["block_guard"]
    r, i = _show(kanban_home, rev), _show(kanban_home, impl)
    ok = _cli(kanban_home, "unblock", rev, "--expect-guard", r["block_guard"], "--expect-related", f"{impl}:{i['block_guard']}", "--reason", "go")
    assert ok.returncode == 0, ok.stderr
    again = _cli(kanban_home, "unblock", rev, "--expect-guard", r["block_guard"], "--expect-related", f"{impl}:{i['block_guard']}", "--reason", "go")
    assert again.returncode == 1
    shown = _show(kanban_home, rev)
    assert shown["task"]["status"] == "ready" and [c["body"] for c in shown["comments"]] == ["UNBLOCK: go"]


def test_cli_worker_role_is_denied_before_anything_is_written(kanban_home):
    with kbc.connect_closing() as conn:
        rev, impl = _pair(conn)
    r, i = _show(kanban_home, rev), _show(kanban_home, impl)
    denied = _cli(kanban_home, "unblock", rev, "--expect-guard", r["block_guard"], "--expect-related", f"{impl}:{i['block_guard']}",
                  extra_env={"HERMES_KANBAN_TASK": impl})
    assert denied.returncode != 0 and "orchestrator-only" in denied.stderr
    assert _show(kanban_home, rev)["task"]["status"] == "blocked"


def test_legacy_single_card_guarded_unblock_is_unchanged(kanban_home):
    with kbc.connect_closing() as conn:
        rev, _ = _pair(conn)
    r = _show(kanban_home, rev)
    ok = _cli(kanban_home, "unblock", rev, "--expect-guard", r["block_guard"], "--reason", "legacy")
    assert ok.returncode == 0, ok.stderr
    assert _show(kanban_home, rev)["task"]["status"] == "ready"


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _store_files(home) -> list[str]:
    return sorted(str(p.relative_to(home)) for p in home.rglob("*") if p.is_file() and "kanban" in p.name)


G = "0" * 64
_BAD_RELATED = [
    ("unblock", "t_a", "--expect-guard", G, "--expect-related", ""),
    ("unblock", "t_a", "--expect-guard", G, "--expect-related", "t_b"),
    ("unblock", "t_a", "--expect-guard", G, "--expect-related", f"t_b:{G[:63]}"),
    ("unblock", "t_a", "--expect-guard", G, "--expect-related", f"t_b:{'A' * 64}"),
    ("unblock", "t_a", "--expect-guard", G, "--expect-related", f" t_b:{G}"),
    ("unblock", "t_a", "--expect-guard", G, "--expect-related", f"t_b:{G}\n"),
    ("unblock", "t_a", "--expect-guard", G, "--expect-related", f"t_b:{G}:{G}"),
    ("unblock", "t_a", "--expect-related", f"t_b:{G}"),  # partial: related without the review guard
    ("unblock", "t_a", "--expect-guard", G, "--expect-related", f"t_a:{G}"),  # self
    ("unblock", "t_a", "t_c", "--expect-guard", G, "--expect-related", f"t_b:{G}"),  # more than one id
]


@pytest.mark.parametrize("argv", _BAD_RELATED)
def test_malformed_or_partial_related_binding_refuses_before_the_store_exists(fresh_home, argv):
    r = _cli(fresh_home, *argv)
    assert r.returncode == 2 and "--expect-" in r.stderr and not r.stdout
    assert _store_files(fresh_home) == []


def test_parse_related_is_exact():
    assert parse_related(f"t_ab:{G}") == ("t_ab", G)
    assert all(parse_related(v) is None for v in (None, "", f"t_ab:{G.upper().replace('0', 'A')}", f"t_ab :{G}", f":{G}", "t_ab:"))


def test_a_valid_related_binding_still_auto_initialises_and_the_op_refuses_the_unknown_card(fresh_home):
    argv = ["unblock", "t_a", "--expect-guard", G, "--expect-related", f"t_b:{G}"]
    import argparse
    parser = kc.build_parser(argparse.ArgumentParser(prog="t").add_subparsers(dest="_top"))
    assert kc.kanban_command(parser.parse_args(argv)) == 1
    assert any(f.startswith("kanban.db") for f in _store_files(fresh_home))
