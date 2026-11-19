"""``kanban show --json`` block_guard + ``unblock``/``block --expect-guard``: a write applies only to the card state read.

Real board DB under a temp HERMES_HOME. Stale-read cases interleave a second connection deterministically; the race case lets
two writers contend and asserts the serial-order invariant; CLI cases run the real ``hermes_cli.main`` process for exit codes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_guard import block_guard, unblock_if_guard

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _parked(conn, **kw) -> str:
    return kb.create_task(conn, title="review", assignee="reviewer", created_by="orchestrator",
                          initial_status="blocked", **kw)


def _read_guard(conn, tid) -> str:
    return block_guard(kb.get_task(conn, tid), kb.list_events(conn, tid), kb.list_runs(conn, tid))


def _kinds(conn, tid) -> list[str]:
    return [e.kind for e in kb.list_events(conn, tid)]


def _insert_run(conn, tid, *, profile="reviewer", summary="PASS", metadata=None) -> int:
    """A terminal run row written WITHOUT touching tasks/events (what a late/corrected run record looks like)."""
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at, summary, metadata) "
        "VALUES (?, ?, 'done', 'completed', ?, ?, ?, ?)",
        (tid, profile, now, now, summary, json.dumps(metadata) if metadata is not None else None))
    conn.commit()
    return cur.lastrowid


def _cli(home, *args, cwd=REPO):
    env = {"HOME": str(home.parent), "PATH": os.environ["PATH"], "HERMES_HOME": str(home), "PYTHONPATH": str(REPO)}
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, check=False)


def test_guard_matches_when_unchanged_and_unblocks_with_comment_atomically(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _parked(conn)
        guard = _read_guard(conn, tid)
        assert unblock_if_guard(conn, tid, guard, comment=("orchestrator", "UNBLOCK: frozen")) is True
        assert kb.get_task(conn, tid).status == "ready"
        assert [c.body for c in kb.list_comments(conn, tid)] == ["UNBLOCK: frozen"]
        assert _kinds(conn, tid)[-2:] == ["unblocked", "commented"]


def test_human_hold_after_read_refuses_and_writes_nothing(kanban_home):
    with kbc.connect_closing() as reader, kbc.connect_closing() as human:
        tid = _parked(reader)
        guard = _read_guard(reader, tid)  # what the gate classified
        assert kb.block_task(human, tid, reason="needs Andrew", kind="needs_input") is True  # lands afterwards
        events_before = _kinds(human, tid)
        assert unblock_if_guard(reader, tid, guard, comment=("orchestrator", "UNBLOCK: x")) is False
        task = kb.get_task(human, tid)
        assert (task.status, task.block_kind) == ("blocked", "needs_input")
        assert _kinds(human, tid) == events_before and kb.list_comments(human, tid) == []


@pytest.mark.parametrize("change", ["comment", "unblock_reblock", "assign"])
def test_any_later_change_refuses(kanban_home, change):
    with kbc.connect_closing() as reader, kbc.connect_closing() as other:
        tid = _parked(reader)
        guard = _read_guard(reader, tid)
        if change == "comment":
            kb.add_comment(other, tid, "andrew", "hold on")
        elif change == "assign":
            kb.assign_task(other, tid, "coder")
        else:
            kb.unblock_task(other, tid)
            kb.block_task(other, tid, reason="again", kind="needs_input")
        assert unblock_if_guard(reader, tid, guard) is False
        assert kb.get_task(other, tid).status == "blocked"


@pytest.mark.parametrize("late", ["insert_run", "correct_verdict_metadata", "correct_run_profile"])
def test_run_only_change_after_read_refuses_with_no_task_or_event_change(kanban_home, late):
    """The classification reads run profile/summary/metadata. A late or corrected run row changes none of task/events."""
    with kbc.connect_closing() as reader, kbc.connect_closing() as other:
        tid = _parked(reader)
        run = _insert_run(other, tid, summary="PASS", metadata={"verdict": "PASS"}) if late != "insert_run" else None
        guard = _read_guard(reader, tid)
        events_before, task_before = _kinds(other, tid), kb.get_task(other, tid)
        if late == "insert_run":
            _insert_run(other, tid, summary="CHANGES_REQUIRED S1: fix", metadata={"verdict": "CHANGES_REQUIRED"})
        elif late == "correct_verdict_metadata":
            other.execute("UPDATE task_runs SET metadata = ?, summary = ? WHERE id = ?",
                          (json.dumps({"verdict": "PARTIAL"}), "PARTIAL: corrected", run))
            other.commit()
        else:
            other.execute("UPDATE task_runs SET profile = 'coder' WHERE id = ?", (run,))
            other.commit()
        assert _kinds(other, tid) == events_before  # no event change: the V3 guard could not see this
        assert kb.get_task(other, tid).status == task_before.status
        assert _read_guard(other, tid) != guard
        assert unblock_if_guard(reader, tid, guard, comment=("orchestrator", "UNBLOCK: x")) is False
        assert kb.get_task(other, tid).status == "blocked" and kb.list_comments(other, tid) == []
        assert _kinds(other, tid) == events_before


def test_activated_card_is_never_reblocked_or_unblocked_by_a_stale_guard(kanban_home):
    """The card was released (and claimed) by someone else after the read: a stale guard is a no-op."""
    with kbc.connect_closing() as reader, kbc.connect_closing() as other:
        tid = _parked(reader)
        guard = _read_guard(reader, tid)
        assert kb.unblock_task(other, tid) is True
        assert kb.claim_task(other, tid) is not None
        before = (kb.get_task(other, tid).status, _kinds(other, tid))
        assert unblock_if_guard(reader, tid, guard) is False
        assert (kb.get_task(other, tid).status, _kinds(other, tid)) == before


def test_unknown_task_refused(kanban_home):
    with kbc.connect_closing() as conn:
        assert unblock_if_guard(conn, "t_missing", "0" * 64) is False


def test_concurrent_human_hold_and_guarded_release_serialise(kanban_home):
    """Whichever write wins, the board never shows a release that the human hold preceded."""
    for _ in range(15):
        with kbc.connect_closing() as setup:
            tid = _parked(setup)
            guard = _read_guard(setup, tid)
        go, results = threading.Barrier(2), {}

        def release():
            with kbc.connect_closing() as conn:
                go.wait()
                results["released"] = unblock_if_guard(conn, tid, guard)

        def hold():
            with kbc.connect_closing() as conn:
                go.wait()
                results["held"] = kb.block_task(conn, tid, reason="needs Andrew", kind="needs_input")

        threads = [threading.Thread(target=release), threading.Thread(target=hold)]
        [t.start() for t in threads]
        [t.join(30) for t in threads]
        with kbc.connect_closing() as conn:
            kinds = _kinds(conn, tid)
            if results["released"]:  # release first; the hold then re-blocked the now-ready card
                assert kinds.count("blocked") == 2 and kinds.index("unblocked") < len(kinds) - 1 - kinds[::-1].index("blocked")
                assert results["held"] is True
            else:  # hold first: the stale release wrote nothing
                assert "unblocked" not in kinds and results["held"] is True
                assert kb.get_task(conn, tid).block_kind == "needs_input"


# --- rollback composition (BEGIN IMMEDIATE + savepoint): nothing half-written -----------------------------------

def test_comment_failure_rolls_the_unblock_back(kanban_home, monkeypatch):
    with kbc.connect_closing() as conn:
        tid = _parked(conn)
        guard = _read_guard(conn, tid)
        monkeypatch.setattr(kb, "add_comment", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("comment write failed")))
        with pytest.raises(RuntimeError, match="comment write failed"):
            unblock_if_guard(conn, tid, guard, comment=("orchestrator", "UNBLOCK: x"))
        monkeypatch.undo()
        assert kb.get_task(conn, tid).status == "blocked" and _kinds(conn, tid) == ["created", "blocked"]
        assert not conn.in_transaction
        assert unblock_if_guard(conn, tid, _read_guard(conn, tid)) is True  # the connection is not poisoned


def test_commit_failure_rolls_everything_back(kanban_home, monkeypatch):
    with kbc.connect_closing() as conn:
        tid = _parked(conn)
        guard = _read_guard(conn, tid)
        real = kbc._execute_boundary_with_retry

        def failing(c, sql):
            if sql == "COMMIT":
                raise sqlite3_error("disk I/O error on COMMIT")
            return real(c, sql)

        import sqlite3
        sqlite3_error = sqlite3.OperationalError
        monkeypatch.setattr(kbc, "_execute_boundary_with_retry", failing)
        with pytest.raises(sqlite3.OperationalError, match="COMMIT"):
            unblock_if_guard(conn, tid, guard, comment=("orchestrator", "UNBLOCK: x"))
        monkeypatch.undo()
        assert not conn.in_transaction
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.list_comments(conn, tid) == [] and _kinds(conn, tid) == ["created", "blocked"]


def test_caller_transaction_is_preserved_by_the_savepoint(kanban_home):
    """unblock_task nests (savepoint) under an outer transaction; an outer rollback undoes it."""
    with kbc.connect_closing() as conn:
        tid = _parked(conn)
        with pytest.raises(RuntimeError, match="outer"):
            with kbc.write_txn(conn):
                assert kb.unblock_task(conn, tid) is True
                raise RuntimeError("outer failure")
        assert kb.get_task(conn, tid).status == "blocked"


# --- CLI: show publishes the guard, run-only changes move it, invalid tokens never fall through ----------------

def test_cli_show_publishes_guard_and_unblock_honours_it(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _parked(conn)
    shown = json.loads(kc.run_slash(f"show {tid} --json"))
    assert len(shown["block_guard"]) == 64
    with kbc.connect_closing() as conn:
        kb.block_task(conn, tid, reason="needs Andrew", kind="needs_input")
    out = kc.run_slash(f"unblock {tid} --expect-guard {shown['block_guard']} --reason 'frozen candidate'")
    assert "state changed" in out and "Unblocked" not in out
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "blocked" and kb.list_comments(conn, tid) == []
    fresh = json.loads(kc.run_slash(f"show {tid} --json"))["block_guard"]
    assert fresh != shown["block_guard"]
    assert "Unblocked" in kc.run_slash(f"unblock {tid} --expect-guard {fresh} --reason 'frozen candidate'")
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "ready"
        assert [c.body for c in kb.list_comments(conn, tid)] == ["UNBLOCK: frozen candidate"]


def test_cli_run_only_late_insert_moves_the_shown_guard_and_refuses_the_unblock(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _parked(conn)
    shown = json.loads(kc.run_slash(f"show {tid} --json"))
    with kbc.connect_closing() as conn:
        _insert_run(conn, tid, summary="CHANGES_REQUIRED S1: fix", metadata={"verdict": "CHANGES_REQUIRED"})
    r = _cli(kanban_home, "unblock", tid, "--expect-guard", shown["block_guard"], "--reason", "frozen")
    assert r.returncode == 1 and "state changed" in r.stderr and "Unblocked" not in r.stdout
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "blocked" and kb.list_comments(conn, tid) == []
    assert json.loads(kc.run_slash(f"show {tid} --json"))["block_guard"] != shown["block_guard"]


def test_cli_show_with_run_filter_still_binds_every_run(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _parked(conn)
        _insert_run(conn, tid)
    plain = json.loads(kc.run_slash(f"show {tid} --json"))["block_guard"]
    filtered = json.loads(kc.run_slash(f"show {tid} --json --state-type outcome --state-name nothing"))["block_guard"]
    assert plain == filtered


@pytest.mark.parametrize("bad", ["", " ", "nothex", "0" * 63, "A" * 64, "0" * 65])
def test_cli_invalid_or_empty_guard_refuses_before_the_handler_touches_the_store(kanban_home, bad):
    """rc 2 (usage) for a nonexistent id proves validation ran BEFORE the per-id op (which would report rc 1 'cannot ...')."""
    with kbc.connect_closing() as conn:
        tid = _parked(conn)
        before = (_kinds(conn, tid), kb.list_comments(conn, tid))
    for verb in (("unblock", "t_missing", "--reason", "r"), ("block", "t_missing", "why"), ("unblock", tid, "--reason", "r")):
        r = _cli(kanban_home, verb[0], *verb[1:], "--expect-guard", bad)
        assert r.returncode == 2, (verb, r.stdout, r.stderr)
        assert "--expect-guard must be the 64-hex" in r.stderr and not r.stdout
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "blocked"
        assert (_kinds(conn, tid), kb.list_comments(conn, tid)) == before


def test_cli_empty_guard_never_falls_through_to_a_plain_unblock(kanban_home):
    with kbc.connect_closing() as conn:
        a, b = _parked(conn), _parked(conn)
    for ids in ([a], [a, b]):
        r = _cli(kanban_home, "unblock", *ids, "--expect-guard", "", "--reason", "x")
        assert r.returncode == 2 and "64-hex" in r.stderr and not r.stdout
    with kbc.connect_closing() as conn:
        assert {kb.get_task(conn, a).status, kb.get_task(conn, b).status} == {"blocked"}
        assert kb.list_comments(conn, a) == []


def test_cli_guard_needs_exactly_one_id_and_plain_unblock_is_unchanged(kanban_home):
    with kbc.connect_closing() as conn:
        a, b = _parked(conn), _parked(conn)
    r = _cli(kanban_home, "unblock", a, b, "--expect-guard", "0" * 64)
    assert r.returncode == 2 and "exactly one task id" in r.stderr
    assert "Unblocked" in kc.run_slash(f"unblock {a} {b}")  # generic human unblock: no guard, same as before
    with kbc.connect_closing() as conn:
        assert {kb.get_task(conn, a).status, kb.get_task(conn, b).status} == {"ready"}


# --- guarded block (the SMC fence's conditional writer) ---------------------------------------------------------

def _ready_reviewer(conn) -> str:
    tid = kb.create_task(conn, title="review", assignee="reviewer", created_by="orchestrator")
    assert kb.get_task(conn, tid).status == "ready"
    return tid


def test_guarded_block_blocks_an_unchanged_ready_card_and_plain_block_is_unchanged(kanban_home):
    with kbc.connect_closing() as conn:
        a, b = _ready_reviewer(conn), _ready_reviewer(conn)
    guard = json.loads(kc.run_slash(f"show {a} --json"))["block_guard"]
    assert "Blocked" in kc.run_slash(f"block {a} --kind capability --expect-guard {guard} 'FENCE: x'")
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, a)
        assert (task.status, task.block_kind) == ("blocked", "capability")
        assert kb.list_comments(conn, a) == []  # the guarded path writes no pre-comment
    assert "Blocked" in kc.run_slash(f"block {b} 'plain'")
    with kbc.connect_closing() as conn:
        assert [c.body for c in kb.list_comments(conn, b)] == ["BLOCKED: plain"]


def test_stale_guarded_block_leaves_a_claimed_card_running(kanban_home):
    """The ready-read / unconditional-block race: the reviewer claims the card between the read and the block."""
    with kbc.connect_closing() as conn:
        tid = _ready_reviewer(conn)
    guard = json.loads(kc.run_slash(f"show {tid} --json"))["block_guard"]
    with kbc.connect_closing() as conn:
        claimed = kb.claim_task(conn, tid)
        before = (kb.get_task(conn, tid).status, kb.get_task(conn, tid).current_run_id, _kinds(conn, tid))
    r = _cli(kanban_home, "block", tid, "--kind", "capability", "--expect-guard", guard, "FENCE: x")
    assert r.returncode == 1 and "state changed" in r.stderr
    with kbc.connect_closing() as conn:
        after = (kb.get_task(conn, tid).status, kb.get_task(conn, tid).current_run_id, _kinds(conn, tid))
        assert after == before and after[1] == claimed.current_run_id and after[0] == "running"
        assert kb.list_comments(conn, tid) == []


# --- invalid guard refuses BEFORE auto-init creates the store (fresh home, no pre-existing DB) -----------------

@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _store_files(home) -> list[str]:
    return sorted(str(p.relative_to(home)) for p in home.rglob("*") if p.is_file() and "kanban" in p.name)


def _parse_and_run(argv) -> int:
    import argparse
    top = argparse.ArgumentParser(prog="t")
    parser = kc.build_parser(top.add_subparsers(dest="_top"))
    return kc.kanban_command(parser.parse_args(argv))


_BAD_GUARD_CALLS = [
    ("unblock", "t_a", "--expect-guard", ""),
    ("unblock", "t_a", "--expect-guard", "nothex"),
    ("unblock", "t_a", "t_b", "--expect-guard", "0" * 64),
    ("block", "t_a", "why", "--expect-guard", ""),
    ("block", "t_a", "why", "--expect-guard", "nothex"),
    ("block", "t_a", "--ids", "t_b", "--expect-guard", "0" * 64),
]


@pytest.mark.parametrize("argv", _BAD_GUARD_CALLS)
def test_parser_and_command_refuse_a_bad_guard_without_creating_the_store(fresh_home, argv):
    assert _parse_and_run(list(argv)) == 2
    assert _store_files(fresh_home) == []


@pytest.mark.parametrize("argv", _BAD_GUARD_CALLS)
def test_cli_process_refuses_a_bad_guard_without_creating_the_store(fresh_home, argv):
    r = _cli(fresh_home, *argv)
    assert r.returncode == 2 and "--expect-guard" in r.stderr and not r.stdout
    assert _store_files(fresh_home) == []


@pytest.mark.parametrize("argv", [("unblock", "t_a", "--expect-guard", "0" * 64), ("unblock", "t_a"),
                                  ("block", "t_a", "why", "--expect-guard", "0" * 64), ("block", "t_a", "why")])
def test_valid_or_plain_calls_still_auto_init_the_store(fresh_home, argv):
    assert _parse_and_run(list(argv)) == 1  # unknown card: refused by the op, after init
    assert any(f.startswith("kanban.db") for f in _store_files(fresh_home))
