"""``complete_task`` rejections and audits belong to the run that owns the card.

A worker whose run was superseded (block -> unblock -> a successor claim) still holds
its old ``expected_run_id``. Whatever it submits -- phantom ``created_cards``, no
evidence, valid evidence, acceptance metadata, scratch artifacts -- must be refused
without touching a single row the successor owns: the content gates audit rejections
only for a caller that still owns the run (or has no run claim at all, i.e. a human).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_pr_acceptance_store as acceptance_store

PHANTOM = "t_00000000dead"
BOUND_S = 10.0  # upper bound for every cross-thread wait; a healthy run never nears it

# What a superseded worker may submit; every variant must be refused identically.
STALE_SUBMISSIONS = {
    "phantom_cards": {"summary": "finished", "created_cards": [PHANTOM]},
    "empty_summary": {"summary": "   "},
    "no_evidence": {},
    "valid_evidence": {"summary": "finished the work", "result": "all green"},
}
STALE_PARAMS = pytest.mark.parametrize(
    "submission", list(STALE_SUBMISSIONS.values()), ids=list(STALE_SUBMISSIONS))


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _db_state(conn) -> dict[str, list[tuple]]:
    """Every row of every table, so 'nothing changed' covers events/runs/attachments too."""
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]
    return {
        t: sorted((tuple(r) for r in conn.execute(f"SELECT * FROM {t}")), key=repr)
        for t in tables
    }


def _kinds(conn, tid: str) -> list[str]:
    return [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]


def _hand_over(conn, tid: str, old_run_id: int) -> int:
    """Supersede ``old_run_id`` through the native lifecycle: the successor really claims."""
    assert kb.block_task(conn, tid, reason="reassigned", expected_run_id=old_run_id)
    assert kb.unblock_task(conn, tid)
    successor = kb.claim_task(conn, tid)
    assert successor is not None and successor.current_run_id != old_run_id
    return successor.current_run_id


def _claimed(conn, **create_kwargs) -> tuple[str, int]:
    tid = kb.create_task(conn, title="demo", assignee="coder", **create_kwargs)
    return tid, kb.claim_task(conn, tid).current_run_id


def _superseded(conn, **create_kwargs) -> tuple[str, int, int]:
    tid, old_run = _claimed(conn, **create_kwargs)
    return tid, old_run, _hand_over(conn, tid, old_run)


# --- superseded callers change nothing ------------------------------------------------

@STALE_PARAMS
def test_superseded_worker_completion_leaves_database_untouched(conn, submission):
    tid, old_run, new_run = _superseded(conn)
    before = _db_state(conn)

    assert kb.complete_task(conn, tid, expected_run_id=old_run, **submission) is False

    assert _db_state(conn) == before
    task = kb.get_task(conn, tid)
    assert (task.status, task.current_run_id) == ("running", new_run)


@STALE_PARAMS
def test_run_claim_on_card_without_a_run_leaves_database_untouched(conn, submission):
    tid = kb.create_task(conn, title="never claimed", assignee="coder")
    before = _db_state(conn)

    assert kb.complete_task(conn, tid, expected_run_id=987654, **submission) is False

    assert _db_state(conn) == before
    assert kb.get_task(conn, tid).status == "ready"


@STALE_PARAMS
def test_run_claim_on_unknown_card_leaves_database_untouched(conn, submission):
    kb.create_task(conn, title="bystander", assignee="coder")
    before = _db_state(conn)

    assert kb.complete_task(conn, "t_00000000beef", expected_run_id=1, **submission) is False

    assert _db_state(conn) == before


def test_superseded_worker_acceptance_submission_leaves_database_untouched(conn, monkeypatch):
    def _no_gh(*_a, **_k):
        raise AssertionError("a superseded run must not spend an acceptance lookup")

    monkeypatch.setattr(acceptance_store, "collect_acceptance", _no_gh)
    tid, old_run, new_run = _superseded(conn, completion_contract="acme/repo")
    before = _db_state(conn)

    assert kb.complete_task(
        conn, tid, summary="published", expected_run_id=old_run,
        metadata={"published_pr": "https://github.com/acme/repo/pull/7"},
    ) is False

    assert _db_state(conn) == before
    assert kb.get_task(conn, tid).current_run_id == new_run


def test_superseded_worker_artifacts_are_not_staged(conn):
    tid = kb.create_task(conn, title="render", assignee="coder")
    workspace = kbw.resolve_workspace(kb.get_task(conn, tid))
    kbw.set_workspace_path(conn, tid, workspace)
    artifact = workspace / "chart.png"
    artifact.write_bytes(b"png-bytes")
    old_run = kb.claim_task(conn, tid).current_run_id
    new_run = _hand_over(conn, tid, old_run)
    before = _db_state(conn)

    assert kb.complete_task(
        conn, tid, summary="rendered", expected_run_id=old_run,
        metadata={"artifacts": [str(artifact)]},
    ) is False

    assert _db_state(conn) == before
    assert artifact.read_bytes() == b"png-bytes"
    attachments = kb.task_attachments_dir(tid)
    assert not attachments.exists() or not any(attachments.iterdir())
    assert kb.get_task(conn, tid).current_run_id == new_run


# --- legitimate callers keep their audits and their completions -----------------------

def test_current_owner_phantom_cards_rejection_is_audited_then_retry_completes(conn):
    tid, run = _claimed(conn)

    with pytest.raises(kb.HallucinatedCardsError):
        kb.complete_task(conn, tid, summary="done", created_cards=[PHANTOM], expected_run_id=run)

    task = kb.get_task(conn, tid)
    assert (task.status, task.current_run_id) == ("running", run)
    assert _kinds(conn, tid).count("completion_blocked_hallucination") == 1
    audit = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completion_blocked_hallucination'",
        (tid,),
    ).fetchone()
    assert json.loads(audit["payload"])["phantom_cards"] == [PHANTOM]

    assert kb.complete_task(conn, tid, summary="done", expected_run_id=run) is True
    done = conn.execute(
        "SELECT run_id FROM task_events WHERE task_id = ? AND kind = 'completed'", (tid,)).fetchone()
    assert done["run_id"] == run
    assert conn.execute("SELECT outcome FROM task_runs WHERE id = ?", (run,)).fetchone()["outcome"] == "completed"


def test_current_owner_empty_completion_rejection_is_audited(conn):
    tid, run = _claimed(conn)

    with pytest.raises(kb.EmptyCompletionError):
        kb.complete_task(conn, tid, summary="  ", expected_run_id=run)

    assert kb.get_task(conn, tid).current_run_id == run
    assert _kinds(conn, tid).count("completion_blocked_empty_result") == 1
    assert kb.complete_task(conn, tid, summary="now with evidence", expected_run_id=run) is True


def test_successor_owner_completes_after_old_worker_was_refused(conn):
    tid, old_run, new_run = _superseded(conn)
    assert kb.complete_task(conn, tid, summary="stale", expected_run_id=old_run) is False

    assert kb.complete_task(conn, tid, summary="successor result", expected_run_id=new_run) is True

    assert kb.get_task(conn, tid).status == "done"
    assert conn.execute("SELECT outcome FROM task_runs WHERE id = ?", (new_run,)).fetchone()["outcome"] == "completed"


def test_human_without_run_claim_still_gets_rejection_audits_and_can_complete(conn):
    phantom_tid = kb.create_task(conn, title="phantom", assignee="coder")
    with pytest.raises(kb.HallucinatedCardsError):
        kb.complete_task(conn, phantom_tid, summary="done", created_cards=[PHANTOM])
    assert "completion_blocked_hallucination" in _kinds(conn, phantom_tid)

    empty_tid = kb.create_task(conn, title="empty", assignee="coder")
    with pytest.raises(kb.EmptyCompletionError):
        kb.complete_task(conn, empty_tid)
    assert "completion_blocked_empty_result" in _kinds(conn, empty_tid)

    assert kb.complete_task(conn, empty_tid, result="closed by operator") is True
    assert kb.get_task(conn, empty_tid).status == "done"


# --- the ownership check must hold across the unlocked-check -> write seam ------------

@STALE_PARAMS
def test_successor_claim_between_preflight_and_write_is_not_mutated(conn, monkeypatch, submission):
    """The successor claims on its OWN connection at the moment the old worker is about to
    take its first writer reservation, i.e. after every unlocked read it made. Whatever the
    implementation does there (re-verify under the reservation, or reserve first), nothing
    the old worker submitted may land on the successor's rows."""
    tid, old_run = _claimed(conn)
    at_seam, successor_done = threading.Event(), threading.Event()
    outcome: dict = {}

    def successor():
        try:
            assert at_seam.wait(BOUND_S), "old worker never reached its write seam"
            with kbc.connect_closing() as rival:
                outcome["run"] = _hand_over(rival, tid, old_run)
        except BaseException as exc:  # reported on the main thread
            outcome["error"] = exc
        finally:
            successor_done.set()

    thread = threading.Thread(target=successor, daemon=True)
    thread.start()

    real_write_txn = kb.write_txn
    seam = {"armed": True}

    def write_txn_after_preflight(c, *args, **kwargs):
        if c is conn and seam["armed"]:
            seam["armed"] = False
            at_seam.set()
            assert successor_done.wait(BOUND_S), "successor claim did not finish"
            seam["state"] = _db_state(conn)
        return real_write_txn(c, *args, **kwargs)

    monkeypatch.setattr(kb, "write_txn", write_txn_after_preflight)

    result = kb.complete_task(conn, tid, expected_run_id=old_run, **submission)

    thread.join(BOUND_S)
    assert not thread.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert "state" in seam, "the old worker never reached a writer reservation"
    assert result is False
    assert _db_state(conn) == seam["state"]
    task = kb.get_task(conn, tid)
    assert (task.status, task.current_run_id) == ("running", outcome["run"])

    monkeypatch.setattr(kb, "write_txn", real_write_txn)
    assert kb.complete_task(conn, tid, summary="successor result", expected_run_id=outcome["run"]) is True
