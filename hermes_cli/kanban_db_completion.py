"""``complete_task`` preflight: the run-ownership fence and the content gates.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` (``write_txn`` included) keeps working.

A caller holding ``expected_run_id`` may only audit or complete a card while that
run still owns it. A rejection audit is a write, so each one re-checks ownership
inside its own writer reservation; the unlocked reads before it only save work.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Optional


def run_is_owned(conn: sqlite3.Connection, task_id: str, expected_run_id: Optional[int]) -> bool:
    """True when the caller makes no run claim (a human) or ``expected_run_id`` is
    the card's current run. Authorizes a write only when called under the
    writer reservation that covers it."""
    if expected_run_id is None:
        return True
    row = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return row is not None and row["current_run_id"] == int(expected_run_id)


def gate_created_cards(
    conn: sqlite3.Connection, task_id: str, created_cards: Optional[Iterable[str]],
    preview_text: Optional[str], expected_run_id: Optional[int],
) -> Optional[list[str]]:
    """Verify ``created_cards`` BEFORE the main write txn; returns the verified
    ids. A phantom id is recorded in its own tiny txn (auditable) then raised
    as :class:`HallucinatedCardsError` without touching task state. ``None``
    when the run was superseded: nothing is audited."""
    if not created_cards:
        return []
    verified_cards, phantom_cards = _kb._verify_created_cards(conn, task_id, created_cards)
    if phantom_cards:
        with _kb.write_txn(conn):
            if not run_is_owned(conn, task_id, expected_run_id):
                return None
            _kb._append_event(
                conn, task_id, "completion_blocked_hallucination",
                {
                    "phantom_cards": phantom_cards,
                    "verified_cards": verified_cards,
                    "summary_preview": _kb._first_line(preview_text, 200) or None,
                },
            )
        raise _kb.HallucinatedCardsError(phantom_cards, task_id)
    return verified_cards


def _substantive_text(value: Optional[str]) -> bool:
    return bool(value is not None and str(value).strip())


def gate_empty_completion(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: Optional[str],
    summary: Optional[str],
    expected_run_id: Optional[int],
) -> bool:
    """Refuse a completion that would leave the card with no evidence; False
    when the run was superseded (nothing is audited).

    Review approvals are exempt: a human vouches for the card and
    ``_REVIEW_APPROVED_NOTE`` is the documented record.
    """
    row = conn.execute(
        "SELECT status, result FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return True
    if row["status"] == "review":
        return True
    stored = row["result"]
    if _substantive_text(result) or _substantive_text(summary) or _substantive_text(stored):
        return True
    with _kb.write_txn(conn):
        if not run_is_owned(conn, task_id, expected_run_id):
            return False
        _kb._append_event(
            conn, task_id, "completion_blocked_empty_result",
            {
                "result_preview": _kb._first_line(result, 200) or None,
                "summary_preview": _kb._first_line(summary, 200) or None,
            },
        )
    raise _kb.EmptyCompletionError(task_id)


def prepare_completion(
    conn: sqlite3.Connection, task_id: str, *, result: Optional[str], summary: Optional[str],
    metadata: Optional[dict], created_cards: Optional[Iterable[str]], expected_run_id: Optional[int],
):
    """Everything ``complete_task`` decides before its write txn, none of it
    holding the writer over external acceptance lookups.

    Returns ``(verified_cards, metadata, acceptance)``, or ``None`` when the
    completion is refused (unsatisfied parents, superseded run, rejected
    acceptance preflight) with no durable side effect. The unlocked checks here
    only short-circuit; the write txn's CAS on ``current_run_id`` is what
    authorizes the terminal write.
    """
    from hermes_cli.kanban_pr_acceptance_store import prepare_acceptance

    # Cheap pre-check; re-checked inside the txn to close the parent-reopen race.
    if not _kb._parents_satisfied(conn, task_id) or not run_is_owned(conn, task_id, expected_run_id):
        return None
    verified_cards = gate_created_cards(conn, task_id, created_cards, summary or result, expected_run_id)
    if verified_cards is None:
        return None
    if not gate_empty_completion(
        conn, task_id, result=result, summary=summary, expected_run_id=expected_run_id,
    ):
        return None
    metadata = _kb._merge_completion_prose_artifacts(
        conn, task_id, metadata, summary=summary, result=result,
    )
    acceptance = prepare_acceptance(conn, task_id, expected_run_id, metadata)
    if acceptance is False:
        return None
    return verified_cards, metadata, acceptance


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb
