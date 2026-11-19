"""Guarded block/unblock: change a card only if it is still in the state a caller read.

``kanban show --json`` publishes ``block_guard`` over the task, event and run rows it just listed;
``unblock``/``block --expect-guard`` recompute it inside the write transaction, so a change that arrived
after the caller classified the card (a human ``needs_input``, a re-block, a new or corrected run, a claim)
refuses the write instead of acting on a stale read. ``unblock --expect-related TASK:TOKEN`` validates a second
card's guard in the SAME transaction (a release decided on another card's handoff).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

_TOKEN = re.compile(r"[0-9a-f]{64}")
_RELATED = re.compile(r"([A-Za-z0-9_.-]+):([0-9a-f]{64})")
# Same run fields ``kanban show --json`` prints (kanban_output._SHOW_RUN_FIELDS).
_RUN_FIELDS = ("id", "profile", "step_key", "status", "outcome", "summary", "error",
               "metadata", "worker_pid", "started_at", "ended_at")


def is_guard_token(value) -> bool:
    return isinstance(value, str) and _TOKEN.fullmatch(value) is not None


def parse_related(value) -> Optional[tuple[str, str]]:
    """``TASK:TOKEN`` -> ``(task_id, token)``; anything else (including whitespace or case changes) -> None."""
    m = _RELATED.fullmatch(value) if isinstance(value, str) else None
    return (m[1], m[2]) if m else None


def block_guard(task, events, runs) -> str:
    """Opaque token over the task fields, EVERY event and EVERY run row a release decision can depend on.

    The task id (so a token cannot be replayed for another card) and ``workspace_path`` are bound too:
    ``set_workspace_path`` changes the workspace with no event or run row.

    Computed from already-loaded rows, so it can only describe what the caller was shown. Run rows are bound
    because the classification reads run profile/summary/metadata, which can change without any task or event row
    changing (a late run insert, a corrected verdict summary).
    """
    state = [task.id, task.workspace_path, task.status, task.block_kind, task.current_run_id, task.assignee, task.skills,
             [[e.id, e.kind, e.payload, e.run_id] for e in events],
             [[getattr(r, f) for f in _RUN_FIELDS] for r in runs]]
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


def guard_matches(conn: sqlite3.Connection, task_id: str, expect: str) -> bool:
    """Recompute the guard from the rows as they are NOW (call inside the write transaction)."""
    task = kb.get_task(conn, task_id)
    return task is not None and block_guard(
        task, kb.list_events(conn, task_id), kb.list_runs(conn, task_id)) == expect


def _is_other_card_binding(task_id: str, related) -> bool:
    """A ``(task_id, token)`` tuple of two strings that ``parse_related`` accepts unchanged, naming a card other than ``task_id``."""
    return (isinstance(related, tuple) and len(related) == 2 and all(isinstance(v, str) for v in related)
            and parse_related(f"{related[0]}:{related[1]}") == related and related[0] != task_id)


def unblock_if_guard(
    conn: sqlite3.Connection, task_id: str, expect: str, *, comment: Optional[tuple[str, str]] = None,
    related: Optional[tuple[str, str]] = None,
) -> bool:
    """``unblock_task`` iff the guard still matches, atomically with the optional (author, body) comment.
    ``related`` = (task_id, guard) of ANOTHER card on this board that must be unchanged in the same transaction.
    ``None`` alone means no second card; any other value that is not exactly such a pair refuses before anything is written."""
    if related is not None and not _is_other_card_binding(task_id, related):
        return False
    with kbc.write_txn(conn):
        if not guard_matches(conn, task_id, expect) or (related is not None and not guard_matches(conn, *related)):
            return False
        if not kb.unblock_task(conn, task_id):
            return False  # nothing written yet: the comment must not outlive a refused unblock
        if comment:
            kb.add_comment(conn, task_id, *comment)
        return True


def block_if_guard(conn: sqlite3.Connection, task_id: str, expect: str, **block_kwargs) -> bool:
    """``block_task`` iff the guard still matches; the check runs inside block_task's own write transaction
    (``precondition``), so a card claimed, released or held after the caller read it is left untouched."""
    return kb.block_task(conn, task_id, precondition=lambda: guard_matches(conn, task_id, expect), **block_kwargs)
