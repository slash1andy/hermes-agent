"""``unblock_if_guard(related=...)``: ``None`` alone means "no second card"; every SUPPLIED binding must be a valid pair or nothing is written.

A supplied ``related`` that is empty, falsey, the wrong container or arity, malformed, or the card being unblocked must refuse (``False``)
before any write, never fall back to the legacy single-card release. This is the typed-API boundary the CLI already enforces through
``--expect-related`` parsing; arbitrary hostile Python objects are not a claim here. Real board DB under a temp HERMES_HOME, and every
refusal compares the full SQL rows of all four card tables.
"""
from __future__ import annotations

import argparse
import sqlite3

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_guard import unblock_if_guard
from tests.hermes_cli import test_kanban_crosscard_guard as guard_tests
from tests.hermes_cli.test_kanban_crosscard_guard import G, _cli, _guard, _pair, _show, _store_files

kanban_home = guard_tests.kanban_home  # same fixture object, registered here by name

COMMENT = ("orchestrator", "UNBLOCK: x")


def _board(conn):
    """Every SQL row of every task, event, run and comment on the board."""
    return tuple([tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY 1")] for t in ("tasks", "task_events", "task_runs", "task_comments"))


# (case id, builder(rev, impl, g_rev, g_impl) -> the supplied ``related``)
_SUPPLIED = {
    # falsey scalars and containers
    "empty-tuple": lambda r, i, gr, gi: (),
    "empty-list": lambda r, i, gr, gi: [],
    "empty-dict": lambda r, i, gr, gi: {},
    "empty-str": lambda r, i, gr, gi: "",
    "empty-bytes": lambda r, i, gr, gi: b"",
    "false": lambda r, i, gr, gi: False,
    "zero": lambda r, i, gr, gi: 0,
    "zero-float": lambda r, i, gr, gi: 0.0,
    # truthy wrong containers / scalars (a list or mapping of the right arity would unpack)
    "list-pair": lambda r, i, gr, gi: [i, gi],
    "dict-pair": lambda r, i, gr, gi: {i: gi},
    "set-pair": lambda r, i, gr, gi: {i, gi},
    "two-char-str": lambda r, i, gr, gi: "ab",
    "joined-str": lambda r, i, gr, gi: f"{i}:{gi}",
    "pair-bytes": lambda r, i, gr, gi: f"{i}:{gi}".encode(),
    "true": lambda r, i, gr, gi: True,
    "one": lambda r, i, gr, gi: 1,
    # wrong arity
    "one-tuple": lambda r, i, gr, gi: (i,),
    "three-tuple": lambda r, i, gr, gi: (i, gi, gi),
    "three-tuple-none": lambda r, i, gr, gi: (i, gi, None),
    # wrong inner types
    "none-id": lambda r, i, gr, gi: (None, gi),
    "none-token": lambda r, i, gr, gi: (i, None),
    "int-id": lambda r, i, gr, gi: (1, gi),
    "int-token": lambda r, i, gr, gi: (i, 1),
    "bytes-id": lambda r, i, gr, gi: (i.encode(), gi),
    "bytes-token": lambda r, i, gr, gi: (i, gi.encode()),
    "list-token": lambda r, i, gr, gi: (i, [gi]),
    # malformed id / token strings (no normalization)
    "empty-id": lambda r, i, gr, gi: ("", gi),
    "empty-token": lambda r, i, gr, gi: (i, ""),
    "short-token": lambda r, i, gr, gi: (i, gi[:63]),
    "long-token": lambda r, i, gr, gi: (i, gi + "0"),
    "uppercase-token": lambda r, i, gr, gi: (i, gi.upper().replace("0", "A")),
    "space-before-token": lambda r, i, gr, gi: (i, " " + gi),
    "newline-after-token": lambda r, i, gr, gi: (i, gi + "\n"),
    "space-in-id": lambda r, i, gr, gi: (" " + i, gi),
    "newline-after-id": lambda r, i, gr, gi: (i + "\n", gi),
    "colon-in-id": lambda r, i, gr, gi: (i + ":" + gi, gi),
    # the card being unblocked is never its own second card
    "self-current-guard": lambda r, i, gr, gi: (r, gr),
    "self-implementation-guard": lambda r, i, gr, gi: (r, gi),
    "self-zero-token": lambda r, i, gr, gi: (r, "0" * 64),
}


@pytest.mark.parametrize("case", list(_SUPPLIED))
def test_a_supplied_but_invalid_related_binding_refuses_and_writes_nothing(kanban_home, case):
    with kbc.connect_closing() as conn:
        rev, impl = _pair(conn)
        g_rev, g_impl = _guard(conn, rev), _guard(conn, impl)
        related = _SUPPLIED[case](rev, impl, g_rev, g_impl)
        before = _board(conn)
        assert unblock_if_guard(conn, rev, g_rev, comment=COMMENT, related=related) is False, case
        assert _board(conn) == before and kb.get_task(conn, rev).status == "blocked" and not conn.in_transaction


def test_none_is_the_only_legacy_single_card_sentinel(kanban_home):
    with kbc.connect_closing() as conn:
        rev, _ = _pair(conn)
        runs_before = _board(conn)[2]
        assert unblock_if_guard(conn, rev, _guard(conn, rev), comment=COMMENT, related=None) is True
        assert kb.get_task(conn, rev).status == "ready" and [c.body for c in kb.list_comments(conn, rev)] == ["UNBLOCK: x"]
        assert _board(conn)[2] == runs_before  # the release wrote no run row
    with kbc.connect_closing() as conn:
        rev2, _ = _pair(conn)
        assert unblock_if_guard(conn, rev2, _guard(conn, rev2), comment=COMMENT) is True  # omitted == None


def test_a_valid_other_card_pair_still_releases_and_leaves_the_other_card_untouched(kanban_home):
    with kbc.connect_closing() as conn:
        rev, impl = _pair(conn)
        g_rev, g_impl = _guard(conn, rev), _guard(conn, impl)
        impl_before = [tuple(r) for r in conn.execute("SELECT * FROM tasks WHERE id = ?", (impl,))]
        assert unblock_if_guard(conn, rev, g_rev, comment=COMMENT, related=(impl, g_impl)) is True
        assert kb.get_task(conn, rev).status == "ready" and [c.body for c in kb.list_comments(conn, rev)] == ["UNBLOCK: x"]
        assert [tuple(r) for r in conn.execute("SELECT * FROM tasks WHERE id = ?", (impl,))] == impl_before


def test_a_failed_comment_after_a_valid_pair_rolls_back_every_row(kanban_home, monkeypatch):
    with kbc.connect_closing() as conn:
        rev, impl = _pair(conn)
        g_rev, g_impl = _guard(conn, rev), _guard(conn, impl)
        before = _board(conn)
        monkeypatch.setattr(kb, "add_comment", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            unblock_if_guard(conn, rev, g_rev, comment=COMMENT, related=(impl, g_impl))
        monkeypatch.undo()
        assert _board(conn) == before and not conn.in_transaction


def test_the_guarded_helper_refuses_inside_a_caller_transaction_and_leaves_that_transaction_intact(kanban_home):
    """``write_txn`` is ``allow_nested=False`` here: an open caller transaction is a loud RuntimeError, not a nested release."""
    with kbc.connect_closing() as conn:
        rev, impl = _pair(conn)
        g_rev, g_impl = _guard(conn, rev), _guard(conn, impl)
        before = _board(conn)
        insert = "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, 'caller', ?, 7)"
        with kbc.write_txn(conn):
            conn.execute(insert, (rev, "caller write"))
            conn.execute("SAVEPOINT caller_sp")
            conn.execute(insert, (rev, "inside caller savepoint"))
            with pytest.raises(RuntimeError, match="already inside a transaction"):
                unblock_if_guard(conn, rev, g_rev, comment=COMMENT, related=(impl, g_impl))
            assert conn.in_transaction and kb.get_task(conn, rev).status == "blocked"
            assert [c.body for c in kb.list_comments(conn, rev)] == ["caller write", "inside caller savepoint"]
            conn.execute("ROLLBACK TO caller_sp")  # the caller's own savepoint is still open and still theirs
            conn.execute("RELEASE caller_sp")
        after = _board(conn)
        assert [r[3] for r in after[3] if r[1] == rev] == ["caller write"]
        assert after[:3] == before[:3] and kb.get_task(conn, rev).status == "blocked"


# --- the real CLI and orchestrator path ------------------------------------------------------------------------------------------

def _unblock_argv(rev, g_rev, related):
    return ["unblock", rev, "--expect-guard", g_rev, "--expect-related", related, "--reason", "go"]


@pytest.mark.parametrize("bad", ["self", "malformed", "unknown-card"])
def test_cli_refusals_leave_an_existing_board_unchanged(kanban_home, bad):
    with kbc.connect_closing() as conn:
        rev, impl = _pair(conn)
        g_rev, g_impl = _guard(conn, rev), _guard(conn, impl)
        before = _board(conn)
    related = {"self": f"{rev}:{g_rev}", "malformed": f"{impl}:{g_impl.upper().replace('0', 'A')}", "unknown-card": f"t_nope0:{g_impl}"}[bad]
    r = _cli(kanban_home, *_unblock_argv(rev, g_rev, related))
    assert r.returncode == (1 if bad == "unknown-card" else 2) and not r.stdout
    with kbc.connect_closing() as conn:
        assert _board(conn) == before


def test_orchestrator_command_releases_with_a_valid_pair_and_a_legacy_guard(kanban_home):
    with kbc.connect_closing() as conn:
        rev, impl = _pair(conn)
        rev2, _ = _pair(conn)
    parser = kc.build_parser(argparse.ArgumentParser(prog="t").add_subparsers(dest="_top"))
    pair = _unblock_argv(rev, _show(kanban_home, rev)["block_guard"], f"{impl}:{_show(kanban_home, impl)['block_guard']}")
    assert kc.kanban_command(parser.parse_args(pair)) == 0
    assert kc.kanban_command(parser.parse_args(["unblock", rev2, "--expect-guard", _show(kanban_home, rev2)["block_guard"]])) == 0
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, rev).status == kb.get_task(conn, rev2).status == "ready"


def test_a_malformed_cli_binding_never_creates_the_store(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    r = _cli(home, "unblock", "t_a", "--expect-guard", G, "--expect-related", f"t_a:{G}")
    assert r.returncode == 2 and _store_files(home) == []
