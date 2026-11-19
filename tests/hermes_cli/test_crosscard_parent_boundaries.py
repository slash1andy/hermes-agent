"""Controller probes: a supplied invalid related binding must not degrade to legacy release."""
from pathlib import Path
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_guard import block_guard, unblock_if_guard

@pytest.mark.parametrize('binding', ['self', 'empty'])
def test_related_binding_cannot_silently_disable_the_second_card(tmp_path, monkeypatch, binding):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='owned boundary probe', assignee='reviewer', created_by='orchestrator', initial_status='blocked')
        token = block_guard(kb.get_task(conn, tid), kb.list_events(conn, tid), kb.list_runs(conn, tid))
        related = (tid, token) if binding == 'self' else ()
        result = unblock_if_guard(conn, tid, token, related=related, comment=('orchestrator', 'must not be published'))
        assert result is False, f'{binding} binding released the review instead of refusing'
        assert kb.get_task(conn, tid).status == 'blocked'
        assert kb.list_comments(conn, tid) == []
