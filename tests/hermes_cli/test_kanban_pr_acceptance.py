"""Lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

GYMCORE_REPO = "GymCoreHQ/gymcore"
GYMCORE_APP_ID = 15368
GYMCORE_CHECKS = (
    "composer test-all + metadata",
    "gym-core-ai PHPUnit + PHPStan",
    "changed-asset lint + build",
    "root static pytest",
    "customer-docs",
    "php -l",
)


def _executable_dir():
    """A directory guaranteed exec-capable for the `gh` shim's shebang script.

    pytest's tmp_path can land on a noexec-mounted tmpfs; PATH lookup then
    silently skips the non-executable shim and falls through to a real `gh`
    elsewhere on PATH, which is indistinguishable from a passing empty test
    until the assertions on captured requests fail. Caller removes the dir.
    """
    return tempfile.mkdtemp(dir=str(Path(__file__).resolve().parent))


def _write_gh_shim(gh_path: Path, port: int) -> None:
    gh_path.write_text(
        f"#!{sys.executable}\n"
        "import sys, json, urllib.request, urllib.error\n"
        f"u = 'http://127.0.0.1:{port}/' + sys.argv[2]\n"
        "try:\n"
        "    print(urllib.request.urlopen(u).read().decode())\n"
        "except urllib.error.HTTPError as e:\n"
        "    try:\n"
        "        msg = json.loads(e.read().decode()).get('message', '')\n"
        "    except Exception:\n"
        "        msg = ''\n"
        "    sys.stderr.write('gh: %s (HTTP %d)\\n' % (msg, e.code))\n"
        "    sys.exit(1)\n"
    )
    gh_path.chmod(0o755)


def _last_receipt(conn, tid):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC", (tid,)
    ).fetchone()
    return json.loads(row[0]) if row else None


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                value = [[]]
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                value = [{"total_count": 100 + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(100)]}, {"total_count": 100 + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = [[]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim_dir = _executable_dir()
    _write_gh_shim(Path(shim_dir) / "gh", server.server_port)
    monkeypatch.setenv("PATH", shim_dir + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        shutil.rmtree(shim_dir, ignore_errors=True)


@pytest.fixture
def github_gymcore(tmp_path, monkeypatch):
    """A local GitHub HTTP contract for GymCoreHQ/gymcore's explicit six-check policy."""
    state = {
        "head": "a" * 40, "branch": "main", "requests": [],
        "conclusions": {c: "success" for c in GYMCORE_CHECKS},
        "run_app_ids": {c: GYMCORE_APP_ID for c in GYMCORE_CHECKS},
        "omit_checks": set(), "extra_required": [], "statuses": [],
        "rules_error": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": state["branch"], "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": []}}}}}}
            elif "/rules/branches/" in self.path:
                if state["rules_error"] == "plan":
                    self._error(403, "Upgrade to GitHub Pro or make this repository public to enable this feature.")
                    return
                if state["rules_error"] == "generic":
                    self._error(403, "Resource not accessible by integration")
                    return
                extra = [{"context": c, "integration_id": a} for c, a in state["extra_required"]]
                value = [[{"type": "required_status_checks", "parameters": {"required_status_checks": extra}}]] if extra else [[]]
            elif "/check-runs" in self.path:
                names = [c for c in GYMCORE_CHECKS if c not in state["omit_checks"]]
                names += [c for c, _ in state["extra_required"] if c not in state["omit_checks"]]
                runs = []
                for i, name in enumerate(names):
                    conclusion = state["conclusions"].get(name, "success")
                    runs.append({"id": 100 + i, "name": name, "head_sha": sha,
                                 "app": {"id": state["run_app_ids"].get(name, GYMCORE_APP_ID)},
                                 "status": "in_progress" if conclusion == "pending" else "completed",
                                 "conclusion": conclusion,
                                 "html_url": f"https://github.com/{GYMCORE_REPO}/actions/runs/{100 + i}"})
                filler = {"id": 0, "name": "optional", "head_sha": sha, "app": {"id": GYMCORE_APP_ID},
                          "status": "completed", "conclusion": "skipped", "html_url": ""}
                value = [{"total_count": 100 + len(runs), "check_runs": [{**filler, "id": 1000 + i} for i in range(100)]},
                         {"total_count": 100 + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
            elif "/statuses" in self.path:
                value = [state["statuses"]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": state["branch"]}, "state": "open"}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def _error(self, code, message):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps({"message": message, "status": str(code)}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim_dir = _executable_dir()
    _write_gh_shim(Path(shim_dir) / "gh", server.server_port)
    monkeypatch.setenv("PATH", shim_dir + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        shutil.rmtree(shim_dir, ignore_errors=True)


@pytest.mark.linux_only
def test_pr_completion_requires_current_required_evidence(github):
    with connect() as conn:
        for conclusion in ("failure", "pending", "cancelled", "timed_out", "action_required", "neutral", "skipped", None, "success"):
            github.update(conclusion=conclusion, head="a" * 40)
            tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
            ok = kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert ok is (conclusion == "success")
            task = kb.get_task(conn, tid)
            assert (task.status == "done") is ok
            receipts = [json.loads(r[0]) for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
            assert receipts and receipts[-1]["head_sha"] == "a" * 40
            if not ok:
                assert task.status in {"running", "ready", "blocked", "review"}
                assert "retry" in receipts[-1]["recovery"]
                assert receipts[-1]["checks"][0]["id"] == 42
        for fault in ("missing", "stale", "head_change"):
            github.update(conclusion="success", head="a" * 40)
            github[fault] = True
            tid = kb.create_task(conn, title=fault, completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).status != "done"
            github.pop(fault)
        # Omission and a sibling repository cannot downgrade the stored declaration.
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, summary="local green")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/other/repo/pull/7"})
        before = len(github["requests"])
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        assert kb.complete_task(conn, local, summary="https://github.com/acme/repo/pull/7 is background context")
        assert len(github["requests"]) == before


@pytest.mark.linux_only
def test_acceptance_receipts_and_terminal_write_share_run_ownership(github):
    with connect() as conn:
        for conclusion in ("success", "failure"):
            tid = kb.create_task(conn, title="race", completion_contract="acme/repo")
            owner = kb.claim_task(conn, tid)
            run_id = owner.current_run_id
            def reclaim():
                with connect() as rival:
                    assert kb.block_task(rival, tid, reason="Reassigned during acceptance")
                    assert kb.unblock_task(rival, tid)
                    github["replacement"] = kb.claim_task(rival, tid).current_run_id
            github.update(conclusion=conclusion, race=reclaim)
            assert not kb.complete_task(conn, tid, expected_run_id=run_id,
                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).current_run_id == github["replacement"]
            assert github["replacement"] != run_id
            assert kb.get_task(conn, tid).status != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")


@pytest.mark.linux_only
def test_gymcore_policy_requires_all_six_actions_checks_at_head(github_gymcore):
    pr_url = f"https://github.com/{GYMCORE_REPO}/pull/7"
    with connect() as conn:
        tid = kb.create_task(conn, title="gymcore all green", completion_contract=GYMCORE_REPO)
        assert kb.complete_task(conn, tid, metadata={"published_pr": pr_url})
        receipt = _last_receipt(conn, tid)
        assert receipt["policy_source"] == f"explicit:{GYMCORE_REPO}@main"
        assert {c["context"] for c in receipt["required"]} == set(GYMCORE_CHECKS)
        assert all(c["source"] == "policy" and c["app_id"] == GYMCORE_APP_ID for c in receipt["required"])

        # Each of the six required checks, failing alone, blocks completion.
        for failing in GYMCORE_CHECKS:
            github_gymcore["conclusions"] = {c: ("failure" if c == failing else "success") for c in GYMCORE_CHECKS}
            tid = kb.create_task(conn, title=f"fail {failing}", completion_contract=GYMCORE_REPO)
            assert not kb.complete_task(conn, tid, metadata={"published_pr": pr_url})
            assert kb.get_task(conn, tid).status != "done"
        github_gymcore["conclusions"] = {c: "success" for c in GYMCORE_CHECKS}


@pytest.mark.linux_only
def test_gymcore_policy_rejects_wrong_app_and_legacy_status(github_gymcore):
    pr_url = f"https://github.com/{GYMCORE_REPO}/pull/7"
    with connect() as conn:
        # A check run under the wrong GitHub App can never satisfy an app-pinned context.
        github_gymcore["run_app_ids"]["php -l"] = 999
        tid = kb.create_task(conn, title="wrong app", completion_contract=GYMCORE_REPO)
        assert not kb.complete_task(conn, tid, metadata={"published_pr": pr_url})
        github_gymcore["run_app_ids"]["php -l"] = GYMCORE_APP_ID

        # Nor can a legacy commit status spoofing the same context name.
        github_gymcore["omit_checks"] = {"php -l"}
        github_gymcore["statuses"] = [{"id": 1, "context": "php -l", "state": "success"}]
        tid = kb.create_task(conn, title="legacy spoof", completion_contract=GYMCORE_REPO)
        assert not kb.complete_task(conn, tid, metadata={"published_pr": pr_url})
        github_gymcore["omit_checks"] = set()
        github_gymcore["statuses"] = []


@pytest.mark.linux_only
def test_gymcore_policy_scoped_to_repo_and_branch(github_gymcore):
    pr_url = f"https://github.com/{GYMCORE_REPO}/pull/7"
    with connect() as conn:
        # A different base branch on the same repo does not get the hardcoded floor;
        # with nothing server-declared either, it fails closed on an empty requirement set.
        github_gymcore["branch"] = "develop"
        tid = kb.create_task(conn, title="non-main branch", completion_contract=GYMCORE_REPO)
        assert not kb.complete_task(conn, tid, metadata={"published_pr": pr_url})
        receipt = _last_receipt(conn, tid)
        assert "policy_source" not in receipt
        assert receipt["classification"] == "missing"
        github_gymcore["branch"] = "main"


@pytest.mark.linux_only
def test_gymcore_plan_limited_403_falls_back_to_policy_floor(github_gymcore):
    pr_url = f"https://github.com/{GYMCORE_REPO}/pull/7"
    with connect() as conn:
        # The documented Free-plan Rulesets 403 is not an infra failure here — the
        # explicit six-check policy is the floor regardless of discovery.
        github_gymcore["rules_error"] = "plan"
        tid = kb.create_task(conn, title="plan limited", completion_contract=GYMCORE_REPO)
        assert kb.complete_task(conn, tid, metadata={"published_pr": pr_url})
        receipt = _last_receipt(conn, tid)
        assert receipt["server_discovery"].startswith("unavailable")
        assert receipt["ok"]

        # Any other 403 (bad auth, rate limit, ...) still fails closed as infra.
        github_gymcore["rules_error"] = "generic"
        tid = kb.create_task(conn, title="generic 403", completion_contract=GYMCORE_REPO)
        assert not kb.complete_task(conn, tid, metadata={"published_pr": pr_url})
        receipt = _last_receipt(conn, tid)
        assert receipt["classification"] == "infra"
        assert "server_discovery" not in receipt
        github_gymcore["rules_error"] = None


@pytest.mark.linux_only
def test_gymcore_policy_unions_stricter_server_requirements(github_gymcore):
    pr_url = f"https://github.com/{GYMCORE_REPO}/pull/7"
    with connect() as conn:
        # A server-discovered requirement beyond the hardcoded six is never discarded.
        github_gymcore["extra_required"] = [("security-scan", None)]
        github_gymcore["conclusions"]["security-scan"] = "failure"
        tid = kb.create_task(conn, title="stricter missing", completion_contract=GYMCORE_REPO)
        assert not kb.complete_task(conn, tid, metadata={"published_pr": pr_url})
        receipt = _last_receipt(conn, tid)
        assert any(c["context"] == "security-scan" and c["source"] == "server" for c in receipt["required"])
        assert all(c["source"] == "policy" for c in receipt["required"] if c["context"] in GYMCORE_CHECKS)

        github_gymcore["conclusions"]["security-scan"] = "success"
        tid = kb.create_task(conn, title="stricter met", completion_contract=GYMCORE_REPO)
        assert kb.complete_task(conn, tid, metadata={"published_pr": pr_url})
        github_gymcore["extra_required"] = []


@pytest.mark.linux_only
def test_gymcore_policy_receipt_and_terminal_write_share_run_ownership(github_gymcore):
    pr_url = f"https://github.com/{GYMCORE_REPO}/pull/7"
    with connect() as conn:
        tid = kb.create_task(conn, title="gymcore race", completion_contract=GYMCORE_REPO)
        owner = kb.claim_task(conn, tid)
        run_id = owner.current_run_id
        def reclaim():
            with connect() as rival:
                assert kb.block_task(rival, tid, reason="Reassigned during acceptance")
                assert kb.unblock_task(rival, tid)
                github_gymcore["replacement"] = kb.claim_task(rival, tid).current_run_id
        github_gymcore["race"] = reclaim
        assert not kb.complete_task(conn, tid, expected_run_id=run_id, metadata={"published_pr": pr_url})
        assert kb.get_task(conn, tid).current_run_id == github_gymcore["replacement"]
        assert github_gymcore["replacement"] != run_id
        assert kb.get_task(conn, tid).status != "done"
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
