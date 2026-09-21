"""Regression test for the merge-base used by ``contributor-check.yml``.

Background: the workflow hardcoded ``MERGE_BASE=$(git merge-base origin/main
HEAD)``. A PR that targets a non-``main`` base branch (e.g. a stacked PR)
scans every historical commit between the *real* common ancestor with
``main`` and the PR head — including commits the PR never introduced —
and reports their authors as unmapped contributors even though the PR
itself only adds mapped-author commits.

This test extracts the actual ``run:`` shell block for the
``check-emails`` step from the workflow YAML (so it can't drift from what
CI executes) and runs it against a real git repo shaped like the bug
report: a base branch with unmapped-author history, and a PR branch on
top of it that only adds a commit from an already-mapped email.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "contributor-check.yml"


def _load_check_emails_script() -> str:
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text())
    steps = workflow["jobs"]["check-attribution"]["steps"]
    for step in steps:
        if step.get("id") == "check-emails":
            return step["run"]
    raise AssertionError("check-emails step not found in contributor-check.yml")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
        env={"GIT_AUTHOR_NAME": "Test", "GIT_COMMITTER_NAME": "Test", "HOME": str(repo)},
    )


def _commit(repo: Path, path: str, email: str) -> None:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(path)
    _git(repo, "add", path)
    env = {
        "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": email,
        "HOME": str(repo),
    }
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", f"add {path}"],
        capture_output=True, text=True, check=True, env=env,
    )


def _build_repo(tmp_path: Path) -> Path:
    """Build: main (seed) -> base (2 unmapped historical commits) -> pr (1 mapped commit)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "seed@example.com")
    _git(repo, "config", "user.name", "Test")

    (repo / "contributors" / "emails").mkdir(parents=True)
    (repo / "contributors" / "emails" / "seed@example.com").write_text("")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "release.py").write_text("AUTHOR_MAP = {}\n")
    _git(repo, "add", ".")
    _commit(repo, "seed.txt", "seed@example.com")

    _git(repo, "branch", "dashboard-active-profile-bootstrap")
    _git(repo, "checkout", "-q", "dashboard-active-profile-bootstrap")
    _commit(repo, "historical1.txt", "old-contributor-1@example.com")
    _commit(repo, "historical2.txt", "old-contributor-2@example.com")

    _git(repo, "checkout", "-q", "-b", "pr-branch")
    _commit(repo, "feature.txt", "seed@example.com")

    # Fixed refs/remotes/origin/* namespace without a real network remote.
    _git(repo, "update-ref", "refs/remotes/origin/main", "refs/heads/main")
    _git(repo, "update-ref",
         "refs/remotes/origin/dashboard-active-profile-bootstrap",
         "refs/heads/dashboard-active-profile-bootstrap")
    return repo


def _run_script(repo: Path, tmp_path: Path, *, base_ref: str | None) -> subprocess.CompletedProcess:
    script = _load_check_emails_script()
    script_path = tmp_path / "check-emails.sh"
    script_path.write_text(script)
    github_output = tmp_path / "github_output"
    github_output.write_text("")
    env = {"HOME": str(repo), "PATH": "/usr/bin:/bin", "GITHUB_OUTPUT": str(github_output)}
    if base_ref is not None:
        env["GITHUB_BASE_REF"] = base_ref
    return subprocess.run(
        ["bash", str(script_path)],
        cwd=repo, capture_output=True, text=True, env=env,
    )


def test_pr_targeting_non_main_base_only_scans_pr_commits(tmp_path):
    """A PR targeting a non-main base must not see that base's own history."""
    repo = _build_repo(tmp_path)
    result = _run_script(repo, tmp_path, base_ref="dashboard-active-profile-bootstrap")

    assert "old-contributor-1@example.com" not in result.stdout
    assert "old-contributor-2@example.com" not in result.stdout
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All contributor emails are mapped" in result.stdout


def test_non_pr_context_falls_back_to_main(tmp_path):
    """Without GITHUB_BASE_REF (push/schedule), the full scan against main is preserved."""
    repo = _build_repo(tmp_path)
    result = _run_script(repo, tmp_path, base_ref=None)

    assert result.returncode == 1
    assert "old-contributor-1@example.com" in result.stdout
    assert "old-contributor-2@example.com" in result.stdout
