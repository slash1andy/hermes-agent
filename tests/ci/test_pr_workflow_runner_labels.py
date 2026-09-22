"""Regression test: PR-triggered CI jobs must request runner labels an
ordinary GitHub fork actually has.

Background: several workflow jobs hardcoded larger-runner labels
(``ubuntu-latest-32-core``, ``ubuntu-latest-96-core``,
``windows-latest-32-core``) that only exist on the upstream org's paid
runner group. On a personal fork those jobs queue forever with no
``runner_name`` — GitHub never has a matching runner to assign.

The corpus this test scans is not a hand-picked file list: a hardcoded list
drifts the moment a workflow gains (or loses) a ``pull_request`` trigger or a
reusable-workflow call, and the drift is invisible until a fork PR hangs. It
is instead discovered from the repo's actual workflow graph: every workflow
triggered directly by ``pull_request``/``pull_request_target``, plus every
reusable workflow (``uses: ./.github/workflows/...``) their fork-reachable
jobs call, transitively. Jobs that are already gated off forks entirely
(``if: github.repository == 'NousResearch/hermes-agent'``, as in docker.yml)
or that can never run at all (``if: false``) are exempt: they never queue on
a fork in the first place, so they and any reusable workflow they alone call
are dropped from the corpus.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"

_STANDARD_LABELS = {"ubuntu-latest", "windows-latest", "macos-latest"}
_CANONICAL_REPO_GATE = "github.repository == 'NousResearch/hermes-agent'"
_PR_TRIGGERS = {"pull_request", "pull_request_target"}
_REUSABLE_PREFIX = "./.github/workflows/"


def _load_workflows(workflows_dir: Path) -> dict[str, dict]:
    """filename -> parsed workflow, for every ``*.yml``/``*.yaml`` in the dir."""
    files = sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml"))
    return {f.name: yaml.safe_load(f.read_text()) for f in files}


def _trigger_names(workflow: dict) -> set[str]:
    """Names of a workflow's top-level triggers.

    PyYAML's default (YAML 1.1) resolver reads the bare ``on:`` key as the
    boolean ``True``, not the string ``"on"`` — so the trigger block lives at
    ``workflow[True]`` for every workflow in this repo, not ``workflow["on"]``.
    """
    raw = workflow.get("on", workflow.get(True))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {v for v in raw if isinstance(v, str)}
    if isinstance(raw, dict):
        return set(raw)
    return set()


def _job_unreachable_on_fork(job: dict) -> bool:
    """True if this job provably never runs on an ordinary fork."""
    if_expr = job.get("if")
    if if_expr is False:
        return True
    if isinstance(if_expr, str):
        if if_expr.strip().lower() == "false":
            return True
        if _CANONICAL_REPO_GATE in if_expr:
            return True
    return False


def _pr_reachable_workflows(workflows_dir: Path) -> dict[str, dict]:
    """Every workflow whose jobs an ordinary fork PR can actually queue.

    Seeded from the workflows triggered directly by ``pull_request`` /
    ``pull_request_target``, then walked outward through ``uses:
    ./.github/workflows/...`` reusable-workflow calls on jobs that are
    themselves fork-reachable.
    """
    workflows = _load_workflows(workflows_dir)
    queue = [name for name, wf in workflows.items() if _trigger_names(wf) & _PR_TRIGGERS]
    seen: dict[str, dict] = {}
    while queue:
        name = queue.pop()
        if name in seen or name not in workflows:
            continue
        workflow = workflows[name]
        seen[name] = workflow
        for job in workflow.get("jobs", {}).values():
            if not isinstance(job, dict) or _job_unreachable_on_fork(job):
                continue
            uses = job.get("uses", "")
            if isinstance(uses, str) and uses.startswith(_REUSABLE_PREFIX):
                queue.append(uses[len(_REUSABLE_PREFIX):])
    return seen


def _job_runner_labels(job: dict) -> list[str]:
    """Every literal runner label a job's ``runs-on`` could resolve to.

    ``runs-on`` may also be a YAML list (GitHub matches a runner carrying
    *all* the listed labels), used for self-hosted/custom-labeled runners —
    those labels don't exist on an ordinary fork either.
    """
    raw = job.get("runs-on")
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, str)]
    if isinstance(raw, str) and raw in _STANDARD_LABELS:
        return [raw]
    if isinstance(raw, str) and "matrix." in raw:
        matrix = job.get("strategy", {}).get("matrix", {})
        labels = [entry["runner"] for entry in matrix.get("include", []) if "runner" in entry]
        labels += [v for v in matrix.get("runner", []) if isinstance(v, str)]
        return labels
    if isinstance(raw, str):
        return [raw]
    return []


def _unavailable_fork_labels(workflows_dir: Path = _WORKFLOWS_DIR) -> list[tuple[str, str, str]]:
    """(file, job, label) for every fork-reachable job requesting a
    non-standard runner label."""
    findings = []
    for filename, workflow in sorted(_pr_reachable_workflows(workflows_dir).items()):
        for job_name, job in workflow.get("jobs", {}).items():
            if not isinstance(job, dict) or _job_unreachable_on_fork(job):
                continue
            for label in _job_runner_labels(job):
                if label not in _STANDARD_LABELS:
                    findings.append((filename, job_name, label))
    return findings


def test_pr_reachable_jobs_use_fork_available_runner_labels():
    findings = _unavailable_fork_labels()
    assert not findings, (
        "job(s) request a runner label unavailable to ordinary forks "
        f"(queues forever there): {findings}"
    )


# ---------------------------------------------------------------------------
# Self-tests for the discovery logic above, against synthetic workflow
# corpora — so a change to the graph walk or the ``on:``/``if:`` handling
# fails here instead of silently under- or over-scoping the real corpus.
# ---------------------------------------------------------------------------


def _write_workflow(workflows_dir: Path, name: str, body: str) -> None:
    workflows_dir.mkdir(parents=True, exist_ok=True)
    (workflows_dir / name).write_text(body)


def test_on_key_is_read_despite_pyyaml_boolean_parsing(tmp_path):
    """``on:`` parses as the boolean key ``True`` under PyYAML; a workflow
    using it as a PR trigger must still be discovered."""
    _write_workflow(
        tmp_path, "caller.yml",
        "on:\n  pull_request:\njobs:\n  build:\n    runs-on: ubuntu-latest-32-core\n",
    )
    findings = _unavailable_fork_labels(tmp_path)
    assert findings == [("caller.yml", "build", "ubuntu-latest-32-core")]


def test_reusable_workflow_called_from_pr_workflow_is_scanned(tmp_path):
    _write_workflow(
        tmp_path, "caller.yml",
        "on:\n  pull_request:\njobs:\n  sub:\n    uses: ./.github/workflows/callee.yml\n",
    )
    _write_workflow(
        tmp_path, "callee.yml",
        "on:\n  workflow_call:\njobs:\n  build:\n    runs-on: windows-latest-32-core\n",
    )
    findings = _unavailable_fork_labels(tmp_path)
    assert findings == [("callee.yml", "build", "windows-latest-32-core")]


def test_reusable_workflow_not_called_from_any_pr_workflow_is_excluded(tmp_path):
    """A ``workflow_call``-only workflow with no fork-reachable caller (e.g.
    only invoked from a ``workflow_dispatch``/``schedule`` entrypoint) is not
    part of the fork-reachable corpus."""
    _write_workflow(
        tmp_path, "dispatch-only.yml",
        "on:\n  workflow_dispatch:\njobs:\n  sub:\n    uses: ./.github/workflows/callee.yml\n",
    )
    _write_workflow(
        tmp_path, "callee.yml",
        "on:\n  workflow_call:\njobs:\n  build:\n    runs-on: windows-latest-32-core\n",
    )
    assert _unavailable_fork_labels(tmp_path) == []


def test_canonical_repo_gated_job_is_excluded(tmp_path):
    _write_workflow(
        tmp_path, "docker.yml",
        "on:\n  pull_request:\njobs:\n"
        "  build:\n"
        "    if: github.repository == 'NousResearch/hermes-agent'\n"
        "    runs-on: ubuntu-latest-32-core\n",
    )
    assert _unavailable_fork_labels(tmp_path) == []


def test_canonical_repo_gated_job_does_not_pull_in_its_callee(tmp_path):
    _write_workflow(
        tmp_path, "caller.yml",
        "on:\n  pull_request:\njobs:\n"
        "  sub:\n"
        "    if: github.repository == 'NousResearch/hermes-agent'\n"
        "    uses: ./.github/workflows/callee.yml\n",
    )
    _write_workflow(
        tmp_path, "callee.yml",
        "on:\n  workflow_call:\njobs:\n  build:\n    runs-on: windows-latest-32-core\n",
    )
    assert _unavailable_fork_labels(tmp_path) == []


def test_literal_if_false_job_is_excluded(tmp_path):
    _write_workflow(
        tmp_path, "caller.yml",
        "on:\n  pull_request:\njobs:\n"
        "  sub:\n"
        "    if: false\n"
        "    uses: ./.github/workflows/callee.yml\n",
    )
    _write_workflow(
        tmp_path, "callee.yml",
        "on:\n  workflow_call:\njobs:\n  build:\n    runs-on: windows-latest-32-core\n",
    )
    assert _unavailable_fork_labels(tmp_path) == []


def test_matrix_runner_labels_are_resolved(tmp_path):
    _write_workflow(
        tmp_path, "caller.yml",
        "on:\n  pull_request:\njobs:\n"
        "  build:\n"
        "    runs-on: ${{ matrix.runner }}\n"
        "    strategy:\n"
        "      matrix:\n"
        "        include:\n"
        "          - arch: amd64\n"
        "            runner: ubuntu-latest\n"
        "          - arch: arm64\n"
        "            runner: ubuntu-latest-32-arm-core\n",
    )
    findings = _unavailable_fork_labels(tmp_path)
    assert findings == [("caller.yml", "build", "ubuntu-latest-32-arm-core")]


def test_list_runs_on_labels_are_inspected(tmp_path):
    """``runs-on: [self-hosted, linux, x64]`` requests a runner carrying all
    three labels; none exist on an ordinary fork, so each must be flagged."""
    _write_workflow(
        tmp_path, "caller.yml",
        "on:\n  pull_request:\njobs:\n"
        "  build:\n"
        "    runs-on: [self-hosted, linux, x64]\n",
    )
    findings = _unavailable_fork_labels(tmp_path)
    assert findings == [
        ("caller.yml", "build", "self-hosted"),
        ("caller.yml", "build", "linux"),
        ("caller.yml", "build", "x64"),
    ]


def test_standard_labels_pass_clean(tmp_path):
    _write_workflow(
        tmp_path, "caller.yml",
        "on:\n  pull_request:\njobs:\n"
        "  linux:\n    runs-on: ubuntu-latest\n"
        "  mac:\n    runs-on: macos-latest\n"
        "  win:\n    runs-on: windows-latest\n",
    )
    assert _unavailable_fork_labels(tmp_path) == []
