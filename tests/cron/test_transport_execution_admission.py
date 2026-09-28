"""Transport-process cron execution admission — independent of delivery-queue draining.

A shared transport process (one Photon listener multiplexing several profiles) must be able to
refuse LOCAL cron execution (script subprocess, AIAgent construction, tick dispatch) for jobs it
merely stores or drains delivery for, while still delivering already-queued results. The gate is
the root/launch PROCESS config (``cron.execution_enabled``), read independent of any profile scope
bound over a tick or a run — a named profile's own ``config.yaml`` must never be able to
re-enable it from inside that profile's scope, and a host that mirrors the served profile into
``HERMES_HOME`` per turn (``get_routing_process_hermes_home`` — the pinned identity, not the live
env var) must not be able to smuggle the profile's own policy in either.

Every denial asserted here requires the returned error to name the policy
(``_DENIAL_MARKER = "cron.execution_enabled"``) — a bare "success is False" would also pass on an
unrelated failure (missing provider credentials, absent script) with no gate at all, which is
exactly the false-positive this suite must not reproduce.

These tests drive the real entry points: ``cron.scheduler.run_job`` (shared by the tick dispatcher
and the ``fire_due`` webhook path), ``cron.scheduler_tick._tick_admitted`` (the real tick body —
NOT mocked; only the deepest dispatch call, ``scheduler.run_one_job``, is spied on so a real
mid-tick side effect — ``advance_next_runs``/``claim_job_for_fire`` mutating ``next_run_at`` and
stamping a claim BEFORE ``run_job`` ever runs — can't silently destroy a due occurrence once
admission moves earlier than run_job), real ``cron.jobs.create_job``/``update_job``, and the real
``cron.scheduler_provider._profile_cron_scope`` scope binder plus ``hermes_constants.
pin_process_hermes_home`` — no fabricated helper API. Subprocess launches are captured (not faked
away) via a spy wrapping the real ``subprocess.Popen``; AIAgent construction/inference is captured
via a spy replacing ``run_agent.AIAgent`` and stubbing ``resolve_runtime_provider`` (the same
pattern ``tests/cron/test_scheduler.py::TestRunJobWakeGate`` uses, needed because a hermetic CI env
has no provider credentials — without the stub, "AIAgent never called" would be vacuously true for
the wrong reason).
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, Mock
import subprocess

import pytest
import yaml

import hermes_constants
from cron import jobs as cron_jobs
from cron import scheduler
from cron import scheduler_provider
from cron.scheduler_provider import InProcessCronScheduler
from cron import scheduler_script
from cron import scheduler_tick
from cron import delivery_queue
from hermes_time import now as hermes_now
import run_agent


_DENIAL_MARKER = "cron.execution_enabled"

SENTINEL_SCRIPT = "print('SYNTHETIC_OK')\n"

_FAKE_RUNTIME = {
    "provider": "openrouter",
    "api_mode": "chat_completions",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "test-key",
    "source": "stub",
    "requested_provider": None,
}


def _write_root_config(home, cron_section=None):
    home.mkdir(parents=True, exist_ok=True)
    if cron_section is not None:
        (home / "config.yaml").write_text(
            yaml.safe_dump({"cron": cron_section}), encoding="utf-8")


def _write_sentinel_script(home) -> str:
    scripts_dir = home / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "sentinel.py").write_text(SENTINEL_SCRIPT, encoding="utf-8")
    return "sentinel.py"


def _job_dict(name, *, script=None, no_agent=False, prompt="Say hello"):
    """Minimal valid ``run_job`` dict — mirrors ``TestRunJobWakeGate._make_job``; run_job never
    ticks this record, so the schedule field's shape does not need to match ``parse_schedule``."""
    job = {"id": f"job_{name}", "name": name, "schedule": "*/5 * * * *"}
    if no_agent:
        job["no_agent"] = True
        job["script"] = script
    else:
        job["prompt"] = prompt
        if script:
            job["script"] = script
    return job


def _popen_spy(monkeypatch):
    spy = Mock(wraps=subprocess.Popen)
    monkeypatch.setattr(scheduler_script.subprocess, "Popen", spy)
    return spy


@pytest.fixture
def root_home(tmp_path, monkeypatch):
    home = tmp_path / "root"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def stub_runtime_provider(monkeypatch):
    """Same stub as ``test_scheduler.py::TestRunJobWakeGate`` — resolves BEFORE AIAgent is
    constructed, so its absence would block a hermetic run for the wrong reason."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: dict(_FAKE_RUNTIME))


def _fake_agent():
    agent = MagicMock()
    agent.run_conversation = MagicMock(return_value={"final_response": "ok", "messages": []})
    return agent


# ── (1) default-enabled: no_agent job reaches the real script execution seam ────────────────────


def test_default_enabled_no_agent_job_reaches_subprocess_seam(root_home, monkeypatch):
    # No config.yaml at all: missing config must retain current (enabled) behavior.
    script_name = _write_sentinel_script(root_home)
    job = _job_dict("watchdog", script=script_name, no_agent=True)
    popen_spy = _popen_spy(monkeypatch)

    success, _doc, final_response, error = scheduler.run_job(job)

    assert error is None
    assert success is True
    assert "SYNTHETIC_OK" in final_response
    popen_spy.assert_called_once()


# ── (1b) default-enabled: regular agent job reaches real AIAgent construction, no network ───────


def test_default_enabled_agent_job_reaches_aiagent_construction(
        root_home, monkeypatch, stub_runtime_provider):
    _write_root_config(root_home, {"execution_enabled": True})
    job = _job_dict("brief")
    agent = _fake_agent()
    agent_cls = Mock(return_value=agent)
    monkeypatch.setattr(run_agent, "AIAgent", agent_cls)

    success, _doc, final_response, error = scheduler.run_job(job)

    agent_cls.assert_called_once()
    assert success is True
    assert error is None
    assert final_response == "ok"


# ── (2) root disabled: no_agent script path makes zero subprocess calls ─────────────────────────


def test_root_disabled_no_agent_job_never_spawns_subprocess(root_home, monkeypatch):
    _write_root_config(root_home, {"execution_enabled": False})
    script_name = _write_sentinel_script(root_home)
    job = _job_dict("watchdog", script=script_name, no_agent=True)
    popen_spy = _popen_spy(monkeypatch)

    success, doc, final_response, error = scheduler.run_job(job)

    popen_spy.assert_not_called()
    assert success is False
    assert final_response == ""
    assert error and _DENIAL_MARKER in error
    assert doc


# ── (3) root disabled: regular agent path makes zero AIAgent construction calls ─────────────────


def test_root_disabled_agent_job_never_constructs_aiagent(
        root_home, monkeypatch, stub_runtime_provider):
    # Runtime-provider stub proves credentials are NOT why AIAgent is unreached: without the
    # execution-admission gate this job would resolve fine and reach construction.
    _write_root_config(root_home, {"execution_enabled": False})
    job = _job_dict("brief")
    agent_cls = Mock(return_value=_fake_agent())
    monkeypatch.setattr(run_agent, "AIAgent", agent_cls)

    success, _doc, final_response, error = scheduler.run_job(job)

    agent_cls.assert_not_called()
    assert success is False
    assert final_response == ""
    assert error and _DENIAL_MARKER in error


# ── (4) root disabled dominates a named profile's own config, no leaked scope (A->B->A) ─────────


def test_root_disabled_dominates_named_profile_declaring_true(tmp_path, monkeypatch):
    root = tmp_path / "root"
    profile_home = tmp_path / "profiles" / "work"
    _write_root_config(root, {"execution_enabled": False})
    _write_root_config(profile_home, {"execution_enabled": True})
    root_script = _write_sentinel_script(root)
    profile_script = _write_sentinel_script(profile_home)
    monkeypatch.setenv("HERMES_HOME", str(root))
    hermes_constants.pin_process_hermes_home(str(root))
    popen_spy = _popen_spy(monkeypatch)

    try:
        root_job = _job_dict("root-watchdog", script=root_script, no_agent=True)
        profile_job = _job_dict("profile-watchdog", script=profile_script, no_agent=True)

        success_a1, _, _, error_a1 = scheduler.run_job(root_job)
        assert success_a1 is False
        assert error_a1 and _DENIAL_MARKER in error_a1

        with scheduler_provider._profile_cron_scope(str(profile_home)):
            # A host that mirrors the served profile into HERMES_HOME per turn (gateway/AGENTS.md,
            # Hermes WebUI) must still resolve the PINNED root, not this live env value.
            with monkeypatch.context() as scoped:
                scoped.setenv("HERMES_HOME", str(profile_home))
                success_b, _, _, error_b = scheduler.run_job(profile_job)
        assert success_b is False
        assert error_b and _DENIAL_MARKER in error_b

        success_a2, _, _, error_a2 = scheduler.run_job(root_job)
        assert success_a2 is False
        assert error_a2 and _DENIAL_MARKER in error_a2
    finally:
        hermes_constants.pin_process_hermes_home(None)

    popen_spy.assert_not_called()


# ── (5) malformed explicit policy values fail closed, never silently enable ─────────────────────


@pytest.mark.parametrize(
    "malformed_value",
    ["true", 1, None],
    ids=["string-true", "int-one", "explicit-null"],
)
def test_malformed_execution_enabled_fails_closed(root_home, monkeypatch, malformed_value):
    _write_root_config(root_home, {"execution_enabled": malformed_value})
    script_name = _write_sentinel_script(root_home)
    job = _job_dict("watchdog", script=script_name, no_agent=True)
    popen_spy = _popen_spy(monkeypatch)

    success, _doc, final_response, error = scheduler.run_job(job)

    popen_spy.assert_not_called()
    assert success is False
    assert final_response == ""
    assert error and _DENIAL_MARKER in error


# ── (6) native enqueue/drain remains available while execution is denied ────────────────────────


def test_delivery_queue_drains_while_execution_disabled(root_home, tmp_path, monkeypatch):
    # Queue primitive only (enqueue/drain + a mocked final send) — the full route-consumer contract
    # (producer cannot consume its own queued work) is the second slice's job, not this one's.
    _write_root_config(root_home, {"execution_enabled": False})
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "deliveries.db")

    delivery_queue.enqueue("exec-transport-1", {"id": "job-transport-1"}, "queued brief")
    send = Mock(return_value=None)

    assert delivery_queue.drain(send) == 1
    send.assert_called_once_with({"id": "job-transport-1"}, "queued brief", False)


# ── (7) native tick denial: due job's next_run_at/claim state unchanged, no worker dispatched ────


def _create_due_job(home, *, script_name):
    with cron_jobs.use_cron_store(home):
        job = cron_jobs.create_job(
            prompt=None, schedule="1h", name="tick-watchdog", no_agent=True, script=script_name)
        past = (hermes_now() - timedelta(seconds=5)).isoformat()
        cron_jobs.update_job(job["id"], {"next_run_at": past})
        return job["id"], past


def _load_job_record(home, job_id):
    with cron_jobs.use_cron_store(home):
        return next(j for j in cron_jobs.load_jobs() if j["id"] == job_id)


def test_root_disabled_tick_leaves_due_job_unclaimed_and_dispatches_nothing(
        root_home, monkeypatch):
    _write_root_config(root_home, {"execution_enabled": False})
    script_name = _write_sentinel_script(root_home)
    job_id, due_at = _create_due_job(root_home, script_name=script_name)
    run_one_job_spy = Mock(return_value=True)
    monkeypatch.setattr(scheduler, "run_one_job", run_one_job_spy)

    scheduler_tick._tick_admitted(verbose=False)

    run_one_job_spy.assert_not_called()
    record = _load_job_record(root_home, job_id)
    assert record["next_run_at"] == due_at
    assert "fire_claim" not in record


def test_root_disabled_tick_dominates_named_profile_declaring_true(tmp_path, monkeypatch):
    root = tmp_path / "root"
    profile_home = tmp_path / "profiles" / "work"
    _write_root_config(root, {"execution_enabled": False})
    _write_root_config(profile_home, {"execution_enabled": True})
    monkeypatch.setenv("HERMES_HOME", str(root))
    script_name = _write_sentinel_script(profile_home)
    run_one_job_spy = Mock(return_value=True)
    monkeypatch.setattr(scheduler, "run_one_job", run_one_job_spy)

    with scheduler_provider._profile_cron_scope(str(profile_home)):
        job_id, due_at = _create_due_job(profile_home, script_name=script_name)
        scheduler_tick._tick_admitted(verbose=False)
        record = _load_job_record(profile_home, job_id)

    run_one_job_spy.assert_not_called()
    assert record["next_run_at"] == due_at
    assert "fire_claim" not in record


def test_execution_disabled_after_tick_advance_restores_original_due_occurrence(
        root_home, monkeypatch):
    """A real tick may advance and claim before admission flips; denial must restore that slot."""
    from cron.occurrences import completed_occurrence, scheduled_instant

    _write_root_config(root_home, {"execution_enabled": True})
    script_name = _write_sentinel_script(root_home)
    job_id, due_at = _create_due_job(root_home, script_name=script_name)
    completed = subprocess.CompletedProcess([], 0, stdout="SYNTHETIC_OK\\n", stderr="")
    setattr(completed, "poll", Mock(return_value=0))
    setattr(completed, "communicate", Mock(return_value=(completed.stdout, completed.stderr)))
    popen_spy = Mock(return_value=completed)
    monkeypatch.setattr(scheduler_script.subprocess, "Popen", popen_spy)
    agent_cls = Mock()
    monkeypatch.setattr(run_agent, "AIAgent", agent_cls)

    real_submit = scheduler._submit_with_guard
    flipped = False

    def flip_after_advance(job, pool, process_job):
        nonlocal flipped
        if not flipped:
            flipped = True
            _write_root_config(root_home, {"execution_enabled": False})
        return real_submit(job, pool, process_job)

    monkeypatch.setattr(scheduler, "_submit_with_guard", flip_after_advance)
    scheduler_tick._tick_admitted(verbose=False)

    denied = _load_job_record(root_home, job_id)
    assert denied["next_run_at"] == due_at
    assert denied.get("fire_claim") is None
    assert completed_occurrence(denied, scheduled_instant(due_at)) is False
    popen_spy.assert_not_called()
    agent_cls.assert_not_called()

    _write_root_config(root_home, {"execution_enabled": True})
    scheduler_tick._tick_admitted(verbose=False)

    popen_spy.assert_called_once()
    final = _load_job_record(root_home, job_id)
    assert final.get("fire_claim") is None


# ── (8) native manual/webhook fire denial: claim_fire never claims, no execution/subprocess ──────


def _create_job(home, *, script_name):
    with cron_jobs.use_cron_store(home):
        return cron_jobs.create_job(
            prompt=None, schedule="1h", name="fire-watchdog", no_agent=True, script=script_name)


def test_root_disabled_manual_fire_denied_leaves_job_unclaimed(root_home, monkeypatch):
    _write_root_config(root_home, {"execution_enabled": False})
    script_name = _write_sentinel_script(root_home)
    job = _create_job(root_home, script_name=script_name)
    popen_spy = _popen_spy(monkeypatch)

    fired = InProcessCronScheduler().fire_due(job["id"], manual=True)

    assert fired is False
    popen_spy.assert_not_called()
    record = _load_job_record(root_home, job["id"])
    assert record["next_run_at"] == job["next_run_at"]
    assert "fire_claim" not in record


def test_default_enabled_manual_fire_reaches_subprocess_seam(root_home, monkeypatch):
    # Positive pair for (8): the same manual-fire path, enabled, actually dispatches.
    script_name = _write_sentinel_script(root_home)
    job = _create_job(root_home, script_name=script_name)
    popen_spy = _popen_spy(monkeypatch)

    fired = InProcessCronScheduler().fire_due(job["id"], manual=True)

    assert fired is True
    popen_spy.assert_called_once()


# ── (9) malformed cron SECTION (not a mapping) and broken YAML both fail closed ──────────────────


@pytest.mark.parametrize(
    "non_mapping_cron_section",
    ["oops-not-a-mapping", None],
    ids=["string", "explicit-null"],
)
def test_malformed_cron_section_not_a_mapping_fails_closed(
        root_home, monkeypatch, non_mapping_cron_section):
    # An explicit `cron: null` is a PRESENT, non-mapping section — distinct from the key being
    # absent entirely (which preserves the enabled default, see test_default_enabled_*). A helper
    # that reads `effective.get("cron")` cannot tell the two apart (both come back as `None`) and
    # must not conflate "explicitly null" with "absent".
    root_home.mkdir(parents=True, exist_ok=True)
    (root_home / "config.yaml").write_text(
        yaml.safe_dump({"cron": non_mapping_cron_section}), encoding="utf-8")
    script_name = _write_sentinel_script(root_home)
    job = _job_dict("watchdog", script=script_name, no_agent=True)
    popen_spy = _popen_spy(monkeypatch)

    success, _doc, final_response, error = scheduler.run_job(job)

    popen_spy.assert_not_called()
    assert success is False
    assert final_response == ""
    assert error and _DENIAL_MARKER in error


def test_broken_yaml_root_config_fails_closed(root_home, monkeypatch):
    root_home.mkdir(parents=True, exist_ok=True)
    (root_home / "config.yaml").write_text("cron: [unterminated\n", encoding="utf-8")
    script_name = _write_sentinel_script(root_home)
    job = _job_dict("watchdog", script=script_name, no_agent=True)
    popen_spy = _popen_spy(monkeypatch)

    success, _doc, final_response, error = scheduler.run_job(job)

    popen_spy.assert_not_called()
    assert success is False
    assert final_response == ""
    # Broken YAML must fail closed WITHOUT leaking the parser's raw exception text/file content.
    assert error and _DENIAL_MARKER in error
    assert "unterminated" not in error


# ── (10) claimed-then-denied: policy flips to false BETWEEN claim_fire and fire_claimed ──────────


@pytest.mark.parametrize("manual", [True, False], ids=["manual-run-now", "scheduled-due-slot"])
def test_claimed_fire_denied_releases_claim_and_preserves_due_slot(root_home, monkeypatch, manual):
    """claim_fire() while allowed (real store CAS) -> config flips false -> fire_claimed() denied:
    zero subprocess, the claim is released immediately (no TTL wait), the original due instant is
    not consumed (occurrences.py: a failed/denied attempt is not "completed", so the slot stays
    eligible), and re-enabling lets the SAME occurrence run exactly once right away — no TTL aging."""
    from cron.occurrences import completed_occurrence

    _write_root_config(root_home, {"execution_enabled": True})
    script_name = _write_sentinel_script(root_home)
    job = _create_job(root_home, script_name=script_name)
    due_at = (hermes_now() - timedelta(seconds=5)).isoformat()
    with cron_jobs.use_cron_store(root_home):
        cron_jobs.update_job(job["id"], {"next_run_at": due_at})
    popen_spy = _popen_spy(monkeypatch)
    provider = InProcessCronScheduler()

    claimed = provider.claim_fire(job["id"], manual=manual)
    assert claimed is not None
    instant = claimed.get("_scheduled_instant")

    _write_root_config(root_home, {"execution_enabled": False})
    provider.fire_claimed(claimed)

    popen_spy.assert_not_called()
    record = _load_job_record(root_home, job["id"])
    assert record.get("fire_claim") is None  # released immediately, no TTL wait needed
    assert record["next_run_at"] == due_at  # original due slot not advanced beyond
    if instant is not None:
        assert completed_occurrence(record, instant) is False  # denied run never "completed" it

    _write_root_config(root_home, {"execution_enabled": True})
    reclaimed = provider.claim_fire(job["id"], manual=manual)  # immediately, no TTL aging
    assert reclaimed is not None
    provider.fire_claimed(reclaimed)

    popen_spy.assert_called_once()
    final_record = _load_job_record(root_home, job["id"])
    if instant is not None:
        # A later replay of the SAME already-run occurrence must not be able to re-fire it.
        assert completed_occurrence(final_record, instant) is True


def test_denied_fire_never_clobbers_a_successors_claim(root_home, monkeypatch):
    """A late/stale denial handler for an OLD claim must not touch a DIFFERENT owner's later,
    live claim on the same job (native storage only — no fabricated release helper)."""
    _write_root_config(root_home, {"execution_enabled": True})
    script_name = _write_sentinel_script(root_home)
    job = _create_job(root_home, script_name=script_name)
    popen_spy = _popen_spy(monkeypatch)
    provider = InProcessCronScheduler()

    stale_claim = provider.claim_fire(job["id"])
    assert stale_claim is not None

    successor_claim = {"at": hermes_now().isoformat(), "by": "successor:fresh-token"}
    with cron_jobs.use_cron_store(root_home):
        cron_jobs.update_job(job["id"], {"fire_claim": successor_claim})

    _write_root_config(root_home, {"execution_enabled": False})
    provider.fire_claimed(stale_claim)  # late handling for the now-superseded claim

    popen_spy.assert_not_called()
    record = _load_job_record(root_home, job["id"])
    assert record.get("fire_claim") == successor_claim


# ── (11) admission resolves root/process secret authority, not a served profile's scope ──────────


def test_admission_resolves_root_secret_not_served_profile_scope(root_home, tmp_path, monkeypatch):
    """Drives the REAL admission boundary (``cron_execution_denied_reason``) with a thin wrapper
    around the real loader that only CAPTURES the effective config it returns (never synthesizes a
    result or implements policy), proving admission reads root/process secret authority for a
    root-owned scalar field even while a named profile's secret scope is installed — without
    touching the generic loader's own (intentionally profile-scope-aware) contract."""
    from cron import scheduler_admission
    from agent.secret_scope import set_secret_scope, reset_secret_scope

    monkeypatch.setenv("_TRANSPORT_ADMISSION_SENTINEL", "root-value")
    _write_root_config(root_home, {
        "execution_enabled": True, "model_provider": "${_TRANSPORT_ADMISSION_SENTINEL}"})
    captured = {}
    real_loader = scheduler_admission.load_user_config_effective

    def _capturing_loader(*args, **kwargs):
        effective = real_loader(*args, **kwargs)
        captured["model_provider"] = (effective.get("cron") or {}).get("model_provider")
        return effective

    monkeypatch.setattr(scheduler_admission, "load_user_config_effective", _capturing_loader)
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    token = set_secret_scope(
        {"_TRANSPORT_ADMISSION_SENTINEL": "profile-value"}, profile_home=str(profile_home))
    try:
        assert scheduler_admission.cron_execution_denied_reason() is None
    finally:
        reset_secret_scope(token)

    assert captured["model_provider"] == "root-value"
    # Restored: an unscoped read afterward sees the process env again, unaffected.
    assert real_loader(root_home / "config.yaml", fail_closed=True
                        )["cron"]["model_provider"] == "root-value"
