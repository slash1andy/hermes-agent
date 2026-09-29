"""Cron jobs persist the durable external-delivery outcome, not just model success."""
import time

import pytest

from cron import delivery_queue, executions, jobs, scheduler
from gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_cli import cron as cron_cli
from tools.cronjob_job_args import _format_job
from tools.cronjob_tools import _manual_run_completion, _manual_run_delivery_note


def _assert_unknown_diagnostics(saved, *, model_failed):
    diagnostics = "\n".join(
        cron_cli._job_warnings(saved) + cron_cli._cron_doctor_issues_for_job(saved)
    ).casefold()
    assert any(word in diagnostics for word in ("unknown", "uncertain", "unconfirmed"))
    assert "not delivered" not in diagnostics
    display = cron_cli._last_run_display(saved).casefold()
    assert any(word in display for word in ("unknown", "uncertain", "unconfirmed"))
    if not model_failed:
        assert "failed" not in display


@pytest.mark.parametrize(
    ("queue_state", "expected_job_status", "expected_delivery_outcome"),
    [
        ("pending", "delivery_queued", "queued"),
        ("unknown", "delivery_unknown", "unknown"),
        ("delivered", "ok", "delivered"),
        ("failed", "delivery_failed", "failed"),
        ("failed_after_wait", "delivery_failed", "failed"),
    ],
)
def test_real_external_delivery_outcome_updates_job_and_ledger(
    tmp_path, monkeypatch, queue_state, expected_job_status, expected_delivery_outcome,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)

    def forbidden_direct_send(*args, **kwargs):
        pytest.fail("external worker sent directly instead of using the native queue")

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", forbidden_direct_send)

    def run(job, **kwargs):
        return True, "retained output", "final response", None

    monkeypatch.setattr(scheduler, "run_job", run)
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:fixture")
    execution = executions.create_execution(job["id"], source="fixture")
    job["execution_id"] = execution["id"]
    execution_id = execution["id"]
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution_id)

    original_wait = delivery_queue.enqueue_and_wait

    def wait_with_zero_deadline(execution_id, job, content, *, for_failure=False):
        if queue_state == "failed_after_wait":
            result = original_wait(
                execution_id, job, content, for_failure=for_failure, timeout=0
            )
            assert result is None
            pending = delivery_queue.get_status(execution_id)
            assert pending is not None
            assert pending["status"] == "pending"
            claimed = delivery_queue.claim_next()
            assert claimed is not None
            assert claimed["execution_id"] == execution_id
            assert delivery_queue._finish(execution_id, error="captured final send failed")
            return result
        queued = delivery_queue.enqueue(execution_id, job, content, for_failure=for_failure)
        if queue_state == "unknown":
            assert delivery_queue.claim_next()["execution_id"] == execution_id
        elif queue_state in {"delivered", "failed"} and queued["status"] == "pending":
            assert delivery_queue.claim_next()["execution_id"] == execution_id
            error = "captured final send failed" if queue_state == "failed" else None
            assert delivery_queue._finish(execution_id, error=error)
        return original_wait(
            execution_id, job, content, for_failure=for_failure, timeout=0
        )

    monkeypatch.setattr(delivery_queue, "enqueue_and_wait", wait_with_zero_deadline)
    assert scheduler.run_one_job(job)

    queue_row = delivery_queue.get_status(execution_id)
    assert queue_row is not None
    assert queue_row["status"] == ("failed" if queue_state == "failed_after_wait" else queue_state)
    row = executions.latest_execution(job["id"])
    assert row["delivery_outcome"] == expected_delivery_outcome
    saved = jobs.get_job(job["id"]) or {}
    assert saved["last_status"] == expected_job_status
    assert saved["last_delivery_outcome"] == expected_delivery_outcome
    assert saved.get("last_delivery_queued") is None
    if queue_state == "pending":
        note = _manual_run_delivery_note(job["deliver"], saved).casefold()
        assert any(word in note for word in ("queued", "in progress"))
        assert "bot chat" not in note
        assert "was delivered" not in note
        assert _format_job(saved)["last_delivery_outcome"] == expected_delivery_outcome
        completion = _manual_run_completion(
            {"success": True, "error": None},
            job["id"],
            job["name"],
            job["deliver"],
            started_at=time.time(),
        )
        result_line = next(line for line in completion["summary"].splitlines() if line.startswith("Result:"))
        assert any(word in result_line.casefold() for word in ("queued", "in progress"))
        assert "Result: ok" not in result_line
        assert "Result: FAILED" not in result_line
        assert "do not resend" in completion["summary"].casefold()
        assert completion["status"] == "completed"
    elif queue_state == "unknown":
        assert scheduler.drain_delivery_queue({}, None) == 0
        note = _manual_run_delivery_note(job["deliver"], saved).casefold()
        assert any(word in note for word in ("unknown", "uncertain", "unconfirmed"))
        assert "failed" not in note
        assert "was delivered" not in note
        assert _format_job(saved)["last_delivery_outcome"] == expected_delivery_outcome
        _assert_unknown_diagnostics(saved, model_failed=False)
        completion = _manual_run_completion(
            {"success": False, "error": saved["last_delivery_error"]},
            job["id"],
            job["name"],
            job["deliver"],
            started_at=time.time(),
        )
        result_line = next(line for line in completion["summary"].splitlines() if line.startswith("Result:"))
        assert any(word in result_line.casefold() for word in ("unknown", "uncertain", "unconfirmed"))
        assert "Result: ok" not in result_line
        assert "Result: FAILED" not in result_line
        assert "do not resend" in completion["summary"].casefold()
        assert completion["status"] == "error"
        jobs.mark_job_run(job["id"], True)
        assert (jobs.get_job(job["id"]) or {}).get("last_delivery_outcome") is None


@pytest.mark.parametrize("model_mode", ["failure", "crash"])
def test_model_failure_preserves_error_when_external_delivery_is_unknown(
    tmp_path, monkeypatch, model_mode,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)

    def forbidden_direct_send(*args, **kwargs):
        pytest.fail("external worker sent directly instead of using the native queue")

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", forbidden_direct_send)

    def run(job, **kwargs):
        if model_mode == "crash":
            raise RuntimeError("isolated provider failure")
        return False, "retained output", "final response", "isolated provider failure"

    monkeypatch.setattr(scheduler, "run_job", run)
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:fixture")
    execution = executions.create_execution(job["id"], source="fixture")
    job["execution_id"] = execution["id"]
    execution_id = execution["id"]
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution_id)

    original_wait = delivery_queue.enqueue_and_wait

    def wait_with_zero_deadline(execution_id, job, content, *, for_failure=False):
        delivery_queue.enqueue(execution_id, job, content, for_failure=for_failure)
        assert delivery_queue.claim_next()["execution_id"] == execution_id
        return original_wait(
            execution_id, job, content, for_failure=for_failure, timeout=0
        )

    monkeypatch.setattr(delivery_queue, "enqueue_and_wait", wait_with_zero_deadline)
    assert scheduler.run_one_job(job) is (model_mode != "crash")

    assert delivery_queue.get_status(execution_id)["status"] == "unknown"
    row = executions.latest_execution(job["id"])
    assert row["delivery_outcome"] == "unknown"
    saved = jobs.get_job(job["id"]) or {}
    assert saved["last_status"] == "error"
    assert "isolated provider failure" in saved["last_error"]
    assert saved["last_delivery_outcome"] == "unknown"
    assert scheduler.drain_delivery_queue({}, None) == 0
    note = _manual_run_delivery_note(job["deliver"], saved).casefold()
    assert any(word in note for word in ("unknown", "uncertain", "unconfirmed"))
    assert "failed" not in note
    assert "was delivered" not in note
    assert _format_job(saved)["last_delivery_outcome"] == "unknown"
    _assert_unknown_diagnostics(saved, model_failed=True)
