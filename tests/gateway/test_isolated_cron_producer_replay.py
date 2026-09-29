"""Real producer replay contract for one isolated cron execution/lane/slot."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cron import delivery_queue
from cron.scheduler_provider import _profile_cron_scope
from gateway.config import Platform
from gateway.run import _profile_runtime_scope

from cron import scheduler_delivery
from tests.gateway.test_api_server_isolated_cron_delivery import (  # noqa: F401
    _key,
    _reset_multiplex,
    adapter,
    profiles,
    root_home,
    runner,
    stub_primary,
    standalone_spy,
)
from tests.gateway.test_isolated_cron_producer import (
    _DecoyLocalSend,
    _app_with_receipt_capture,
    _executor_home,
    _listener_drain,
    _run_executor_job,
)


@pytest.mark.parametrize(
    "terminal_status",
    ("unknown", "failed", "suppressed"),
)
@pytest.mark.asyncio
async def test_native_terminal_receipt_replay_preserves_terminal_status(
    terminal_status,
    adapter,
    runner,
    stub_primary,
    standalone_spy,
    profiles,
    tmp_path,
    monkeypatch,
):
    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(
            tmp_path, gateway_url=gateway_url, cron_key=_key("alice"))
        decoy = _DecoyLocalSend()
        decoy_map = {Platform.TELEGRAM: decoy}
        loop = asyncio.get_running_loop()

        job, ok, _stored_job, _ledger, local_queue_row = await asyncio.to_thread(
            _run_executor_job, executor_home, decoy_map, loop, monkeypatch)

        assert ok is True
        assert job.get("execution_id")
        assert local_queue_row is None
        assert decoy.calls == []
        assert standalone_spy == []
        assert stub_primary.calls == []
        assert len(captured) == 1
        _path, status, first_receipt = captured[0]
        assert status == 202
        assert first_receipt and first_receipt.get("execution_id")
        remote_id = first_receipt["execution_id"]

        with adapter._profile_scope("alice"):
            pending = delivery_queue.get_status(remote_id)
            assert pending is not None and pending["status"] == "pending"
            claimed = delivery_queue.claim_next()
            assert claimed is not None and claimed["execution_id"] == remote_id
            native_error = None if terminal_status == "suppressed" else (
                f"native {terminal_status} fixture"
            )
            assert delivery_queue._finish(
                remote_id,
                error=native_error,
                suppressed=terminal_status == "suppressed",
                unknown=terminal_status == "unknown",
            ) is True
            terminal = delivery_queue.get_status(remote_id)
        assert terminal is not None
        assert terminal["status"] == terminal_status
        assert terminal["error"] == native_error

        async def replay():
            with _profile_runtime_scope(Path(executor_home)):
                with _profile_cron_scope(executor_home):
                    return await asyncio.to_thread(
                        scheduler_delivery._deliver_result,
                        job,
                        "hello from cron",
                        adapters=decoy_map,
                        loop=loop,
                    )

        replay_error = await replay()
        if terminal_status == "suppressed":
            assert replay_error is None
        else:
            assert replay_error
        assert len(captured) == 2
        _path, status, replay_receipt = captured[1]
        assert status == 200
        assert replay_receipt and replay_receipt["execution_id"] == remote_id
        assert replay_receipt["status"] == terminal_status
        assert job.get("_native_queue_status") == terminal_status
        assert job.get("_notification_all_targets_suppressed", False) is (
            terminal_status == "suppressed"
        )
        assert decoy.calls == []
        assert standalone_spy == []
        assert stub_primary.calls == []

        with adapter._profile_scope("alice"):
            assert await _listener_drain(runner, "alice", loop) == 0
            unchanged = delivery_queue.get_status(remote_id)
        assert unchanged is not None
        assert unchanged["status"] == terminal_status
        assert unchanged["error"] == native_error
        assert stub_primary.calls == []

        repeated_error = await replay()
        if terminal_status == "suppressed":
            assert repeated_error is None
        else:
            assert repeated_error
        assert len(captured) == 3
        _path, status, repeated_receipt = captured[2]
        assert status == 200
        assert repeated_receipt and repeated_receipt["execution_id"] == remote_id
        assert repeated_receipt["status"] == terminal_status
        with adapter._profile_scope("alice"):
            repeated = delivery_queue.get_status(remote_id)
        assert repeated is not None
        assert repeated["status"] == terminal_status
        assert repeated["error"] == native_error
        assert stub_primary.calls == []


@pytest.mark.asyncio
async def test_repeated_admission_same_execution_lane_slot_replays_without_second_send(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch,
):
    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(
            tmp_path, gateway_url=gateway_url, cron_key=_key("alice"))
        decoy = _DecoyLocalSend()
        decoy_map = {Platform.TELEGRAM: decoy}
        loop = asyncio.get_running_loop()

        job, ok, _stored_job, _ledger, local_queue_row = await asyncio.to_thread(
            _run_executor_job, executor_home, decoy_map, loop, monkeypatch)

        assert ok is True
        assert job.get("execution_id")
        assert local_queue_row is None
        assert decoy.calls == []
        assert standalone_spy == []
        assert stub_primary.calls == []

        assert len(captured) == 1
        _path, status, first_receipt = captured[0]
        assert status == 202
        assert first_receipt and first_receipt.get("execution_id")
        remote_id = first_receipt["execution_id"]

        with adapter._profile_scope("alice"):
            original_row = delivery_queue.get_status(remote_id)
        assert original_row is not None
        assert original_row["status"] == "pending"
        original_content = original_row["content"]

        async def admit(content: str):
            with _profile_runtime_scope(Path(executor_home)):
                with _profile_cron_scope(executor_home):
                    return await asyncio.to_thread(
                        scheduler_delivery._deliver_result,
                        job,
                        content,
                        adapters=decoy_map,
                        loop=loop,
                    )

        same_error = await admit("hello from cron")
        assert same_error is None
        assert len(captured) == 2
        _path, status, replay_receipt = captured[1]
        assert status == 202
        assert replay_receipt and replay_receipt["execution_id"] == remote_id
        assert replay_receipt.get("status") == "pending"
        assert job.get("_native_queue_status") == "queued"

        with adapter._profile_scope("alice"):
            unchanged = delivery_queue.get_status(remote_id)
        assert unchanged is not None
        assert unchanged["status"] == "pending"
        assert unchanged["content"] == original_content

        conflict_error = await admit("changed plaintext")
        assert conflict_error
        assert len(captured) == 3
        _path, status, conflict_receipt = captured[2]
        assert status == 409
        assert conflict_receipt is not None
        assert conflict_receipt.get("error")
        assert job.get("_native_queue_status") == "unknown"
        assert decoy.calls == []
        assert standalone_spy == []

        with adapter._profile_scope("alice"):
            after_conflict = delivery_queue.get_status(remote_id)
        assert after_conflict is not None
        assert after_conflict["status"] == "pending"
        assert after_conflict["content"] == original_content

        with adapter._profile_scope("alice"):
            assert await _listener_drain(runner, "alice", loop) == 1
            delivered = delivery_queue.get_status(remote_id)
        assert delivered is not None and delivered["status"] == "delivered"
        assert len(stub_primary.calls) == 1

        with adapter._profile_scope("alice"):
            assert await _listener_drain(runner, "alice", loop) == 0
        assert len(stub_primary.calls) == 1

        final_error = await admit("hello from cron")
        assert final_error is None
        assert len(captured) == 4
        _path, status, final_receipt = captured[3]
        assert status == 200
        assert final_receipt and final_receipt["execution_id"] == remote_id
        assert job.get("_native_queue_status") == "delivered"
        assert decoy.calls == []
        assert standalone_spy == []
        assert len(stub_primary.calls) == 1

        with adapter._profile_scope("alice"):
            terminal = delivery_queue.get_status(remote_id)
        assert terminal is not None and terminal["status"] == "delivered"
