"""Transport-response uncertainty after native admission has already succeeded."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from cron import delivery_queue
from gateway.config import Platform
from tests.gateway.test_api_server_isolated_cron_delivery import (  # noqa: F401
    _key,
    _reset_multiplex,
    adapter,
    profiles,
    root_home,
    runner,
    standalone_spy,
    stub_primary,
)
from tests.gateway.test_isolated_cron_producer import (
    _DecoyLocalSend,
    _app_with_receipt_capture,
    _executor_home,
    _listener_drain,
    _run_executor_job,
)


_FAULT_SENTINEL = "private-transport-fault-sentinel-7d2a"


@pytest.mark.parametrize(
    "fault_kind",
    (
        "503",
        "malformed-202",
        "mismatched-202",
        "unexpected-200-status",
        "unhashable-202-status",
    ),
    ids=(
        "http-503",
        "malformed-202",
        "mismatched-execution-id",
        "unexpected-http-200-status",
        "unhashable-http-202-status",
    ),
)
@pytest.mark.asyncio
async def test_admitted_delivery_response_uncertainty_is_unknown_until_native_drain(
    fault_kind,
    adapter,
    runner,
    stub_primary,
    standalone_spy,
    profiles,
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    @web.middleware
    async def fault(request: web.Request, handler):
        response = await handler(request)
        if request.path.endswith("/cron/deliveries"):
            assert response.status == 202
            assert captured and captured[-1][1] == 202
            if fault_kind == "503":
                return web.Response(
                    status=503,
                    body=f"{_FAULT_SENTINEL}: service unavailable".encode(),
                )
            if fault_kind == "malformed-202":
                return web.Response(
                    status=202,
                    body=f'{{"execution_id":"{_FAULT_SENTINEL}"'.encode(),
                    content_type="application/json",
                )
            if fault_kind == "mismatched-202":
                return web.Response(
                    status=202,
                    text=json.dumps(
                        {
                            "execution_id": f"{_FAULT_SENTINEL}-mismatched",
                            "status": "pending",
                        }
                    ),
                    content_type="application/json",
                )
            if fault_kind == "unexpected-200-status":
                return web.Response(
                    status=200,
                    text=json.dumps(
                        {
                            "execution_id": captured[-1][2]["execution_id"],
                            "status": "not-a-terminal-status",
                        }
                    ),
                    content_type="application/json",
                )
            return web.Response(
                status=202,
                text=json.dumps(
                    {
                        "execution_id": captured[-1][2]["execution_id"],
                        "status": [],
                    }
                ),
                content_type="application/json",
            )
        return response

    # The profile middleware remains first; this replaces only the response after
    # the capture middleware has observed the real native admission receipt.
    app.middlewares.insert(1, fault)

    caplog.set_level(logging.DEBUG)
    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(
            tmp_path, gateway_url=gateway_url, cron_key=_key("alice")
        )
        decoy = _DecoyLocalSend()
        decoy_map = {Platform.TELEGRAM: decoy}
        loop = asyncio.get_running_loop()

        job, ok, stored_job, ledger, local_queue_row = await asyncio.to_thread(
            _run_executor_job, executor_home, decoy_map, loop, monkeypatch
        )

        assert ok is True
        assert len(captured) == 1
        _path, status, genuine_receipt = captured[0]
        assert status == 202
        assert genuine_receipt and genuine_receipt.get("execution_id")
        remote_id = genuine_receipt["execution_id"]

        assert stored_job is not None
        assert stored_job["last_status"] == "delivery_unknown"
        assert stored_job["last_delivery_outcome"] == "unknown"
        saved_error = stored_job.get("last_delivery_error")
        assert isinstance(saved_error, str) and saved_error.strip()
        assert _FAULT_SENTINEL not in saved_error
        assert ledger is not None
        assert ledger["delivery_outcome"] == "unknown"
        assert _FAULT_SENTINEL not in caplog.text

        assert local_queue_row is None
        assert decoy.calls == []
        assert standalone_spy == []
        assert stub_primary.calls == []

        with adapter._profile_scope("alice"):
            pending = delivery_queue.get_status(remote_id)
        assert pending is not None and pending["status"] == "pending"

        with adapter._profile_scope("alice"):
            processed = await _listener_drain(runner, "alice", loop)
        assert processed == 1
        assert len(stub_primary.calls) == 1

        with adapter._profile_scope("alice"):
            processed_again = await _listener_drain(runner, "alice", loop)
            delivered = delivery_queue.get_status(remote_id)
        assert processed_again == 0
        assert delivered is not None and delivered["status"] == "delivered"
        assert len(stub_primary.calls) == 1
        assert decoy.calls == []
        assert standalone_spy == []
        assert len(captured) == 1
