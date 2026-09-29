"""Configured native HTTP producer safety matrix: admission is narrow, single-shot, and bounded."""

from __future__ import annotations

import asyncio
import http.client
import logging
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from cron import delivery_queue
from gateway.config import Platform
from tests.gateway.test_api_server_isolated_cron_delivery import (
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

_URL_SENTINEL = "fictional-http-url-sentinel-4e91"
_SECRET_SENTINEL = "fictional-http-key-sentinel-8a20"
_RESPONSE_SENTINEL = "fictional-hostile-response-sentinel-5c37"
_FIXED_REQUEST_ERROR = "forbidden HTTP request sentinel"


@pytest.fixture
def forbidden_http_requests(monkeypatch):
    calls: list[tuple] = []

    def request(self, method, path, *args, **kwargs):
        calls.append((self.__class__, method, path))
        raise AssertionError(_FIXED_REQUEST_ERROR)

    monkeypatch.setattr(http.client.HTTPConnection, "request", request)
    monkeypatch.setattr(http.client.HTTPSConnection, "request", request)
    return calls


@pytest.mark.parametrize(
    "gateway_url",
    (
        "http://[broken",
        "http:///p/alice/cron/deliveries",
        "http://127.0.0.1:fictional-http-url-sentinel-4e91/p/alice/cron/deliveries",
        "http://127.0.0.1:0/p/alice/cron/deliveries",
        "http://127.0.0.1:65536/p/alice/cron/deliveries",
        "http://fictional-http-url-sentinel-4e91:password@127.0.0.1/p/alice/cron/deliveries",
        "http://127.0.0.1/p/alice/cron/deliveries?fictional-http-url-sentinel-4e91",
        "http://127.0.0.1/p/alice/cron/deliveries#fictional-http-url-sentinel-4e91",
        " http://127.0.0.1/p/alice/cron/deliveries",
        "http://127.0.0.1/p/alice/cron/deliveries ",
        "http://127.0.0.1/p/alice/cron/deliveries\n",
        "http://127.0.0.1/fictional-http-url-sentinel-4e91",
        "http://127.0.0.1/p/alice/cron/deliveries/",
    ),
    ids=(
        "broken-authority",
        "missing-host",
        "textual-port",
        "port-zero",
        "port-too-large",
        "embedded-credentials",
        "query",
        "fragment",
        "leading-whitespace",
        "trailing-whitespace",
        "control-character",
        "wrong-path",
        "trailing-slash",
    ),
)
@pytest.mark.asyncio
async def test_malformed_configured_full_native_endpoint_fails_before_http_or_fallback(
    gateway_url: str,
    tmp_path: Path,
    monkeypatch,
    caplog,
    forbidden_http_requests,
    standalone_spy,
    _reset_multiplex,
):
    caplog.set_level(logging.DEBUG)
    executor_home = _executor_home(
        tmp_path,
        gateway_url=gateway_url,
        cron_key=_SECRET_SENTINEL,
    )
    decoy = _DecoyLocalSend()

    job, processed, stored, ledger, local_row = await asyncio.to_thread(
        _run_executor_job,
        executor_home,
        {Platform.TELEGRAM: decoy},
        asyncio.get_running_loop(),
        monkeypatch,
    )

    assert processed is True
    assert stored is not None and stored["last_status"] == "delivery_failed"
    assert stored.get("last_delivery_error", "").strip()
    assert "cron.delivery_gateway_url" in stored["last_delivery_error"]
    assert ledger is not None and ledger["delivery_outcome"] == "failed"
    assert local_row is None
    assert forbidden_http_requests == []
    assert decoy.calls == []
    assert standalone_spy == []
    assert _URL_SENTINEL not in caplog.text
    assert _SECRET_SENTINEL not in caplog.text
    assert gateway_url not in caplog.text
    assert gateway_url not in stored.get("last_delivery_error", "")
    assert _URL_SENTINEL not in stored.get("last_delivery_error", "")
    assert _SECRET_SENTINEL not in stored.get("last_delivery_error", "")


@pytest.mark.parametrize(
    "fault_kind",
    ("redirect-301", "redirect-302", "redirect-303", "redirect-307", "redirect-308", "oversize"),
)
@pytest.mark.asyncio
async def test_native_admission_is_single_shot_when_response_is_redirect_or_oversize(
    fault_kind: str,
    adapter,
    runner,
    stub_primary,
    standalone_spy,
    profiles,
    root_home,
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    captured: list = []
    trap_calls: list[tuple] = []
    app = _app_with_receipt_capture(adapter, captured)

    async def trap_handler(request: web.Request) -> web.Response:
        trap_calls.append((request.method, request.path))
        return web.Response(text="trap reply")

    trap_app = web.Application()
    trap_app.router.add_route("*", "/{tail:.*}", trap_handler)
    loop = asyncio.get_running_loop()
    caplog.set_level(logging.DEBUG)

    async with TestClient(TestServer(trap_app)) as trap_client:
        trap_url = str(trap_client.make_url("/trap"))
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.setenv(name, trap_url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)

        @web.middleware
        async def hostile_response(request: web.Request, handler):
            response = await handler(request)
            if not request.path.endswith("/cron/deliveries"):
                return response
            assert response.status == 202
            assert captured and captured[-1][1] == 202
            if fault_kind == "oversize":
                return web.Response(body=(f"{_RESPONSE_SENTINEL}:" + "x" * (64 * 1024 + 1)).encode())
            status = int(fault_kind.rsplit("-", 1)[1])
            return web.Response(status=status, headers={"Location": trap_url})

        app.middlewares.insert(1, hostile_response)
        read_sizes: list[int | None] = []
        original_read = http.client.HTTPResponse.read

        def passive_read(response, amt=None):
            read_sizes.append(amt)
            return original_read(response, amt)

        monkeypatch.setattr(http.client.HTTPResponse, "read", passive_read)

        async with TestClient(TestServer(app)) as client:
            gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
            executor_home = _executor_home(
                tmp_path,
                gateway_url=gateway_url,
                cron_key=_key("alice"),
            )
            decoy = _DecoyLocalSend()
            job, processed, stored, ledger, local_row = await asyncio.to_thread(
                _run_executor_job,
                executor_home,
                {Platform.TELEGRAM: decoy},
                loop,
                monkeypatch,
            )

            assert processed is True
            assert len(captured) == 1
            _path, status, receipt = captured[0]
            assert status == 202 and receipt and receipt.get("execution_id")
            remote_id = receipt["execution_id"]
            assert stored is not None and stored["last_status"] == "delivery_unknown"
            assert ledger is not None and ledger["delivery_outcome"] == "unknown"
            assert stored.get("last_delivery_error", "").strip()
            assert local_row is None
            assert decoy.calls == []
            assert standalone_spy == []
            assert trap_calls == []
            assert _RESPONSE_SENTINEL not in caplog.text
            assert _RESPONSE_SENTINEL not in stored.get("last_delivery_error", "")
            assert read_sizes and all(isinstance(size, int) and 0 < size <= 65537 for size in read_sizes)

            with adapter._profile_scope("alice"):
                pending = delivery_queue.get_status(remote_id)
                assert pending is not None and pending["status"] == "pending"
                assert await _listener_drain(runner, "alice", loop) == 1
                assert await _listener_drain(runner, "alice", loop) == 0
            assert len(captured) == 1
            assert len(trap_calls) == 0
            assert len(stub_primary.calls) == 1
            assert decoy.calls == []
            assert standalone_spy == []
