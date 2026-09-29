"""Effective scheduler config must fail closed without changing ordinary local delivery."""

from __future__ import annotations

import asyncio
import http.client
import logging
from pathlib import Path

import pytest

from cron import jobs as cron_jobs
from cron.scheduler_provider import _profile_cron_scope
from gateway.config import Platform
from gateway.run import _profile_runtime_scope
from hermes_cli.config_read_errors import FailedConfigRead
from tests.gateway.test_api_server_isolated_cron_delivery import (
    _key,
    _reset_multiplex,
    standalone_spy,
)
from tests.gateway.test_isolated_cron_producer import _DecoyLocalSend, _executor_home


_CONFIG_SENTINEL = "fictional-config-failure-sentinel"
_CONTENT = "fixture-only-content-marker"


def _deliver_in_native_scopes(home: Path, effective_config, decoy, loop, monkeypatch):
    import cron.scheduler as scheduler
    from cron.scheduler_delivery import _deliver_result

    with _profile_runtime_scope(home):
        with _profile_cron_scope(home):
            job = cron_jobs.create_job(
                "fixture only", "every 1h", deliver="telegram:1001"
            )
            if isinstance(effective_config, BaseException):
                def _raise_config_read():
                    raise effective_config
                read_config = _raise_config_read
            else:
                def _return_config():
                    return effective_config
                read_config = _return_config
            monkeypatch.setattr(scheduler, "load_config", read_config)
            result = _deliver_result(
                job, _CONTENT, adapters={Platform.TELEGRAM: decoy}, loop=loop
            )
    return result, job


def _http_request_counter(monkeypatch):
    calls: list[tuple] = []

    def request(self, method, url, *args, **kwargs):
        calls.append((self.__class__, method, url))
        raise AssertionError("fictional config test attempted outbound HTTP")

    monkeypatch.setattr(http.client.HTTPConnection, "request", request)
    monkeypatch.setattr(http.client.HTTPSConnection, "request", request)
    return calls


@pytest.mark.parametrize(
    "effective_config",
    [
        OSError(_CONFIG_SENTINEL),
        None,
        [],
        {"cron": []},
        {"cron": None},
        {"cron": "invalid"},
        FailedConfigRead({"cron": {}}, error=OSError("harmless-config-read")),
    ],
    ids=(
        "reader-oserror",
        "none-root",
        "nonmapping-root",
        "cron-list",
        "cron-none",
        "cron-string",
        "failed-config-read",
    ),
)
@pytest.mark.asyncio
async def test_effective_config_reader_errors_fail_closed_before_transport(
    effective_config, tmp_path: Path, monkeypatch, standalone_spy, _reset_multiplex, caplog
):
    caplog.set_level(logging.ERROR)
    http_calls = _http_request_counter(monkeypatch)
    home = _executor_home(tmp_path, gateway_url="https://127.0.0.1/fictional", cron_key=_key("executor"))
    decoy = _DecoyLocalSend()
    loop = asyncio.get_running_loop()

    result, _job = await asyncio.to_thread(
        _deliver_in_native_scopes,
        home,
        effective_config,
        decoy,
        loop,
        monkeypatch,
    )

    assert isinstance(result, str) and result.strip()
    assert _CONFIG_SENTINEL not in result
    assert _CONFIG_SENTINEL not in caplog.text
    assert decoy.calls == []
    assert standalone_spy == []
    assert http_calls == []


@pytest.mark.parametrize(
    "effective_config",
    [{}, {"cron": {}}, {"cron": {"delivery_gateway_url": ""}}, {"cron": {"delivery_gateway_url": None}}],
    ids=("empty-root", "empty-cron", "empty-url", "none-url"),
)
@pytest.mark.asyncio
async def test_absent_or_empty_effective_gateway_retains_local_delivery(
    effective_config, tmp_path: Path, monkeypatch, standalone_spy, _reset_multiplex
):
    http_calls = _http_request_counter(monkeypatch)
    home = _executor_home(tmp_path, gateway_url="https://127.0.0.1/fictional", cron_key=_key("executor"))
    decoy = _DecoyLocalSend()
    loop = asyncio.get_running_loop()

    result, job = await asyncio.to_thread(
        _deliver_in_native_scopes,
        home,
        effective_config,
        decoy,
        loop,
        monkeypatch,
    )

    assert result is None
    assert len(decoy.calls) == 1
    chat_id, content, _metadata = decoy.calls[0]
    assert chat_id == "1001"
    assert _CONTENT in content
    assert job["prompt"] == "fixture only"
    assert standalone_spy == []
    assert http_calls == []
