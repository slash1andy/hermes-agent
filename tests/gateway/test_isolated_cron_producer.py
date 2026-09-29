"""Behavioral contract for the (not yet implemented) native isolated executor producer: a real
``cron.scheduler.run_one_job`` on a SEPARATE executor home, configured with
``cron.delivery_gateway_url``, must admit its outbound text through the existing authenticated
``POST /p/{profile}/cron/deliveries`` endpoint (see ``test_api_server_isolated_cron_delivery.py``,
whose receiver side is real and already wired) instead of using any local adapter — delivery
happens only on the receiver's own later native drain.

CURRENT LEAF SCOPE: only the first positive slice (real run_one_job -> real HTTP receipt -> real
native drain). No replay/conflict/failure/scope-crossover/fallback cases here; see
``native-isolated-producer-contract.md`` for the full future matrix.

The listener side is reused unmodified from ``test_api_server_isolated_cron_delivery.py`` via an
ordinary package import (real ``APIServerAdapter``/profile-prefix middleware/route table, real
``cron.delivery_queue``, real ``cron.scheduler.drain_delivery_queue`` -> real ``_deliver_result``).
Only the executor side (a distinct temp home, its own local live-adapter decoy, and a passive
receipt-capturing middleware) is built here — the contract explicitly derives the remote
operation identity from execution+lane+slot, so the receiver queue is never looked up by the
executor's own raw durable execution id; the test reads the real 202 receipt instead.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from cron import delivery_queue
from cron import executions as cron_executions
from cron import jobs as cron_jobs
from cron.scheduler_provider import _profile_cron_scope
from gateway.config import Platform
from gateway.platforms.base import SendResult

from tests.gateway.test_api_server_isolated_cron_delivery import (  # noqa: F401 (fixtures)
    _key,
    _reset_multiplex,
    adapter,
    profiles,
    root_home,
    runner,
    stub_primary,
    standalone_spy,
)
from tests.gateway.test_api_server_isolated_cron_delivery import _drain as _listener_drain


class _DecoyLocalSend:
    """The executor's OWN local live-adapter double. Explicit remote mode (``cron.
    delivery_gateway_url`` configured) must win before this is ever reached — proves no direct
    local fallback."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def send(self, chat_id, content, metadata=None, **_kwargs) -> SendResult:
        self.calls.append((chat_id, content, metadata))
        return SendResult(success=True, message_id=f"decoy-{len(self.calls)}")


def _executor_home(tmp_path: Path, *, gateway_url: str, cron_key: str) -> Path:
    """A temp home distinct from the listener's profile homes: no shared jobs/queue directories."""
    home = tmp_path / "executor-home"
    home.mkdir(parents=True)
    (home / ".env").write_text(f"CRON_DELIVERY_KEY={cron_key}\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "platforms": {"telegram": {"enabled": True}},
            "cron": {"delivery_gateway_url": gateway_url, "wrap_response": True},
        }),
        encoding="utf-8")
    return home


def _app_with_receipt_capture(adapter, captured: list) -> web.Application:
    """Mirrors ``connect()``'s own registration exactly (see the sibling ``app`` fixture), plus one
    ADDITIONAL passive middleware placed AFTER the profile-prefix middleware so it wraps the real
    route handler directly: it awaits the REAL handler, records the real JSON receipt/status/path
    it produced, and returns the response UNCHANGED. It implements no auth/queue/dedup/policy."""

    @web.middleware
    async def _capture_receipt(request: "web.Request", handler):
        response = await handler(request)
        if request.path.endswith("/cron/deliveries"):
            body = None
            if response.body:
                try:
                    body = json.loads(response.body.decode("utf-8"))
                except Exception:
                    body = None
            captured.append((request.path, response.status, body))
        return response

    application = web.Application(
        middlewares=[adapter._make_profile_prefix_middleware(), _capture_receipt])
    for method, path, route_handler in adapter._http_route_table():
        application.router.add_route(method, path, route_handler)
        application.router.add_route(method, f"/p/{{profile}}{path}", route_handler)
    application.router.add_route("*", "/p/{profile}/{tail:.*}", adapter._handle_profile_ingress)
    return application


def _run_executor_job(executor_home: Path, adapters: dict, loop, monkeypatch):
    """Runs entirely on the worker thread `run_one_job` itself runs on: job creation, the run, and
    every executor-scoped store read happen under the SAME ``_profile_cron_scope`` (real home
    override + real cron store scoping — the identical seam the native ticker uses), never a bare
    home-override with no cron-store/secret-scope binding."""
    import cron.scheduler as scheduler

    with _profile_cron_scope(executor_home):
        job = cron_jobs.create_job(
            "fixture only", "every 1h", name="nightly export", deliver="telegram:1001")
        monkeypatch.setattr(
            scheduler, "run_job",
            lambda *_a, **_kw: (True, "doc", "hello from cron", None))
        ok = scheduler.run_one_job(job, adapters=adapters, loop=loop)
        stored_job = cron_jobs.get_job(job["id"])
        ledger = cron_executions.latest_execution(job["id"])
        local_queue_row = delivery_queue.get_status(job.get("execution_id", ""))
    return job, ok, stored_job, ledger, local_queue_row


@pytest.mark.asyncio
async def test_run_one_job_admits_through_configured_delivery_gateway_then_native_drain_sends_once(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch,
):
    # wrap_response: true in BOTH homes for this check — the executor's own real _deliver_result
    # wraps ONCE using the real job's name/id before ever reaching a transport lane; the listener's
    # drain of a marked isolated envelope (no job name/id of its own) must not wrap again.
    alice_config_path = profiles["alice"] / "config.yaml"
    alice_config = yaml.safe_load(alice_config_path.read_text(encoding="utf-8"))
    alice_config["cron"] = {**alice_config.get("cron", {}), "wrap_response": True}
    alice_config_path.write_text(yaml.safe_dump(alice_config), encoding="utf-8")

    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(
            tmp_path, gateway_url=gateway_url, cron_key=_key("alice"))
        decoy = _DecoyLocalSend()

        loop = asyncio.get_running_loop()
        job, ok, stored_job, ledger, local_queue_row = await asyncio.to_thread(
            _run_executor_job, executor_home, {Platform.TELEGRAM: decoy}, loop, monkeypatch)

        assert ok is True
        assert decoy.calls == [], (
            "explicit remote delivery config must win: the executor's own local live adapter "
            "must never be used once cron.delivery_gateway_url is configured")
        assert standalone_spy == [], "no standalone fallback once remote delivery is configured"
        assert stub_primary.calls == [], "no send before the receiver's own native drain"
        assert local_queue_row is None, (
            "the executor must never admit its own local cron.delivery_queue row — that queue is "
            "the restart-safe-worker/receiver mechanism, not this HTTP producer path")

        # The contract derives the remote operation identity from execution+lane+slot, never the
        # executor's raw durable execution id — read the real 202 receipt instead of guessing it.
        assert len(captured) == 1, "exactly one HTTP request must reach the real native handler"
        _path, status, receipt = captured[0]
        assert status == 202
        assert receipt is not None and receipt.get("execution_id")
        remote_id = receipt["execution_id"]

        with adapter._profile_scope("alice"):
            pending = delivery_queue.get_status(remote_id)
            mirrored_job = cron_jobs.get_job(job["id"])
        assert pending is not None and pending["status"] == "pending"
        assert mirrored_job is None, "a marked isolated envelope must never mirror into jobs.json"

        queued_content = pending["content"]
        assert job["name"] in queued_content
        assert job["id"] in queued_content
        assert queued_content.count("Cronjob Response:") == 1
        assert queued_content.count("hello from cron") == 1

        with adapter._profile_scope("alice"):
            processed = await _listener_drain(runner, "alice", loop)
        assert processed == 1
        assert len(stub_primary.calls) == 1
        chat_id, content, _meta = stub_primary.calls[0]
        assert chat_id == "1001"
        # The listener never re-wraps a marked envelope: sent content is byte-identical to what
        # was queued — a change here would mean the plaintext (or the header) was wrapped twice.
        assert content == queued_content

        with adapter._profile_scope("alice"):
            delivered = delivery_queue.get_status(remote_id)
            processed_again = await _listener_drain(runner, "alice", loop)
        assert delivered is not None and delivered["status"] == "delivered"
        assert processed_again == 0
        assert len(stub_primary.calls) == 1
        assert standalone_spy == []
        assert decoy.calls == []

        assert stored_job is not None
        assert stored_job["last_status"] == "delivery_queued"
        assert stored_job["last_delivery_outcome"] == "queued"
        assert not stored_job.get("last_delivery_queued"), (
            "last_delivery_queued is the Bot Chat receipt field — this HTTP producer path must "
            "never stuff its receipt there")
        assert ledger is not None and ledger["delivery_outcome"] == "queued"
