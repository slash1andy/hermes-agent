"""A->B->A proof that isolated cron producers keep home, key, and target bindings separate."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cron import delivery_queue
from cron import executions as cron_executions
from cron import jobs as cron_jobs
from cron.scheduler_provider import _profile_cron_scope
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
)
from tests.gateway.test_api_server_isolated_cron_delivery import _drain as _listener_drain


def _run_profile_job(
    executor_home: Path,
    adapters: dict,
    loop,
    monkeypatch,
    *,
    chat: str,
    label: str,
):
    """Create and run one real job while every store and secret lookup is scoped to its home."""
    import cron.scheduler as scheduler

    with _profile_cron_scope(executor_home):
        job = cron_jobs.create_job(
            prompt=f"fixture prompt {label}",
            schedule="every 1h",
            deliver=f"telegram:{chat}",
            name=label,
        )
        monkeypatch.setattr(
            scheduler,
            "run_job",
            lambda *_args, **_kwargs: (True, "doc", f"fixture output {label}", None),
        )
        ok = scheduler.run_one_job(job, adapters=adapters, loop=loop)
        stored_job = cron_jobs.get_job(job["id"])
        ledger = cron_executions.latest_execution(job["id"])
        queue_row = delivery_queue.get_status(job.get("execution_id", ""))
    return job, ok, stored_job, ledger, queue_row


@pytest.mark.asyncio
async def test_run_one_job_a_b_a_keeps_executor_and_receiver_profiles_isolated(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch,
):
    captured: list[tuple[str, int, dict | None]] = []
    app = _app_with_receipt_capture(adapter, captured)
    loop = asyncio.get_running_loop()
    alice_decoy = _DecoyLocalSend()
    bob_decoy = _DecoyLocalSend()
    alice_adapters = {Platform.TELEGRAM: alice_decoy}
    bob_adapters = {Platform.TELEGRAM: bob_decoy}

    async with TestClient(TestServer(app)) as client:
        alice_url = str(client.make_url("/p/alice/cron/deliveries"))
        bob_url = str(client.make_url("/p/bob/cron/deliveries"))
        alice_home = _executor_home(
            tmp_path / "executor-alice", gateway_url=alice_url, cron_key=_key("alice"))
        bob_home = _executor_home(
            tmp_path / "executor-bob", gateway_url=bob_url, cron_key="")

        alice_one, alice_ok, alice_stored, alice_ledger, alice_local = await asyncio.to_thread(
            _run_profile_job,
            alice_home,
            alice_adapters,
            loop,
            monkeypatch,
            chat="1001",
            label="alice-first",
        )

        monkeypatch.setenv("CRON_DELIVERY_KEY", _key("bob"))
        bob_missing, bob_missing_ok, bob_missing_stored, bob_missing_ledger, bob_missing_local = (
            await asyncio.to_thread(
                _run_profile_job,
                bob_home,
                bob_adapters,
                loop,
                monkeypatch,
                chat="2002",
                label="bob-missing-key",
            )
        )
        (bob_home / ".env").write_text(
            f"CRON_DELIVERY_KEY={_key('bob')}\n", encoding="utf-8")

        bob_one, bob_ok, bob_stored, bob_ledger, bob_local = await asyncio.to_thread(
            _run_profile_job,
            bob_home,
            bob_adapters,
            loop,
            monkeypatch,
            chat="2002",
            label="bob-positive",
        )
        alice_two, alice_two_ok, alice_two_stored, alice_two_ledger, alice_two_local = (
            await asyncio.to_thread(
                _run_profile_job,
                alice_home,
                alice_adapters,
                loop,
                monkeypatch,
                chat="1001",
                label="alice-second",
            )
        )

        assert [alice_ok, bob_missing_ok, bob_ok, alice_two_ok] == [True] * 4
        assert alice_local is None and bob_local is None and alice_two_local is None
        assert bob_missing_local is None
        assert bob_missing_stored is not None
        assert bob_missing_ledger is not None
        assert bob_missing_stored["last_status"] == "delivery_failed"
        assert bob_missing_ledger["delivery_outcome"] == "failed"
        assert alice_decoy.calls == [] and bob_decoy.calls == []
        assert standalone_spy == []

        assert len(captured) == 3
        assert all(status == 202 and receipt and receipt["execution_id"] for _, status, receipt in captured)
        receipt_ids = [receipt["execution_id"] for _, _, receipt in captured]
        assert len(set(receipt_ids)) == 3
        receipt_by_path = Counter(path for path, _, _ in captured)
        assert receipt_by_path == Counter({"/p/alice/cron/deliveries": 2, "/p/bob/cron/deliveries": 1})

        successful = {
            "alice-first": ("alice", alice_one),
            "bob-positive": ("bob", bob_one),
            "alice-second": ("alice", alice_two),
        }
        for (path, _status, receipt), (label, (owner, job)) in zip(captured, successful.items()):
            assert path == f"/p/{owner}/cron/deliveries"
            remote_id = receipt["execution_id"]
            with adapter._profile_scope(owner):
                own_row = delivery_queue.get_status(remote_id)
            sibling = "bob" if owner == "alice" else "alice"
            with adapter._profile_scope(sibling):
                sibling_row = delivery_queue.get_status(remote_id)
            assert own_row is not None and own_row["status"] == "pending"
            assert sibling_row is None
            assert job["id"] in own_row["content"]
            assert job["name"] in own_row["content"]
            assert f"fixture output {label}" in own_row["content"]

        with _profile_cron_scope(alice_home):
            assert {job["id"] for job in cron_jobs.load_jobs()} == {
                alice_one["id"], alice_two["id"]}
            assert cron_jobs.get_job(bob_one["id"]) is None
            assert cron_jobs.get_job(bob_missing["id"]) is None
        with _profile_cron_scope(bob_home):
            assert {job["id"] for job in cron_jobs.load_jobs()} == {
                bob_missing["id"], bob_one["id"]}
            assert cron_jobs.get_job(alice_one["id"]) is None
            assert cron_jobs.get_job(alice_two["id"]) is None

        for stored, ledger in (
            (alice_stored, alice_ledger), (bob_stored, bob_ledger),
            (alice_two_stored, alice_two_ledger),
        ):
            assert stored["last_status"] == "delivery_queued"
            assert stored["last_delivery_outcome"] == "queued"
            assert ledger["delivery_outcome"] == "queued"

        with adapter._profile_scope("alice"):
            assert await _listener_drain(runner, "alice", loop) == 2
        with adapter._profile_scope("bob"):
            assert await _listener_drain(runner, "bob", loop) == 1
        assert Counter(chat for chat, _content, _metadata in stub_primary.calls) == Counter(
            {"1001": 2, "2002": 1})
        assert len(stub_primary.calls) == 3

        expected = {
            "1001": [(alice_one["id"], alice_one["name"]), (alice_two["id"], alice_two["name"])],
            "2002": [(bob_one["id"], bob_one["name"])],
        }
        for chat, content, _metadata in stub_primary.calls:
            matches = [pair for pair in expected[chat] if pair[0] in content and pair[1] in content]
            assert len(matches) == 1
        with adapter._profile_scope("alice"):
            assert await _listener_drain(runner, "alice", loop) == 0
        with adapter._profile_scope("bob"):
            assert await _listener_drain(runner, "bob", loop) == 0
        assert len(stub_primary.calls) == 3
        assert standalone_spy == []
        assert alice_decoy.calls == [] and bob_decoy.calls == []
