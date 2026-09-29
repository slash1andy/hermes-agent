"""Real native isolated-cron-producer lane matrix: mixed Bot Chat + remote fan-out, a source-proven
per-target rejection, failure-lane warning suppression, and a local-artifact-only target under a
configured remote gateway.

Reuses the real receiver (``test_api_server_isolated_cron_delivery``) and executor helpers
(``test_isolated_cron_producer``) unmodified: real ``cron.scheduler.run_one_job`` ->
``_deliver_result``, real HTTP admission through ``APIServerAdapter``, real
``cron.delivery_queue``/``cron.scheduler.drain_delivery_queue``, and a real Bot Chat live-owner
mailbox (``tools.bot_live_delivery``) via ``hermes_cli.active_sessions.try_acquire_active_session``
— never a monkeypatch of ``_deliver_to_bot_chat`` or any queue/policy implementation. Only the
model seam (``cron.scheduler.run_job``) and the final live-adapter double (``_DecoyLocalSend``) are
replaced, exactly as the sibling producer test files do.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer

from cron import delivery_queue
from cron import executions as cron_executions
from cron import jobs as cron_jobs
from cron.scheduler_provider import _profile_cron_scope
from cron.scheduler_remote_delivery import operation_id
from gateway.config import Platform

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
from tests.gateway.test_isolated_cron_producer import (
    _DecoyLocalSend,
    _app_with_receipt_capture,
    _executor_home,
    _listener_drain,
)


def _run_lane_job(executor_home, adapters, loop, monkeypatch, run_job_result, **job_kwargs):
    """Runs one real job on the same worker thread ``run_one_job`` itself runs on, under the exact
    ``_profile_cron_scope`` the native ticker uses. Only ``cron.scheduler.run_job`` (the model seam)
    is replaced."""
    import cron.scheduler as scheduler

    with _profile_cron_scope(executor_home):
        job = cron_jobs.create_job("fixture only", "every 1h", **job_kwargs)
        monkeypatch.setattr(scheduler, "run_job", lambda *_a, **_kw: run_job_result)
        ok = scheduler.run_one_job(job, adapters=adapters, loop=loop)
        stored_job = cron_jobs.get_job(job["id"])
        ledger = cron_executions.latest_execution(job["id"])
        local_queue_row = delivery_queue.get_status(job.get("execution_id", ""))
    return job, ok, stored_job, ledger, local_queue_row


def _setup_bot_chat_live_owner(executor_home):
    """A real, capable Bot Chat live owner (current test process pid — never a subprocess), via
    the exact ``SessionDB``/``try_acquire_active_session`` sequence the mailbox contract tests use.
    Caller must ``lease.release()`` and ``db.close()``."""
    from hermes_state import SessionDB
    from hermes_cli.active_sessions import try_acquire_active_session

    db = SessionDB(db_path=executor_home / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat", surface="desktop", config={}, registry_home=executor_home,
        metadata={"live_session_id": "live", "bot_live_delivery_consumer": True})
    assert refusal is None
    return db, lease


def _canary_touch_guard(monkeypatch, canary_path) -> list:
    """Records any read/open/stat/resolve/is_file/exists touch of the ONE given synthetic canary path, mirroring the
    narrow decoy-path guard in ``test_media_directive_denied_before_queueing_no_file_read`` — never
    a global filesystem spy (which would also catch legitimate config/.env reads)."""
    touched: list = []
    canary_path.write_bytes(b"synthetic canary")
    def _is_target(path):
        return str(path) == str(canary_path)

    def _guard(name, real):
        def _wrapped(self, *args, **kwargs):
            if _is_target(self):
                touched.append(name)
            return real(self, *args, **kwargs)
        return _wrapped

    monkeypatch.setattr(Path, "resolve", _guard("resolve", Path.resolve))
    monkeypatch.setattr(Path, "stat", _guard("stat", Path.stat))
    monkeypatch.setattr(Path, "is_file", _guard("is_file", Path.is_file))
    monkeypatch.setattr(Path, "read_bytes", _guard("read_bytes", Path.read_bytes))
    monkeypatch.setattr(Path, "open", _guard("open", Path.open))
    monkeypatch.setattr(Path, "exists", _guard("exists", Path.exists))
    return touched


@pytest.mark.asyncio
async def test_mixed_bot_chat_and_remote_telegram_admits_remote_once_and_queues_bot_chat_locally(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch,
):
    """deliver="bot-chat,telegram:1001": the remote telegram target is admitted through the
    configured gateway exactly once; the local Bot Chat target is admitted through the real native
    mailbox (a live, capable owner — never a CLI subprocess), producing its own queued receipt with
    a native hex delivery id. Neither store's receipt leaks into the other's bookkeeping."""
    from hermes_state import SessionDB
    from hermes_cli.active_sessions import try_acquire_active_session
    from tools import bot_live_delivery as mailbox

    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(tmp_path, gateway_url=gateway_url, cron_key=_key("alice"))
        decoy = _DecoyLocalSend()
        loop = asyncio.get_running_loop()

        db = SessionDB(db_path=executor_home / "state.db")
        db.create_session(session_id="chat", source="cli")
        db.set_session_title("chat", "Bot Chat")
        lease, refusal = try_acquire_active_session(
            session_id="chat", surface="desktop", config={}, registry_home=executor_home,
            metadata={"live_session_id": "live", "bot_live_delivery_consumer": True})
        assert refusal is None
        try:
            job, ok, stored_job, ledger, local_queue_row = await asyncio.to_thread(
                _run_lane_job, executor_home, {Platform.TELEGRAM: decoy}, loop, monkeypatch,
                (True, "doc", "hello from mixed fan-out", None),
                deliver="bot-chat,telegram:1001", name="mixed fan-out")
        finally:
            lease.release()
            db.close()

        assert ok is True
        assert decoy.calls == [], "no local telegram send once remote delivery is configured"
        assert standalone_spy == []
        assert stub_primary.calls == [], "no send before the receiver's own native drain"
        assert local_queue_row is None, "the executor must never admit its own local delivery_queue row"

        assert len(captured) == 1, "exactly one real remote admission (telegram only)"
        _path, status, receipt = captured[0]
        assert status == 202
        remote_id = receipt["execution_id"]

        with adapter._profile_scope("alice"):
            pending = delivery_queue.get_status(remote_id)
        assert pending is not None and pending["status"] == "pending"

        receipts = job.get("_bot_chat_delivery_receipts") or {}
        assert set(receipts) == {"bot-chat:(own)"}, (
            "the remote telegram admission must never be recorded in the Bot Chat receipt mapping")
        bot_chat_receipt = receipts["bot-chat:(own)"]
        assert bot_chat_receipt["status"] == "queued"
        delivery_id = bot_chat_receipt["delivery_id"]
        assert re.fullmatch(r"[0-9a-f]{64}", delivery_id), "native hex delivery identity"

        mailbox_ticket = mailbox.read_delivery_result(executor_home, delivery_id)
        assert mailbox_ticket is not None and mailbox_ticket["status"] == "queued"
        assert mailbox_ticket["message"].count("hello from mixed fan-out") == 1

        assert stored_job is not None
        assert stored_job["last_status"] == "delivery_queued"
        assert stored_job["last_delivery_outcome"] == "queued"
        assert ledger is not None and ledger["delivery_outcome"] == "queued"

        with adapter._profile_scope("alice"):
            processed = await _listener_drain(runner, "alice", loop)
        assert processed == 1
        assert len(stub_primary.calls) == 1
        chat_id, _content, _meta = stub_primary.calls[0]
        assert chat_id == "1001"
        assert standalone_spy == []
        assert decoy.calls == []


@pytest.mark.asyncio
async def test_fanout_one_queued_one_source_proven_rejection_aggregates_unknown(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch,
):
    """deliver="telegram:1001,telegram:2002" against a gateway URL scoped to alice: 1001 is a route
    alice owns (queued); 2002 belongs to bob's route, so alice's own receiver rejects it
    pre-admission (403 -> "failed"). The mixed admit/reject batch aggregates conservatively to
    "unknown" rather than "queued", exactly two admission attempts are made (no retry of either),
    only the allowed target ever gets a receiver queue row, and the native drain sends once."""
    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(tmp_path, gateway_url=gateway_url, cron_key=_key("alice"))
        decoy = _DecoyLocalSend()
        loop = asyncio.get_running_loop()

        job, ok, stored_job, ledger, local_queue_row = await asyncio.to_thread(
            _run_lane_job, executor_home, {Platform.TELEGRAM: decoy}, loop, monkeypatch,
            (True, "doc", "hello from fan-out", None),
            deliver="telegram:1001,telegram:2002", name="fan-out rejection")

        assert ok is True
        assert local_queue_row is None
        assert decoy.calls == []
        assert standalone_spy == []
        assert stub_primary.calls == []

        assert len(captured) == 2, "exactly two admission attempts, no retry of either target"
        statuses = sorted(status for _p, status, _r in captured)
        assert statuses == [202, 403]

        queued_receipt = next(r for _p, s, r in captured if s == 202)
        rejected_body = next(r for _p, s, r in captured if s == 403)
        assert rejected_body is not None and rejected_body.get("error")
        queued_id = queued_receipt["execution_id"]

        with adapter._profile_scope("alice"):
            allowed_row = delivery_queue.get_status(queued_id)
        assert allowed_row is not None and allowed_row["status"] == "pending"

        # The rejected target (slot 1, "success" lane — this run's model succeeded) never reached
        # admission: no queue row exists under its own operation id either.
        rejected_id = operation_id(str(job["execution_id"]), "success", 1)
        with adapter._profile_scope("alice"):
            assert delivery_queue.get_status(rejected_id) is None

        assert stored_job is not None
        assert stored_job["last_status"] == "delivery_unknown"
        assert stored_job["last_delivery_outcome"] == "unknown"
        assert ledger is not None and ledger["delivery_outcome"] == "unknown"

        with adapter._profile_scope("alice"):
            processed = await _listener_drain(runner, "alice", loop)
        assert processed == 1
        assert len(stub_primary.calls) == 1
        chat_id, _content, _meta = stub_primary.calls[0]
        assert chat_id == "1001"

        with adapter._profile_scope("alice"):
            processed_again = await _listener_drain(runner, "alice", loop)
        assert processed_again == 0
        assert len(stub_primary.calls) == 1
        assert standalone_spy == []
        assert decoy.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cron_key_present", [True, False], ids=["key-present", "key-missing"])
async def test_failure_lane_warning_suppressed_sends_nothing_but_stays_model_error(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch, cron_key_present,
):
    """A real model failure whose failure_deliver target (telegram) has warning notifications
    disabled is a suppressed disposition, never a send: no remote POST, no local send, and the
    run's own outcome stays a model error (never reinterpreted as a successful/delivered run).
    ``deliver="local"`` proves the explicit ``failure_deliver`` override is what gets honored, not
    the success-lane destination.

    A suppressed notice is decided PER TARGET, before any remote transport concern — it must stay
    "suppressed" even when the (never-to-be-used) remote key is missing, never surface as a
    configuration failure. That is the boundary this parametrization pins down: with the key
    present this is the already-established behavior; with the key missing, the global configured
    key precheck in ``_deliver_result`` currently runs BEFORE the per-target suppression check and
    reports a configuration error for a notice that was never going to be sent — the "key-missing"
    id is expected RED until that ordering is fixed."""
    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(
            tmp_path, gateway_url=gateway_url,
            cron_key=_key("alice") if cron_key_present else "")
        config_path = executor_home / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["display"] = {"platforms": {"telegram": {"suppress_warning_notifications": True}}}
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        decoy = _DecoyLocalSend()
        loop = asyncio.get_running_loop()

        job, ok, stored_job, ledger, local_queue_row = await asyncio.to_thread(
            _run_lane_job, executor_home, {Platform.TELEGRAM: decoy}, loop, monkeypatch,
            (False, "doc", "", "synthetic model failure"),
            deliver="local", failure_deliver="telegram:1001", name="suppressed failure lane")

        assert ok is True
        assert captured == [], "no remote POST for a suppressed failure notice"
        assert decoy.calls == []
        assert standalone_spy == []
        assert local_queue_row is None

        assert job.get("_notification_all_targets_suppressed") is True
        assert stored_job is not None
        assert stored_job["last_status"] == "error", "model failure must never be reported as delivered"
        # This scalar is native queue disposition; no queue admission occurred.
        assert stored_job["last_delivery_outcome"] is None
        assert ledger is not None and ledger["delivery_outcome"] == "suppressed", (
            "a suppressed notice must never be reported as a remote configuration failure, "
            "even when the (unused) remote key is missing")


@pytest.mark.asyncio
async def test_failure_lane_warning_enabled_admits_once_and_stays_model_error(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch,
):
    """Positive sibling of the suppression case above, same real model failure and the same
    explicit ``failure_deliver`` override, but warning notifications are left enabled: the failure
    notice is admitted through the configured gateway exactly once, and the run's own outcome is
    still the real model error — only the delivery metadata (queued) differs."""
    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(tmp_path, gateway_url=gateway_url, cron_key=_key("alice"))
        decoy = _DecoyLocalSend()
        loop = asyncio.get_running_loop()

        job, ok, stored_job, ledger, local_queue_row = await asyncio.to_thread(
            _run_lane_job, executor_home, {Platform.TELEGRAM: decoy}, loop, monkeypatch,
            (False, "doc", "", "synthetic model failure"),
            deliver="local", failure_deliver="telegram:1001", name="positive failure lane")

        assert ok is True
        assert decoy.calls == []
        assert standalone_spy == []
        assert local_queue_row is None

        assert len(captured) == 1, "the failure_deliver override, not deliver=local, was admitted"
        _path, status, receipt = captured[0]
        assert status == 202
        remote_id = receipt["execution_id"]

        with adapter._profile_scope("alice"):
            pending = delivery_queue.get_status(remote_id)
        assert pending is not None and pending["status"] == "pending"
        assert job["name"] in pending["content"]

        assert stored_job is not None
        assert stored_job["last_status"] == "error", "model failure stays an error even though delivery admitted"
        assert stored_job["last_delivery_outcome"] == "queued"
        assert ledger is not None and ledger["delivery_outcome"] == "queued"

        with adapter._profile_scope("alice"):
            processed = await _listener_drain(runner, "alice", loop)
        assert processed == 1
        assert len(stub_primary.calls) == 1
        chat_id, _content, _meta = stub_primary.calls[0]
        assert chat_id == "1001"


@pytest.mark.asyncio
async def test_local_artifact_only_target_never_sends_even_with_remote_mode_configured(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch,
):
    """deliver="local" under a configured remote gateway must never reach HTTP or any local
    messaging lane: the real output artifact is still saved to disk and the run's own (successful)
    outcome is reported truthfully, with no delivery outcome stamped at all."""
    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(tmp_path, gateway_url=gateway_url, cron_key=_key("alice"))
        decoy = _DecoyLocalSend()
        loop = asyncio.get_running_loop()

        job, ok, stored_job, ledger, local_queue_row = await asyncio.to_thread(
            _run_lane_job, executor_home, {Platform.TELEGRAM: decoy}, loop, monkeypatch,
            (True, "local-only artifact content", "local-only artifact content", None),
            deliver="local", name="local artifact only")

        assert ok is True
        assert captured == [], "configured remote mode is inert for a local-only target"
        assert decoy.calls == []
        assert standalone_spy == []
        assert local_queue_row is None

        assert stored_job is not None
        assert stored_job["last_status"] == "ok"
        assert not stored_job.get("last_delivery_outcome")

        with _profile_cron_scope(executor_home):
            output_dir = cron_jobs._job_output_dir(job["id"])
            saved_files = list(output_dir.glob("*.md"))
        assert len(saved_files) == 1
        assert "local-only artifact content" in saved_files[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,content",
    [
        ("plain", "bot chat plain content"),
        ("txt-media", "[MEDIA:{canary}] bot chat content"),
        ("extensionless-media", "[MEDIA:{canary}] bot chat extensionless content"),
    ],
    ids=["plain", "media-directive", "extensionless-media"],
)
async def test_bot_chat_only_still_queues_natively_despite_missing_remote_key(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch, case, content,
):
    """deliver="bot-chat" with a configured remote URL but NO usable ``CRON_DELIVERY_KEY``: Bot Chat
    is a local destination that never speaks the remote plaintext protocol, so it must still get a
    real, queued native mailbox receipt regardless of the remote key — with zero HTTP and zero
    messaging send either way. A MEDIA directive in the Bot Chat body changes nothing (Bot Chat
    never extracts/sends media), and the referenced synthetic canary is never opened or read.

    Currently ``_deliver_result``'s "configured but key missing" precheck runs BEFORE the per-target
    loop and returns a configuration error without ever reaching the Bot Chat target — this test is
    expected RED until that ordering is fixed to treat local-only targets independently of remote
    transport config."""
    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(tmp_path, gateway_url=gateway_url, cron_key="")
        canary = executor_home / f"owning-canary-{case}{'' if case == 'extensionless-media' else '.txt'}"
        touched = _canary_touch_guard(monkeypatch, canary)
        decoy = _DecoyLocalSend()
        loop = asyncio.get_running_loop()
        body = content.format(canary=canary)

        db, lease = _setup_bot_chat_live_owner(executor_home)
        try:
            job, ok, stored_job, ledger, local_queue_row = await asyncio.to_thread(
                _run_lane_job, executor_home, {Platform.TELEGRAM: decoy}, loop, monkeypatch,
                (True, "doc", body, None),
                deliver="bot-chat", name="bot chat only, missing remote key")
        finally:
            lease.release()
            db.close()

        assert ok is True
        assert captured == [], "Bot Chat never opens an HTTP connection to the remote protocol"
        assert decoy.calls == []
        assert standalone_spy == []
        assert local_queue_row is None
        assert touched == [], "a MEDIA-directive path in a Bot Chat body must never be opened or read"

        receipts = job.get("_bot_chat_delivery_receipts") or {}
        assert set(receipts) == {"bot-chat:(own)"}, (
            "the Bot Chat target must be admitted independently of the remote key's validity")
        bot_chat_receipt = receipts["bot-chat:(own)"]
        assert bot_chat_receipt["status"] == "queued"
        delivery_id = bot_chat_receipt["delivery_id"]
        assert re.fullmatch(r"[0-9a-f]{64}", delivery_id)

        from tools import bot_live_delivery as mailbox
        mailbox_ticket = mailbox.read_delivery_result(executor_home, delivery_id)
        assert mailbox_ticket is not None and mailbox_ticket["status"] == "queued"
        assert body in mailbox_ticket["message"], "Bot Chat must retain the original media body"

        assert stored_job is not None
        assert stored_job["last_status"] == "delivery_queued"
        # This scalar is native (remote) queue disposition; no remote admission was attempted.
        assert stored_job["last_delivery_outcome"] is None
        assert ledger is not None and ledger["delivery_outcome"] == "queued"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,tag,extension",
    [("upper", "MEDIA", ".txt"), ("lower", "media", ".txt"), ("unicode-case", "medİa", ".txt"),
     ("upper-extensionless", "MEDIA", "")],
    ids=["upper", "lower", "unicode-case", "upper-extensionless"],
)
async def test_remote_only_media_directive_rejected_without_send_or_canary_read(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch, case, tag, extension,
):
    """A remote-only target (no Bot Chat) with a VALID configured key still correctly rejects a
    MEDIA/control directive before ever admitting through the plaintext protocol — this producer
    slice is plaintext-only. The referenced synthetic canary path is never opened or read, no HTTP
    request reaches the real receiver, no local send happens, and the run's own model success is
    reported independently of the delivery rejection (``last_error`` stays unset)."""
    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(tmp_path, gateway_url=gateway_url, cron_key=_key("alice"))
        canary = executor_home / f"owning-canary-{case}{extension}"
        touched = _canary_touch_guard(monkeypatch, canary)
        decoy = _DecoyLocalSend()
        loop = asyncio.get_running_loop()

        job, ok, stored_job, ledger, local_queue_row = await asyncio.to_thread(
            _run_lane_job, executor_home, {Platform.TELEGRAM: decoy}, loop, monkeypatch,
            (True, "doc", f"[{tag}:{canary}] telegram body", None),
            deliver="telegram:1001", name="remote media rejection")

        assert ok is True
        assert captured == [], "no HTTP admission for a rejected media/control directive"
        assert decoy.calls == []
        assert standalone_spy == []
        assert local_queue_row is None
        assert touched == [], "the referenced canary path must never be opened or read"

        assert stored_job is not None
        assert stored_job["last_status"] == "delivery_failed"
        assert stored_job["last_error"] is None, "the model's own run succeeded independently"
        error_text = stored_job.get("last_delivery_error") or ""
        assert "not supported through a configured" in error_text
        assert str(canary) not in error_text, "fixed safe label only — never the raw path"

        assert ledger is not None and ledger["delivery_outcome"] == "failed"


@pytest.mark.asyncio
async def test_mixed_bot_chat_and_remote_target_with_missing_remote_key_keeps_bot_chat_queued(
    adapter, runner, stub_primary, standalone_spy, profiles, tmp_path, monkeypatch,
):
    """deliver="bot-chat,telegram:1001" with a configured remote URL but NO usable
    ``CRON_DELIVERY_KEY``: the local Bot Chat admission must still succeed on its own native store
    (independent of the remote transport's config), the remote target must never reach HTTP or fall
    back to a local send, and the model's own success must be reported separately from the
    definitive pre-remote-transport failure. One accepted local queue plus one definitive
    pre-remote failure aggregates the same conservative way an approved mixed queued/failed
    remote-only fan-out does (see the source-proven-rejection test above): "unknown", never
    collapsed into a blanket "delivered" or "failed".

    Currently the global "configured but key missing" precheck returns before the per-target loop
    ever runs, so Bot Chat is never admitted at all under today's code — this test is expected RED
    until local and remote lanes are decided independently."""
    captured: list = []
    app = _app_with_receipt_capture(adapter, captured)

    async with TestClient(TestServer(app)) as client:
        gateway_url = str(client.make_url("/p/alice/cron/deliveries"))
        executor_home = _executor_home(tmp_path, gateway_url=gateway_url, cron_key="")
        decoy = _DecoyLocalSend()
        loop = asyncio.get_running_loop()

        db, lease = _setup_bot_chat_live_owner(executor_home)
        try:
            job, ok, stored_job, ledger, local_queue_row = await asyncio.to_thread(
                _run_lane_job, executor_home, {Platform.TELEGRAM: decoy}, loop, monkeypatch,
                (True, "doc", "hello mixed missing key", None),
                deliver="bot-chat,telegram:1001", name="mixed fan-out, missing remote key")
        finally:
            lease.release()
            db.close()

        assert ok is True
        assert captured == [], "no HTTP admission is possible without a usable remote key"
        assert decoy.calls == [], "no local fallback send for the remote-configured target"
        assert standalone_spy == []
        assert local_queue_row is None

        receipts = job.get("_bot_chat_delivery_receipts") or {}
        assert set(receipts) == {"bot-chat:(own)"}, (
            "Bot Chat must be admitted independently of the sibling remote target's key problem")
        bot_chat_receipt = receipts["bot-chat:(own)"]
        assert bot_chat_receipt["status"] == "queued"
        delivery_id = bot_chat_receipt["delivery_id"]
        assert re.fullmatch(r"[0-9a-f]{64}", delivery_id)

        from tools import bot_live_delivery as mailbox
        mailbox_ticket = mailbox.read_delivery_result(executor_home, delivery_id)
        assert mailbox_ticket is not None and mailbox_ticket["status"] == "queued"

        assert stored_job is not None
        assert stored_job["last_error"] is None, "the model's own run succeeded independently"
        assert stored_job["last_status"] == "delivery_unknown"
        assert stored_job["last_delivery_outcome"] == "unknown"
        assert ledger is not None and ledger["delivery_outcome"] == "unknown"
