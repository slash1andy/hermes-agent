"""Behavioral contract for the approved (not yet implemented) authenticated isolated cron
delivery extension: ``POST /p/{profile}/cron/deliveries``.

The extension carries a bounded envelope (execution_id/platform/chat_id/thread_id/content) through
a dedicated per-profile ``CRON_DELIVERY_KEY`` (never the broad ``API_SERVER_KEY``), authorizes the
target against real routing configuration (never the caller's say-so), and durably enqueues before
any send: ingress returns 202 with a receipt, delivery happens later when the native scheduler
drains the queue.

Every enqueue/replay/conflict below goes through the real HTTP ingress (``APIServerAdapter``'s
real profile-prefix middleware, real route table + ``/p/{profile}`` mirrors exactly as
``connect()`` registers them, real per-profile ``.env`` secret scopes on disk). Assertions use
exact target codes, never a union that would silently also accept an unrelated status.

The drain step, where exercised, calls the real ``cron.scheduler.drain_delivery_queue(adapters,
loop)`` (which invokes the real ``_deliver_result``) inside ``asyncio.to_thread``, against the real
guarded adapter map (``GatewayAuthorizationMixin._adapters_for_profile`` / own-adapter profiles,
else ``cron.scheduler_preflight.SharedRouteAdapters`` gated by ``_primary_profile_routes_for_
current_home`` reading an actual root ``config.yaml`` — the exact selection
``cron/scheduler_provider.py::_start_multiplex.tick_adapters_for`` makes). Only the final
live-adapter ``send`` (a real ``gateway.platforms.base.SendResult``) and the real standalone
fallback helper (``tools.send_message_tool._send_to_platform``, spied so no network call is ever
made) are mocked — no custom authorization or idempotency logic runs in this file.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Optional

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import secret_scope as ss
from cron import delivery_queue
from cron.scheduler import drain_delivery_queue
from cron.scheduler_preflight import SharedRouteAdapters, _primary_profile_routes_for_current_home
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner

ENDPOINT = "cron/deliveries"
CRON_KEY_LEN = 32
PRIVATE_BODY_MARKER = "fictional-private-body-marker-7f3c"

# Real routing config (gateway/profile_routing.py), on disk under the synthetic root — read by the
# real ``_primary_profile_routes_for_current_home`` exactly as a live multiplex gateway reads it.
_ROOT_PROFILE_ROUTES = [
    {"name": "alice-1001", "platform": "telegram", "profile": "alice", "chat_id": "1001", "enabled": True},
    {"name": "alice-1002", "platform": "telegram", "profile": "alice", "chat_id": "1002", "enabled": True},
    {"name": "bob-2002", "platform": "telegram", "profile": "bob", "chat_id": "2002", "enabled": True},
]


def _key(label: str) -> str:
    """Deterministic >=32-char per-label credential, distinguishable by its prefix."""
    return f"{label}-cron-delivery-key-xxxxxxxx"[:CRON_KEY_LEN].ljust(CRON_KEY_LEN, "0")


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class _StubSend:
    """Records calls in place of a live platform adapter's ``send`` and returns a real
    ``gateway.platforms.base.SendResult`` — the one integration seam the checkpoint allows to be
    mocked (final network send only; never a fake HTTP server)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Optional[dict]]] = []

    async def send(self, chat_id, content, metadata=None, **_kwargs) -> SendResult:
        self.calls.append((chat_id, content, metadata))
        return SendResult(success=True, message_id=f"stub-{len(self.calls)}")


class _FailingStubSend(_StubSend):
    """Same real ``SendResult`` seam, but the outcome is a native rejection or a raised
    ``TimeoutError`` — still only the final live-adapter double, never a patched consumer."""

    def __init__(self, *, raise_timeout: bool = False, raise_connection: bool = False) -> None:
        super().__init__()
        self._raise_timeout = raise_timeout
        self._raise_connection = raise_connection

    async def send(self, chat_id, content, metadata=None, **_kwargs) -> SendResult:
        self.calls.append((chat_id, content, metadata))
        if self._raise_timeout:
            raise TimeoutError("synthetic timeout")
        if self._raise_connection:
            raise ConnectionError(PRIVATE_BODY_MARKER)
        return SendResult(success=False, error="synthetic")


class _BlockingStubSend(_StubSend):
    def __init__(self, release: asyncio.Event, started: threading.Event, cancelled: threading.Event) -> None:
        super().__init__()
        self._release = release
        self.started = started
        self.cancelled = cancelled

    async def send(self, chat_id, content, metadata=None, **_kwargs) -> SendResult:
        self.calls.append((chat_id, content, metadata))
        self.started.set()
        try:
            await self._release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return SendResult(success=True, message_id=f"stub-{len(self.calls)}")


def _profile_home(tmp_path: Path, name: str, *, cron_key: Optional[str],
                  api_server_key: Optional[str] = None, standalone_telegram: bool = False) -> Path:
    home = tmp_path / "profiles" / name
    home.mkdir(parents=True)
    lines = []
    if cron_key:
        lines.append(f"CRON_DELIVERY_KEY={cron_key}")
    if api_server_key:
        lines.append(f"API_SERVER_KEY={api_server_key}")
    if standalone_telegram:
        # A real (dummy) standalone credential: makes the standalone-fallback lane genuinely
        # reachable for THIS platform, matching ``gateway/config.py``'s "configured/enabled" gate
        # (``cron/scheduler_delivery.py::_resolve_target_transport``) — not an unrelated platform.
        lines.append("TELEGRAM_BOT_TOKEN=dummy-standalone-token-000000")
    (home / ".env").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if standalone_telegram:
        # ``cron.wrap_response: false`` so content-exactness assertions hold once this path
        # actually runs — never alters PRODUCTION wrapping, only this synthetic profile's config.
        (home / "config.yaml").write_text(
            yaml.safe_dump({"platforms": {"telegram": {"enabled": True}},
                            "cron": {"wrap_response": False}}),
            encoding="utf-8")
    # Decoy sibling file: a MEDIA directive in the request body must never cause a transport file
    # read, own or sibling's.
    (home / "own-decoy.jpg").write_bytes(b"\xff\xd8\xff\xe0OWN-DECOY")
    return home


@pytest.fixture
def profiles(tmp_path):
    return {
        "alice": _profile_home(tmp_path, "alice", cron_key=_key("alice"), standalone_telegram=True),
        "bob": _profile_home(tmp_path, "bob", cron_key=_key("bob"), standalone_telegram=True),
        "no_key": _profile_home(tmp_path, "no_key", cron_key=None),
        "default": _profile_home(tmp_path, "default", cron_key=None, api_server_key=_key("default-api")),
    }


@pytest.fixture
def root_home(tmp_path):
    """The primary gateway's own home: owns the real ``profile_routes`` config that
    ``_primary_profile_routes_for_current_home`` reads for every OTHER (satellite) profile."""
    root = tmp_path / "root"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(
        yaml.safe_dump({"profile_routes": _ROOT_PROFILE_ROUTES}), encoding="utf-8")
    return root


@pytest.fixture
def stub_primary():
    return _StubSend()


@pytest.fixture
def standalone_spy(monkeypatch):
    """Spies on the REAL standalone-fallback sender (``_deliver_result``'s "no live adapter, but
    the platform is configured/enabled" lane, ``tools.send_message_tool._send_to_platform``) so a
    denial that should never reach it is provable without ever making a network call. Telegram is
    ``enabled`` in this fixture's config (same platform as the live/satellite lane), so this decoy
    is a REAL reachable fallback path for the target platform — not an unrelated platform's
    credential."""
    calls: list[tuple] = []

    # Matches the real signature exactly (tools/send_message_tool.py::_send_to_platform):
    # ``(platform, pconfig, chat_id, message, thread_id=None, media_files=None, force_document=False,
    # mentions=None, args=None)``; the real caller (``cron/scheduler_delivery.py::_standalone_send``)
    # only ever passes ``thread_id``/``media_files`` as keywords, but ``**_kwargs`` keeps this spy
    # accepting the full signature so a future caller passing more of it is never a TypeError.
    async def _fake_send_to_platform(platform, pconfig, chat_id, content, thread_id=None,
                                     media_files=None, **_kwargs):
        calls.append((platform, chat_id, content))
        return {"success": True}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
    return calls


@pytest.fixture
def runner(profiles, root_home, stub_primary, monkeypatch):
    """A bare ``GatewayRunner`` built the way the codebase's own tests build one for authorization
    unit tests (``object.__new__`` + explicit attributes — see
    ``tests/gateway/test_multiplex_transport_matrix.py``), never a full production start/listener.

    alice/bob are credentialless satellites: no adapter of their own
    (``_profile_adapters[name] == {}``), authorized only through the PRIMARY bot for the exact
    chats the root's ``profile_routes`` grants them (alice: 1001, 1002; bob: 2002) — the real
    mechanism ``cron/scheduler_provider.py::_start_multiplex.tick_adapters_for`` uses for a profile
    with no adapters of its own.
    """
    inst = object.__new__(GatewayRunner)
    inst.config = GatewayConfig(multiplex_profiles=True)
    inst.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    inst._primary_profile_name = "default"
    inst._profile_failed_platforms = {}
    inst.adapters = {Platform.TELEGRAM: stub_primary}
    inst._profile_adapters = {"alice": {}, "bob": {}}

    def _profile_matches_home(name, home=None):
        # Real call sites (e.g. ``_primary_profile_routes_for_current_home``) invoke this with no
        # ``home`` — the production implementation falls back to the CURRENT active home, which
        # here is whichever profile ``APIServerAdapter._profile_scope``/``_profile_runtime_scope``
        # has bound via ``set_hermes_home_override``.
        from hermes_constants import get_hermes_home
        current = Path(home) if home is not None else Path(get_hermes_home())
        return current.name == name

    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root_home)
    monkeypatch.setattr("hermes_cli.profiles.profile_matches_home", _profile_matches_home)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profiles[name])
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: list(profiles.items()),
    )
    return inst


@pytest.fixture
def adapter(runner, profiles):
    """A real ``APIServerAdapter`` wired to ``runner`` — ``_profile_scope`` is the SAME method a
    live multiplexed gateway uses, for both the HTTP layer and the drain step below."""
    inst = APIServerAdapter(PlatformConfig(enabled=True))
    inst._api_key = ""  # the default listener never answers cron deliveries; no broad key here
    inst.gateway_runner = runner
    ss.set_multiplex_active(True)
    return inst


@pytest.fixture
def app(adapter):
    """Mirrors ``connect()``'s own registration (gateway/platforms/api_server.py:4250-4255)
    exactly: every route bare and under ``/p/{profile}``, plus the shared-ingress catch-all."""
    application = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        application.router.add_route(method, path, handler)
        application.router.add_route(method, f"/p/{{profile}}{path}", handler)
    application.router.add_route("*", "/p/{profile}/{tail:.*}", adapter._handle_profile_ingress)
    return application


def _body(**overrides) -> dict:
    payload = {
        "execution_id": "exec-" + uuid.uuid4().hex[:8],
        "platform": "telegram",
        "chat_id": "1001",
        "thread_id": None,
        "content": "hello from cron",
    }
    payload.update(overrides)
    return payload


async def _post(client: TestClient, profile: str, body: dict, key: Optional[str]):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return await client.post(f"/p/{profile}/{ENDPOINT}", json=body, headers=headers)


def _guarded_adapters(runner: GatewayRunner, profile_name: str):
    """Selects the adapter map ``drain_delivery_queue`` is handed for one profile's tick, mirroring
    ``cron/scheduler_provider.py::_start_multiplex.tick_adapters_for`` exactly: the profile's OWN
    adapters when it has any, else the primary adapter gated by the root's ``profile_routes`` for
    this exact profile. Construction only — the authorization DECISION for any given send is made
    by the real ``_deliver_result``/``_resolve_target_transport`` (``isinstance(adapters,
    SharedRouteAdapters)`` + ``.get()``), not here."""
    own = runner._adapters_for_profile(profile_name)
    if own:
        return own
    return SharedRouteAdapters(runner.adapters, _primary_profile_routes_for_current_home())


async def _drain(runner: GatewayRunner, profile_name: str, loop: asyncio.AbstractEventLoop) -> int:
    """Bridges the real, sync ``cron.scheduler.drain_delivery_queue`` (which calls the real
    ``_deliver_result``) onto a background thread, exactly as a live gateway's cron tick does."""
    guarded = _guarded_adapters(runner, profile_name)
    return await asyncio.to_thread(drain_delivery_queue, guarded, loop)


class TestHTTPIngressContract:
    """Every assertion below is the TARGET contract, asserted with its exact status code (never a
    union that would also accept 404-for-absent-route). All of them fail today because
    ``gateway/platforms/api_server.py::_http_route_table`` registers no ``cron/deliveries`` route —
    that is the expected, honest RED signal: nothing here enqueues anything outside the real
    ingress, so these tests should turn green unmodified once the route lands.
    """

    @pytest.mark.asyncio
    async def test_authorized_request_returns_202_pending_zero_sends(self, app, stub_primary):
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "alice", _body(), _key("alice"))
            assert resp.status == 202, (
                "durable enqueue -> receipt is the contract; delivery is a later, explicit drain, "
                f"never a synchronous send on this response (got {resp.status})")
            receipt = await resp.json()
            assert receipt["execution_id"]
            assert stub_primary.calls == []

    @pytest.mark.asyncio
    async def test_execution_id_at_protocol_max_returns_202_pending_zero_sends(self, app, stub_primary):
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "alice", _body(execution_id="x" * 128), _key("alice"))
            assert resp.status == 202
            assert stub_primary.calls == []

    @pytest.mark.asyncio
    async def test_enqueue_drain_replay_terminal_200(self, app, adapter, runner, stub_primary):
        """Positive path end to end: HTTP enqueue (202) -> explicit native drain -> HTTP replay
        resolves terminally (200)."""
        loop = asyncio.get_running_loop()
        async with TestClient(TestServer(app)) as client:
            body = _body()
            enqueued = await _post(client, "alice", body, _key("alice"))
            assert enqueued.status == 202
            assert stub_primary.calls == []

            with adapter._profile_scope("alice"):
                processed = await _drain(runner, "alice", loop)
            assert processed == 1
            assert len(stub_primary.calls) == 1
            chat_id, content, _meta = stub_primary.calls[0]
            assert (chat_id, content) == (body["chat_id"], body["content"])

            replay = await _post(client, "alice", body, _key("alice"))
            assert replay.status == 200
            assert (await replay.json())["execution_id"] == body["execution_id"]
            assert len(stub_primary.calls) == 1  # replay never re-sends

    @pytest.mark.asyncio
    async def test_replay_same_id_changed_payload_is_conflict(self, app, stub_primary):
        async with TestClient(TestServer(app)) as client:
            body = _body()
            await _post(client, "alice", body, _key("alice"))
            conflict = await _post(
                client, "alice", _body(execution_id=body["execution_id"], content="different"), _key("alice"))
            assert conflict.status == 409
            assert stub_primary.calls == []

    @pytest.mark.asyncio
    async def test_concurrent_same_profile_same_execution_id_conflicts_and_sends_winner(
        self, app, adapter, runner, stub_primary,
    ):
        loop = asyncio.get_running_loop()
        execution_id = "exec-concurrent-" + uuid.uuid4().hex[:6]
        valid_a_key = _key("alice")
        payload_a = _body(execution_id=execution_id, content="payload A")
        payload_b = _body(execution_id=execution_id, content="payload B")

        async with TestClient(TestServer(app)) as client:
            results = await asyncio.gather(
                _post(client, "alice", payload_a, valid_a_key),
                _post(client, "alice", payload_b, valid_a_key),
            )
            statuses = sorted(resp.status for resp in results)
            assert statuses == [202, 409]
            accepted_body = payload_a if results[0].status == 202 else payload_b

            with adapter._profile_scope("alice"):
                pending = delivery_queue.get_status(execution_id)
            assert pending is not None and pending["status"] == "pending"
            assert pending["content"] == accepted_body["content"]

            with adapter._profile_scope("alice"):
                assert await _drain(runner, "alice", loop) == 1
            assert len(stub_primary.calls) == 1
            assert stub_primary.calls[0][1] == accepted_body["content"]

    @pytest.mark.asyncio
    async def test_existing_legacy_queue_collision_is_conflict_and_unchanged(
        self, app, adapter, stub_primary,
    ):
        execution_id = "exec-legacy-collision-" + uuid.uuid4().hex[:6]
        legacy_job = {"id": "legacy", "deliver": "telegram:1001"}
        legacy_content = "legacy content"
        body = _body(execution_id=execution_id, content="authenticated content")

        with adapter._profile_scope("alice"):
            delivery_queue.enqueue(execution_id, legacy_job, legacy_content)

        async with TestClient(TestServer(app)) as client:
            response = await _post(client, "alice", body, _key("alice"))
            assert response.status == 409

        with adapter._profile_scope("alice"):
            row = delivery_queue.get_status(execution_id)
        assert row is not None
        assert row["content"] == legacy_content
        assert json.loads(row["job_json"]) == legacy_job
        assert stub_primary.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("adapter_map", ["shared", "raw"], ids=["shared_route_adapters", "raw_runner_adapters"])
    async def test_stale_cached_adapter_map_rechecks_revocation_at_drain(
        self, app, adapter, runner, stub_primary, root_home, standalone_spy, adapter_map,
    ):
        loop = asyncio.get_running_loop()
        body = _body(chat_id="1001", content="stale map")
        async with TestClient(TestServer(app)) as client:
            assert (await _post(client, "alice", body, _key("alice"))).status == 202

            with adapter._profile_scope("alice"):
                cached_adapters = (
                    _guarded_adapters(runner, "alice")
                    if adapter_map == "shared" else runner.adapters
                )

            revoked_routes = [
                dict(route, enabled=(route["profile"] != "alice")) for route in _ROOT_PROFILE_ROUTES
            ]
            (root_home / "config.yaml").write_text(
                yaml.safe_dump({"profile_routes": revoked_routes}), encoding="utf-8")

            with adapter._profile_scope("alice"):
                processed = await asyncio.to_thread(
                    drain_delivery_queue, cached_adapters, loop,
                )
                failed = delivery_queue.get_status(body["execution_id"])
            assert processed == 1
            assert stub_primary.calls == []
            assert standalone_spy == []
            assert failed is not None and failed["status"] == "failed"

    @pytest.mark.asyncio
    async def test_root_secret_does_not_authorize_profile_without_key(
        self, app, adapter, stub_primary, monkeypatch,
    ):
        root_secret = _key("alice")
        monkeypatch.setenv("CRON_DELIVERY_KEY", root_secret)
        async with TestClient(TestServer(app)) as client:
            response = await _post(client, "no_key", _body(), root_secret)
            assert response.status == 401

        with adapter._profile_scope("no_key"):
            assert not delivery_queue.queue_path().exists()
        assert stub_primary.calls == []

    @pytest.mark.asyncio
    async def test_two_profiles_same_execution_id_are_independent(self, app, adapter, runner, stub_primary):
        loop = asyncio.get_running_loop()
        shared_id = "exec-shared-" + uuid.uuid4().hex[:6]
        async with TestClient(TestServer(app)) as client:
            alice_body = _body(execution_id=shared_id, chat_id="1001", content="alice payload")
            bob_body = _body(execution_id=shared_id, chat_id="2002", content="bob payload")

            assert (await _post(client, "alice", alice_body, _key("alice"))).status == 202
            assert (await _post(client, "bob", bob_body, _key("bob"))).status == 202

            with adapter._profile_scope("alice"):
                assert await _drain(runner, "alice", loop) == 1
            with adapter._profile_scope("bob"):
                assert await _drain(runner, "bob", loop) == 1

            assert (await _post(client, "alice", alice_body, _key("alice"))).status == 200
            assert (await _post(client, "bob", bob_body, _key("bob"))).status == 200
        assert len(stub_primary.calls) == 2
        sent_pairs = {(chat, content) for chat, content, _meta in stub_primary.calls}
        assert sent_pairs == {("1001", "alice payload"), ("2002", "bob payload")}

    @pytest.mark.asyncio
    async def test_revocation_after_enqueue_denies_at_drain_no_standalone_fallback(
        self, app, adapter, runner, stub_primary, root_home, standalone_spy,
    ):
        """Revocation here is ROUTE authorization revocation (the root's own config.yaml, read
        fresh on every ``_primary_profile_routes_for_current_home`` call — no caching), not just
        removing the ingress key: HTTP enqueue succeeds under the still-valid route, then the root
        disables alice's route before drain runs. Telegram stays ``enabled`` in config throughout
        (a real reachable standalone-fallback path, spied so no network call is made) and must
        never be used once the live/satellite lane is denied.
        """
        loop = asyncio.get_running_loop()
        async with TestClient(TestServer(app)) as client:
            body = _body(chat_id="1001", content="too late")
            assert (await _post(client, "alice", body, _key("alice"))).status == 202

            revoked_routes = [
                dict(route, enabled=(route["profile"] != "alice")) for route in _ROOT_PROFILE_ROUTES
            ]
            (root_home / "config.yaml").write_text(
                yaml.safe_dump({"profile_routes": revoked_routes}), encoding="utf-8")

            with adapter._profile_scope("alice"):
                processed = await _drain(runner, "alice", loop)
            assert processed == 1
            assert stub_primary.calls == []
            assert standalone_spy == []

            denied_replay = await _post(client, "alice", body, _key("alice"))
            assert denied_replay.status in {401, 403, 409}

    @pytest.mark.asyncio
    async def test_tombstone_retention_replay_and_conflict_through_http(
        self, app, adapter, runner, stub_primary, monkeypatch,
    ):
        """With retention at its smallest bound a delivered row is evicted into
        ``delivery_tombstones`` immediately after drain; a same-id HTTP replay must still resolve
        200 through the tombstone (not silently re-send), and a changed-payload HTTP replay is
        still 409. The direct ``delivery_queue.get_status`` call below is native-queue readback
        used only as supplementary evidence, not as the source of any pass/fail decision — every
        state transition is driven by the real HTTP ingress and the real drain."""
        monkeypatch.setattr(delivery_queue, "MAX_TERMINAL_DELIVERIES", 0)
        loop = asyncio.get_running_loop()
        async with TestClient(TestServer(app)) as client:
            body = _body(execution_id="exec-tombstone", chat_id="1001", content="v1")
            assert (await _post(client, "alice", body, _key("alice"))).status == 202

            with adapter._profile_scope("alice"):
                processed = await _drain(runner, "alice", loop)
                tombstoned = delivery_queue.get_status(body["execution_id"])  # readback as evidence
            assert processed == 1
            assert tombstoned is not None and tombstoned["status"] == "delivered"
            assert len(stub_primary.calls) == 1

            replay = await _post(client, "alice", body, _key("alice"))
            assert replay.status == 200

            conflict = await _post(
                client, "alice",
                _body(execution_id=body["execution_id"], chat_id="1001", content="v2"),
                _key("alice"))
            assert conflict.status == 409
            assert len(stub_primary.calls) == 1  # neither replay re-sends

    @pytest.mark.asyncio
    async def test_fixture_control_standalone_fallback_is_reachable(self, adapter, standalone_spy):
        """Fixture control, not part of the isolated-cron-delivery contract: proves
        ``standalone_spy`` actually intercepts the real fallback lane, using an ORDINARY legitimate
        cron job (``deliver='telegram:9999'``) run through the real, unmodified
        ``cron.scheduler_delivery._deliver_result`` with no live adapters. If this ever fails, the
        revocation test's ``standalone_spy == []`` assertion would be meaningless — the lane could
        be unreachable for reasons unrelated to this extension's authorization, and denial would
        pass for the wrong reason."""
        from cron.scheduler_delivery import _deliver_result

        with adapter._profile_scope("alice"):
            job = {"id": "ctrl-1", "deliver": "telegram:9999"}
            error = await asyncio.to_thread(_deliver_result, job, "control content", None, None)
        assert error is None
        assert len(standalone_spy) == 1
        platform, chat_id, content = standalone_spy[0]
        assert (chat_id, content) == ("9999", "control content")

    @pytest.mark.asyncio
    async def test_missing_key_denied_401(self, app, stub_primary):
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "alice", _body(), None)
            assert resp.status == 401
            assert stub_primary.calls == []

    @pytest.mark.asyncio
    async def test_wrong_profile_key_denied_401(self, app, stub_primary):
        """Bob's key targeting alice's profile prefix must never authorize."""
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "alice", _body(), _key("bob"))
            assert resp.status == 401
            assert stub_primary.calls == []

    @pytest.mark.asyncio
    async def test_api_server_key_rejected_as_delivery_key_401(self, app):
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "default", _body(chat_id="1001"), _key("default-api"))
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_profile_without_delivery_key_denied_401(self, app):
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "no_key", _body(chat_id="1001"), _key("alice"))
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_root_default_route_denied_403(self, app, stub_primary):
        """No ``/p/<profile>/`` prefix: the default profile carries no ``CRON_DELIVERY_KEY`` and
        must never answer this endpoint, even with an otherwise-valid key for a named profile."""
        async with TestClient(TestServer(app)) as client:
            headers = {"Authorization": f"Bearer {_key('alice')}"}
            resp = await client.post(f"/{ENDPOINT}", json=_body(), headers=headers)
            assert resp.status == 403
            assert stub_primary.calls == []

    @pytest.mark.asyncio
    async def test_own_key_wrong_target_denied_403(self, app, stub_primary):
        """A valid key for alice targeting a chat the root's ``profile_routes`` never granted her
        (bob's 2002) must hard-deny: the client's target is input, not identity."""
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "alice", _body(chat_id="2002"), _key("alice"))
            assert resp.status == 403
            assert stub_primary.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("overrides", [
        {"job": {"foo": "bar"}},
        {"script": "curl evil.example"},
        {"session_id": "abc123"},
        {"profile": "bob"},
        {"control": "stop"},
    ], ids=["job", "script", "session_id", "profile_field", "control"])
    async def test_unknown_body_field_denied_400(self, app, stub_primary, overrides):
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "alice", _body(**overrides), _key("alice"))
            assert resp.status == 400
            assert stub_primary.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("overrides", [
        {"execution_id": "has spaces"},
        {"execution_id": "x" * 129},
        {"thread_id": 12345},
        {"content": 12345},
        {"chat_id": "y" * 300},
    ], ids=["exec_id_bad_chars", "exec_id_too_long", "thread_id_not_str", "content_not_str", "chat_id_too_long"])
    async def test_malformed_types_or_grammar_denied_400(self, app, stub_primary, overrides):
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "alice", _body(**overrides), _key("alice"))
            assert resp.status == 400
            assert stub_primary.calls == []

    @pytest.mark.asyncio
    async def test_oversized_content_denied_400(self, app, stub_primary):
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "alice", _body(content="a" * 32769), _key("alice"))
            assert resp.status == 400
            assert stub_primary.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tag", ["MEDIA", "media", "MeDiA", "medİa"])
    async def test_media_directive_denied_before_queueing_no_file_read(
        self, app, stub_primary, adapter, profiles, monkeypatch,
        tag,
    ):
        """The spy is narrowed to the two SUPPLIED decoy paths only (own + sibling) — a generic
        global ``Path.read_bytes`` spy also records legitimate reads the auth/config layer
        performs on unrelated files (profile ``.env``, ``config.yaml``), which is not the thing
        under test and produces a false failure. ``Path.exists``/``Path.open`` are narrowed the
        same way to also catch a validation-only touch (stat/existence check) on the decoys."""
        own_decoy = profiles["alice"] / "own-decoy.jpg"
        sibling_decoy = profiles["bob"] / "own-decoy.jpg"
        decoy_paths = {str(own_decoy), str(sibling_decoy)}
        touched: list[str] = []

        def _guard(name, real):
            def _wrapped(self, *args, **kwargs):
                if str(self) in decoy_paths:
                    touched.append(f"{name}:{self}")
                return real(self, *args, **kwargs)
            return _wrapped

        monkeypatch.setattr(Path, "read_bytes", _guard("read_bytes", Path.read_bytes))
        monkeypatch.setattr(Path, "exists", _guard("exists", Path.exists))
        monkeypatch.setattr(Path, "open", _guard("open", Path.open))

        async with TestClient(TestServer(app)) as client:
            for decoy in (own_decoy, sibling_decoy):
                resp = await _post(
                    client, "alice",
                    _body(content=f"[{tag}:{decoy}]", execution_id="exec-" + uuid.uuid4().hex[:8]),
                    _key("alice"),
                )
                assert resp.status == 400
        assert touched == []
        assert stub_primary.calls == []
        with adapter._profile_scope("alice"):
            assert not delivery_queue.queue_path().exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("overrides", [
        {"platform": " TELEGRAM "},
        {"execution_id": "exec-good\n"},
        {"chat_id": "1001\n"},
        {"content": "\ud800"},
    ], ids=["platform_padded_case", "execution_id_trailing_newline", "chat_id_trailing_newline",
            "content_lone_surrogate"])
    async def test_malformed_literal_values_no_normalization_denied_400(
        self, app, stub_primary, adapter, overrides,
    ):
        """Values are checked LITERALLY, never trimmed/lowercased/repaired first: a padded/cased
        platform, a control-character-bearing execution_id/chat_id, and an unpaired UTF-16
        surrogate in content must all be rejected with exactly 400 — the surrogate case
        specifically must not 500 when the handler measures its UTF-8 byte length."""
        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "alice", _body(**overrides), _key("alice"))
            assert resp.status == 400, f"{overrides!r} -> got {resp.status}"
            assert stub_primary.calls == []
        with adapter._profile_scope("alice"):
            assert not delivery_queue.queue_path().exists()

    @pytest.mark.asyncio
    async def test_chunked_wire_body_trailing_junk_denied_400_not_partial_parse(
        self, app, stub_primary, adapter,
    ):
        """Sends the body as a real chunked transfer (no Content-Length) via a native aiohttp
        async generator: a complete, valid JSON object in the FIRST chunk, a scheduling yield
        (``asyncio.sleep(0)`` — no arbitrary wait), then trailing junk bytes in a SECOND chunk. The
        full wire body is therefore invalid JSON as a whole. A handler that reads only enough bytes
        to parse the first complete JSON value (treating a partial ``StreamReader.read(n)`` as EOF)
        would wrongly accept this; the real contract is 400. ``request.content`` is exercised for
        real, never mocked."""

        async def _chunks():
            yield json.dumps(_body()).encode("utf-8")
            await asyncio.sleep(0)
            yield b"}}}not-json-trailer{{{"

        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                f"/p/alice/{ENDPOINT}", data=_chunks(),
                headers={"Authorization": f"Bearer {_key('alice')}", "Content-Type": "application/json"},
            )
            assert resp.status == 400
        assert stub_primary.calls == []
        with adapter._profile_scope("alice"):
            assert not delivery_queue.queue_path().exists()

    @pytest.mark.asyncio
    async def test_oversized_wire_body_via_chunked_transfer_denied_400(self, app, stub_primary, adapter):
        """A wire body over the endpoint's hard cap must 400 regardless of transport shape —
        chunked, no Content-Length — proving the bound is enforced on the raw stream, not only on
        the decoded ``content`` field."""

        async def _chunks():
            yield json.dumps(_body(content="a" * 70_000)).encode("utf-8")

        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                f"/p/alice/{ENDPOINT}", data=_chunks(),
                headers={"Authorization": f"Bearer {_key('alice')}", "Content-Type": "application/json"},
            )
            assert resp.status == 400
        assert stub_primary.calls == []
        with adapter._profile_scope("alice"):
            assert not delivery_queue.queue_path().exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raise_timeout,raise_connection,expected_status",
        [
            (False, False, "unknown"),
            (True, False, "unknown"),
            (False, True, "unknown"),
        ],
        ids=["untyped_native_rejection", "timeout_raised", "connection_raised"],
    )
    async def test_live_adapter_failure_one_attempt_no_resend_on_replay(
        self, app, adapter, runner, standalone_spy,
        raise_timeout, raise_connection, expected_status, caplog,
    ):
        """A real ``SendResult(success=False, error='synthetic')`` is an untyped native rejection:
        the native boundary converts it to generic ``RuntimeError``, so it conservatively remains
        unknown, like raised ``TimeoutError``/``ConnectionError`` outcomes. All drive the SAME live
        adapter exactly once per drain, through the real, unmodified ``_deliver_result`` — only the
        final adapter double is replaced, never the authorization or send consumer. Failed/uncertain
        live sends never authorize another transport.
        """
        failing = _FailingStubSend(
            raise_timeout=raise_timeout,
            raise_connection=raise_connection,
        )
        runner.adapters[Platform.TELEGRAM] = failing
        loop = asyncio.get_running_loop()
        body = _body(chat_id="1001", content="will fail")
        async with TestClient(TestServer(app)) as client:
            assert (await _post(client, "alice", body, _key("alice"))).status == 202

            with adapter._profile_scope("alice"):
                processed = await _drain(runner, "alice", loop)
                terminal = delivery_queue.get_status(body["execution_id"])
            assert processed == 1
            assert len(failing.calls) == 1
            assert terminal is not None and terminal["status"] == expected_status
            assert standalone_spy == []
            assert PRIVATE_BODY_MARKER not in caplog.text

            replay = await _post(client, "alice", body, _key("alice"))
            assert replay.status == 200
            assert (await replay.json())["status"] == expected_status
            with adapter._profile_scope("alice"):
                assert await _drain(runner, "alice", loop) == 0
        assert len(failing.calls) == 1  # replay and another drain never re-attempt the live send
        assert standalone_spy == []

    @pytest.mark.asyncio
    async def test_started_live_send_cancellation_is_unknown_without_resend(
        self, app, adapter, runner, standalone_spy, monkeypatch,
    ):
        """A started native send that cannot produce confirmation is one unknown attempt."""
        import agent.async_utils as async_utils

        loop = asyncio.get_running_loop()
        release = asyncio.Event()
        send_started = threading.Event()
        send_cancelled = threading.Event()
        blocking = _BlockingStubSend(release, send_started, send_cancelled)
        runner.adapters[Platform.TELEGRAM] = blocking

        real_schedule = async_utils.safe_schedule_threadsafe
        cancel_results = []

        def _schedule(coro, target_loop, *args, **kwargs):
            future = real_schedule(coro, target_loop, *args, **kwargs)
            assert future is not None
            real_result = future.result
            real_cancel = future.cancel

            def _result(timeout=None):
                assert send_started.wait(timeout=5)
                return real_result(timeout=0)

            def _cancel(*cancel_args, **cancel_kwargs):
                result = real_cancel(*cancel_args, **cancel_kwargs)
                cancel_results.append(result)
                return result

            future.result = _result
            future.cancel = _cancel
            return future

        monkeypatch.setattr(async_utils, "safe_schedule_threadsafe", _schedule)
        body = _body(chat_id="1001", content="started then cancelled")

        async with TestClient(TestServer(app)) as client:
            assert (await _post(client, "alice", body, _key("alice"))).status == 202

            with adapter._profile_scope("alice"):
                processed = await _drain(runner, "alice", loop)
                terminal = delivery_queue.get_status(body["execution_id"])
            assert processed == 1
            assert send_started.is_set()
            assert cancel_results and all(value is True for value in cancel_results)
            assert len(blocking.calls) == 1
            assert terminal is not None and terminal["status"] == "unknown"
            assert standalone_spy == []

            replay = await _post(client, "alice", body, _key("alice"))
            assert replay.status == 200
            assert (await replay.json())["status"] == "unknown"
            with adapter._profile_scope("alice"):
                assert await _drain(runner, "alice", loop) == 0
        assert len(blocking.calls) == 1
        assert standalone_spy == []
        try:
            assert await asyncio.to_thread(send_cancelled.wait, 5)
        finally:
            release.set()

    @pytest.mark.asyncio
    async def test_valid_default_profile_key_and_route_still_denied_403(
        self, app, stub_primary, profiles, root_home,
    ):
        """Re-verifies the original spec requirement ('default profile/no prefix denied') holds
        even when the default profile is given its OWN valid dedicated ``CRON_DELIVERY_KEY`` and
        the root grants it the same target chat: successful authentication must never confer
        default-profile authority on this endpoint."""
        default_key = _key("default-cron")
        (profiles["default"] / ".env").write_text(
            f"API_SERVER_KEY={_key('default-api')}\nCRON_DELIVERY_KEY={default_key}\n", encoding="utf-8")
        routes_with_default = _ROOT_PROFILE_ROUTES + [
            {"name": "default-1001", "platform": "telegram", "profile": "default",
             "chat_id": "1001", "enabled": True},
        ]
        (root_home / "config.yaml").write_text(
            yaml.safe_dump({"profile_routes": routes_with_default}), encoding="utf-8")

        async with TestClient(TestServer(app)) as client:
            resp = await _post(client, "default", _body(chat_id="1001"), default_key)
            assert resp.status == 403
        assert stub_primary.calls == []

    @pytest.mark.asyncio
    async def test_misdispatched_queue_row_denied_independent_of_recipient_profile_auth(
        self, app, adapter, runner, stub_primary, root_home, standalone_spy,
    ):
        """Simulated misdispatch: bob's OWN queue receives an exact copy of alice's server-built
        envelope (same execution_id, same job_json, same content — never anything invented by this
        test; copied straight from alice's row). The root also grants bob the same chat, but the
        origin profile binding still denies bob's drain. Alice's native drain of her own original
        row succeeds once.
        """
        loop = asyncio.get_running_loop()
        body = _body(chat_id="1001", content="alice's real envelope")
        async with TestClient(TestServer(app)) as client:
            assert (await _post(client, "alice", body, _key("alice"))).status == 202

        with adapter._profile_scope("alice"):
            alice_row = delivery_queue.get_status(body["execution_id"])
        assert alice_row is not None
        copied_job = json.loads(alice_row["job_json"])

        (root_home / "config.yaml").write_text(
            yaml.safe_dump({"profile_routes": _ROOT_PROFILE_ROUTES + [
                {"name": "bob-1001", "platform": "telegram", "profile": "bob",
                 "chat_id": "1001", "enabled": True},
            ]}), encoding="utf-8")

        with adapter._profile_scope("bob"):
            delivery_queue.enqueue(body["execution_id"], copied_job, alice_row["content"])
            bob_processed = await _drain(runner, "bob", loop)
            bob_terminal = delivery_queue.get_status(body["execution_id"])
        assert bob_processed == 1
        assert bob_terminal is not None and bob_terminal["status"] == "failed"
        assert stub_primary.calls == []
        assert standalone_spy == []

        with adapter._profile_scope("alice"):
            alice_processed = await _drain(runner, "alice", loop)
        assert alice_processed == 1
        assert len(stub_primary.calls) == 1
        assert stub_primary.calls[0][1] == "alice's real envelope"

    @pytest.mark.asyncio
    async def test_mirror_delivery_config_is_inert_for_isolated_envelope(
        self, app, adapter, runner, stub_primary, profiles,
    ):
        """``cron.mirror_delivery: true`` is a real, existing config gate
        (``cron/scheduler_delivery.py::_cron_mirror_delivery_enabled`` /
        ``_target_mirror_eligible``), enabled here on the synthetic profile. It has no effect on a
        marked isolated-delivery envelope: an ``explicit`` platform:chat_id target only mirrors
        when the job itself sets ``attach_to_session: true``, and
        ``cron.isolated_delivery.build_envelope`` never sets it — a real, CURRENT guarantee, so
        this is asserted directly rather than as a future contract. (``mirror_explicit_deliveries``
        was searched for in ``cron/scheduler_delivery.py`` and does not exist as a config key —
        only per-job ``attach_to_session`` and global ``cron.mirror_delivery`` do; not invented
        here.)"""
        config_path = profiles["alice"] / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["cron"] = {**config.get("cron", {}), "mirror_delivery": True}
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        loop = asyncio.get_running_loop()
        body = _body(chat_id="1001", content="no mirror expected")
        async with TestClient(TestServer(app)) as client:
            assert (await _post(client, "alice", body, _key("alice"))).status == 202
            with adapter._profile_scope("alice"):
                processed = await _drain(runner, "alice", loop)
        assert processed == 1
        assert len(stub_primary.calls) == 1  # exactly the plain send — no additional mirror-seed send
        assert not (profiles["alice"] / "state.db").exists()
        assert not (profiles["alice"] / "cron" / "jobs.json").exists()


# ---------------------------------------------------------------------------------------------
# Explicitly uncovered by this file:
#
#   - ProfileRoute discriminators beyond an exact chat_id match: guild-only routes, WhatsApp
#     JID/number/LID equivalence, thread_id-scoped routes, non-satellite (owned-adapter) profiles'
#     own chat-level authorization (today's native code does not scope an OWNED adapter's sends by
#     chat_id at all — only the satellite/``SharedRouteAdapters`` path does).
#   - Real Telegram named-DM-topic creation for a non-numeric thread_id.
#   - CORS / browser-origin interaction with this route.
#   - Rate limiting / max_concurrent_runs interaction.
#   - Actual media transfer once the plain-text slice is followed by the real media-delivery slice.
#   - Legacy SQLite schema migration under a profile's queue path (baseline deliveries/tombstones
#     table built by hand at an old schema version, preserved rows, new HTTP enqueue + legacy-row
#     collision): not attempted here — building a faithful pre-migration schema by hand without
#     reading migration source is a large enough surface to risk a test that asserts the wrong
#     thing; reporting it uncovered rather than guessing.
# ---------------------------------------------------------------------------------------------
