"""Native APIServerAdapter regression driven through the REAL client-side proxy dispatch
(``GatewayRunner._run_agent`` -> ``gateway/run_turn.py::_run_agent_via_proxy``), not a
hand-authored request: the outbound call, the wire, real auth, the real profile-prefix
middleware/home-scoping (both client- and server-side, via the real
``_profile_scope_for_source`` -> ``_resolve_profile_home_for_source`` resolver, keyed on
``source.profile``) and the real ``_create_agent`` all run unmodified. Only the final
``AIAgent`` constructor/execution is a labelled synthetic seam, which captures
``platform``/``gateway_session_key`` plus the exact owning home and session_db instance the
native side resolved for the request, proving the declared per-conversation identity actually
reaches (and stays scoped to) the right owner -- not just that a header round-trips.

Client and native-server profile homes use distinct names (``client-a``/``client-b`` vs.
``owner-a``/``owner-b``) because ``hermes_cli.profiles.get_profile_dir``/``profile_exists`` are
patched process-wide and both the client-side dispatch and the server-side ``/p/<profile>/``
middleware read them.

Complements ``test_proxy_profile_transport.py``: the transport and native construction must retain
the declared memory identity while transcript identifiers rotate independently."""

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest
import yaml
from aiohttp import web

from agent import secret_scope as ss
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    body_limit_middleware,
    cors_middleware,
    security_headers_middleware,
)
from gateway.run import GatewayRunner
from gateway.session import Platform, SessionSource, build_session_key


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class _SyntheticAIAgent:
    """Labelled synthetic seam: only the final ``AIAgent`` constructor/execution is faked. Auth,
    profile-prefix scoping and the real ``_create_agent`` body all run unmodified; the owning
    home and session_db are read from the SAME scope ``_create_agent`` itself resolved."""

    def __init__(self, **kwargs):
        from hermes_constants import get_hermes_home

        self.session_id = kwargs.get("session_id")
        self.platform = kwargs.get("platform")
        self.gateway_session_key = kwargs.get("gateway_session_key")
        self._memory_manager = kwargs.get("memory_manager")
        self.session_db = kwargs.get("session_db")
        self.owning_home = str(get_hermes_home())
        self._stream_delta_callback = kwargs.get("stream_delta_callback")

    def run_conversation(self, *, user_message, conversation_history, task_id, **_kwargs):
        text = f"echo:{self.gateway_session_key or ''}:{user_message}"
        if self._stream_delta_callback:
            self._stream_delta_callback(text)
        return {"final_response": text, "messages": [], "api_calls": 1, "tools": []}


def _patch_deterministic_provider_resolver(monkeypatch):
    """Stand-in for real provider/model resolution so the test needs no credentials or network,
    without touching ``_create_agent`` itself or the real config.yaml loader."""
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"provider": "synthetic-provider", "base_url": "https://example.test/v1", "api_mode": "chat"},
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "hermes-agent")
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model="": None))
    monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_a, **_k: set())


def _profile_home(tmp_path: Path, name: str, *, extra_env: Dict[str, str] | None = None) -> Path:
    """A real profile home: ``.env`` for credentials (read by the real scoped secret resolver on
    both the client-dispatch and native-receiver sides) and a minimal real ``config.yaml`` (read
    by the real, unpatched ``_load_gateway_config``)."""
    home = tmp_path / "profiles" / name
    home.mkdir(parents=True)
    lines = [f"{key}={value}" for key, value in (extra_env or {}).items()]
    (home / ".env").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({}), encoding="utf-8")
    return home


async def _start_shared_native_server(server_runner):
    """One real native listener serving every owner via the real ``/p/<profile>/`` prefix
    middleware -- the exact mechanism ``connect()`` wires, reused here (not a fake policy
    middleware): ``_resolve_request_profile`` -> ``_profile_scope`` -> real
    ``_profile_runtime_scope(get_profile_dir(profile))`` rebinds ``get_hermes_home()`` per
    request."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "synthetic-listener-key"}))
    adapter.gateway_runner = server_runner

    middleware = [
        mw for mw in (
            adapter._make_profile_prefix_middleware(), cors_middleware,
            body_limit_middleware, security_headers_middleware,
        ) if mw is not None
    ]
    app = web.Application(middlewares=middleware)
    app["api_server_adapter"] = adapter
    app.router.add_route("POST", "/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_route("POST", "/p/{profile}/v1/chat/completions", adapter._handle_chat_completions)
    app_runner = web.AppRunner(app)
    await app_runner.setup()
    site = web.TCPSite(app_runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return app_runner, adapter, f"http://127.0.0.1:{port}"


def _client_runner():
    """The real GatewayRunner-side proxy dispatch client. Genuinely multiplexed (unlike the
    single-home wire test's client) because this process has already flipped the global
    multiplex-active guard by starting the native server below: a standalone (non-multiplexed)
    client here would silently fall through to ``_standalone_launch_scope`` and lose whatever
    scope it entered (#112878, see ``gateway/AGENTS.md`` "multiplex_profiles: false is not 'no
    scope ever'")."""
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._run_still_current_fn = lambda *args: (lambda: True)
    return runner


def _source():
    return SessionSource(
        platform=Platform.MATRIX, chat_id="!loopback:memory-identity", user_id="@fixture:loopback",
        user_name="fixture-user", chat_type="group",
    )


@pytest.mark.asyncio
async def test_native_api_server_threads_declared_session_key_to_agent_construction_a_b_a(
    tmp_path, monkeypatch,
):
    """A-B-A driven through the real client-side proxy dispatch against two separately-homed
    native owners: each owner's declared ``X-Hermes-Session-Key`` must reach the real
    ``_create_agent`` -> ``AIAgent`` construction unchanged, scoped to that owner's exact home and
    session_db, isolated from the other owner's transcript id; invalid auth is rejected at the
    native receiver before any construction, with no local fallback."""
    _patch_deterministic_provider_resolver(monkeypatch)
    import run_agent

    constructed: List[_SyntheticAIAgent] = []

    def _construct(**kwargs):
        agent = _SyntheticAIAgent(**kwargs)
        constructed.append(agent)
        return agent

    monkeypatch.setattr(run_agent, "AIAgent", _construct)

    key_owner_a, key_owner_b = "synthetic-owner-a-api-key", "synthetic-owner-b-api-key"
    home_owner_a = _profile_home(tmp_path, "owner-a", extra_env={"API_SERVER_KEY": key_owner_a})
    home_owner_b = _profile_home(tmp_path, "owner-b", extra_env={"API_SERVER_KEY": key_owner_b})

    server_runner = object.__new__(GatewayRunner)
    server_runner.config = GatewayConfig(multiplex_profiles=True)
    server_runner._primary_profile_name = "default"
    app_runner, adapter, base_url = await _start_shared_native_server(server_runner)

    # Real client-side routes: each client profile's OWN .env carries the proxy target for one
    # owner, resolved by the real ``_profile_scope_for_source`` -> ``_run_agent_via_proxy`` chain
    # (never a manually pre-entered scope, which a nested real scope would silently discard).
    home_client_a = _profile_home(
        tmp_path, "client-a",
        extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-a", "GATEWAY_PROXY_KEY": key_owner_a})
    home_client_b = _profile_home(
        tmp_path, "client-b",
        extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-b", "GATEWAY_PROXY_KEY": key_owner_b})
    home_client_a_wrongkey = _profile_home(
        tmp_path, "client-a-wrongkey",
        extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-a", "GATEWAY_PROXY_KEY": "wrong-key-for-owner-a"})
    home_client_a_noauth = _profile_home(
        tmp_path, "client-a-noauth", extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-a"})

    profiles = {
        "owner-a": home_owner_a, "owner-b": home_owner_b,
        "client-a": home_client_a, "client-b": home_client_b,
        "client-a-wrongkey": home_client_a_wrongkey, "client-a-noauth": home_client_a_noauth,
    }
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profiles[name])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in profiles)
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: list(profiles.items()))
    ss.set_multiplex_active(True)

    client_runner = _client_runner()
    source = _source()
    # Distinct native profile-qualified conversation keys for the SAME source -- the memory
    # identity axis, independent of the (shared, fixture-fixed) transcript session id below and
    # independent of the CLIENT-side routing profile (``source.profile``, set per call below).
    key_a = build_session_key(source, profile="owner-a")
    key_b = build_session_key(source, profile="owner-b")
    assert key_a != key_b

    try:
        source.profile = "client-a"
        result_a1 = await client_runner._run_agent(
            "current A1", "fixture system", [], source, "session-fixture", session_key=key_a,
        )
        source.profile = "client-b"
        result_b = await client_runner._run_agent(
            "current B", "fixture system", [], source, "session-fixture", session_key=key_b,
        )
        source.profile = "client-a"
        result_a2 = await client_runner._run_agent(
            "current A2", "fixture system", [], source, "session-fixture-next", session_key=key_a,
        )

        assert len(constructed) == 3
        agent_a1, agent_b, agent_a2 = constructed
        # The real receiver ties the constructed agent to platform=api_server and preserves the
        # gateway session identity resolved from the transport header.
        assert [agent.platform for agent in constructed] == ["api_server"] * 3
        assert [agent.gateway_session_key for agent in constructed] == [key_a, key_b, key_a]
        assert [agent.session_id for agent in constructed] == [
            "session-fixture", "session-fixture", "session-fixture-next",
        ]
        # Real per-owner home scoping (not a fake middleware): the constructor observed
        # get_hermes_home() bound to each owner's OWN profile home.
        assert agent_a1.owning_home == agent_a2.owning_home == str(home_owner_a)
        assert agent_b.owning_home == str(home_owner_b)
        assert agent_a1.owning_home != agent_b.owning_home
        # Real per-owner session_db separation, with the SAME cached instance reused across A1/A2.
        assert agent_a1.session_db is agent_a2.session_db
        assert agent_a1.session_db is not agent_b.session_db
        assert [result["final_response"] for result in (result_a1, result_b, result_a2)] == [
            f"echo:{key_a}:current A1", f"echo:{key_b}:current B", f"echo:{key_a}:current A2",
        ]

        # Wrong auth / no auth against owner-a, through real per-profile scoped credentials (never
        # a manually injected secret): rejected at the native receiver before ``_create_agent``
        # runs at all -- never a silently-omitted identity, never a local inference fallback (the
        # module-level ``run_agent.AIAgent`` patch stays the only construction path; a local
        # fallback would still land here as an unexpected entry).
        source.profile = "client-a-wrongkey"
        wrong_auth_result = await client_runner._run_agent(
            "must reject", "fixture system", [], source, "session-fixture", session_key=key_a,
        )
        assert "401" in wrong_auth_result["final_response"]
        assert len(constructed) == 3

        source.profile = "client-a-noauth"
        no_auth_result = await client_runner._run_agent(
            "must reject too", "fixture system", [], source, "session-fixture", session_key=key_a,
        )
        assert "401" in no_auth_result["final_response"]
        assert len(constructed) == 3
    finally:
        await app_runner.cleanup()
        await adapter.disconnect()
