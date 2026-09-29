"""Required-proxy admission for the native API listener's five shared-admission agent routes
(chat completions, responses, session chat, session chat stream, runs) — see
``proxy-api-admission-contract.md``. A process-root ``gateway.proxy_required: true`` (or a
malformed root policy) must refuse EVERY route that would otherwise execute a model turn on this
listener, after authentication but before any pending reservation, session mutation, run claim or
worker/provider setup — while health and authenticated cron-result delivery keep working.

Every test drives the REAL ``APIServerAdapter`` over a real aiohttp ``TestClient``: real route
registration, real ``_check_auth``, real ``gateway.proxy_admission.gateway_proxy_required()``
reading the real process-root ``config.yaml``. The only synthetic seam (positive-path tests only)
is the final ``AIAgent``/provider-resolution boundary inside the real ``_create_agent`` — never a
stubbed ``_create_agent``, ``_run_agent`` or admission gate itself.

Checkpoint scope note: the first pass covered the core route matrix (401 wrong-auth, 403 required,
503 malformed, one representative synthetic-model positive per sync/async lane, health + cron
acceptance). This pass adds the previously-deferred cases: a direct ``_create_agent`` boundary call
bypassing the admission wrapper entirely, the policy-flip-after-admission race on both the sync
chat-completions lane (``_bind_api_server_session`` -> ``_create_agent``) and the async ``/v1/runs``
lane (``_set_run_status("running")`` -> ``_create_agent``), a served-profile-vs-process-root
contradiction proving the root always governs, and broader pending/run-table state-unchanged
assertions on the existing 403 case. All three race/boundary cases document the ACTUAL current
contract (no recheck exists past the single admission-wrapper read), so they are expected to fail
until a production recheck lands — that decision belongs to the controller, not this file.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import hermes_constants
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.gateway.test_api_server_isolated_cron_delivery import _key

pytest_plugins = ("tests.gateway.test_api_server_isolated_cron_delivery",)

_DENIAL_MARKER = "gateway.proxy_required"
_API_KEY = "sk-required-proxy-admission-test-key-0000000000"
_CRON_KEY = "cron-delivery-key-required-proxy-test-0000000"


# ---------------------------------------------------------------------------
# Root policy helpers (process-root config.yaml, matching test_required_proxy_admission.py)
# ---------------------------------------------------------------------------


def _process_root_home():
    return hermes_constants.get_process_hermes_home()


def _write_root_config(*, raw_yaml: "str | None" = None, gateway_section: Any = None) -> None:
    home = _process_root_home()
    config_path = home / "config.yaml"
    if raw_yaml is not None:
        config_path.write_text(raw_yaml, encoding="utf-8")
        return
    config_path.write_text(
        yaml.safe_dump({} if gateway_section is None else {"gateway": gateway_section}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Adapter / app helpers (mirrors tests/gateway/test_api_server.py's own helpers)
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = _API_KEY) -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={"key": api_key} if api_key else {})
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    app.router.add_route("*", "/p/{profile}/{tail:.*}", adapter._handle_profile_ingress)
    return app


# (endpoint, json body) for each of the five shared-admission routes. Session routes address a
# session id that does not exist; admission must refuse BEFORE the 404 session lookup, so the
# route never gets far enough to care.
_ADMISSION_ROUTES = [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/responses", {"input": "hi"}),
    ("/api/sessions/nonexistent/chat", {"message": "hi"}),
    ("/api/sessions/nonexistent/chat/stream", {"message": "hi"}),
    ("/v1/runs", {"input": "hi"}),
]
_ADMISSION_ROUTE_IDS = ["chat-completions", "responses", "session-chat", "session-chat-stream", "runs"]


def _auth_headers(key: Optional[str] = _API_KEY) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}"} if key else {}


# ---------------------------------------------------------------------------
# Synthetic terminal AIAgent / provider seam (positive-path tests only)
# ---------------------------------------------------------------------------


class _TerminalAgent:
    """Stands in for ``run_agent.AIAgent`` — the ONE allowed synthetic seam. Never touches a real
    provider/network; ``run_conversation`` returns an immediately-terminal, unambiguous result."""

    def __init__(self, **kwargs: Any) -> None:
        self.session_prompt_tokens = 1
        self.session_completion_tokens = 1
        self.session_total_tokens = 2
        self.session_id = kwargs.get("session_id") or f"synthetic-{uuid.uuid4().hex[:8]}"
        self.provider = "synthetic"
        self.model = kwargs.get("model") or "synthetic-model"
        self._last_compaction_in_place = False
        self.run_calls: list = []

    def run_conversation(self, **kwargs: Any) -> Dict[str, Any]:
        self.run_calls.append(kwargs)
        return {
            "final_response": "synthetic terminal reply", "messages": [], "api_calls": 1,
            "completed": True, "failed": False, "tools": [],
        }


def _patch_synthetic_runtime(monkeypatch) -> None:
    """Patch ONLY the provider-resolution + final-agent seam that ``_create_agent`` reads at call
    time — everything else in ``_create_agent`` (config loads, toolset resolution, memory
    check-out, process-ownership bookkeeping) runs for real."""
    import gateway.run as gateway_run
    import run_agent

    def _fake_runtime_kwargs() -> dict:
        return {
            "api_key": "synthetic-key", "base_url": "", "provider": "synthetic",
            "requested_provider": None, "api_mode": None, "command": None, "args": [],
            "credential_pool": None, "request_overrides": None,
        }

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _fake_runtime_kwargs)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *a, **k: "synthetic-model")
    monkeypatch.setattr(run_agent, "AIAgent", _TerminalAgent)


@pytest.fixture(autouse=True)
def _clean_root_config():
    # Every test writes its own root config; nothing to reset (isolated per-test HERMES_HOME).
    yield


# ---------------------------------------------------------------------------
# (1) Wrong/missing auth stays 401 regardless of root policy
# ---------------------------------------------------------------------------


class TestWrongAuthStays401:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("endpoint", "payload"), _ADMISSION_ROUTES, ids=_ADMISSION_ROUTE_IDS)
    @pytest.mark.parametrize("gateway_section", [{"proxy_required": True}, {"proxy_required": False}],
                              ids=["required", "not-required"])
    async def test_missing_or_bad_key_is_401(self, monkeypatch, endpoint, payload, gateway_section):
        _write_root_config(gateway_section=gateway_section)
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            missing = await cli.post(endpoint, json=payload)
            assert missing.status == 401
            wrong = await cli.post(endpoint, json=payload, headers=_auth_headers("not-the-key"))
            assert wrong.status == 401


# ---------------------------------------------------------------------------
# (2) proxy_required: true — every route refused (403) before dispatch, fixed safe diagnostic
# ---------------------------------------------------------------------------


class TestRequiredPolicyDeniedBeforeDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("endpoint", "payload"), _ADMISSION_ROUTES, ids=_ADMISSION_ROUTE_IDS)
    async def test_required_true_denies_before_any_dispatch(self, monkeypatch, endpoint, payload):
        _write_root_config(gateway_section={"proxy_required": True})
        adapter = _make_adapter()
        app = _create_app(adapter)

        import gateway.run as gateway_run
        calls = 0
        def _unexpected_runtime_kwargs():
            nonlocal calls
            calls += 1
            raise AssertionError(f"runtime agent resolution must not be reached for {endpoint}")
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _unexpected_runtime_kwargs)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(endpoint, json=payload, headers=_auth_headers())
            body_text = await response.text()

        assert response.status == 403, body_text
        assert _DENIAL_MARKER in body_text
        assert calls == 0
        # No run was ever admitted for the async /v1/runs lane.
        assert adapter._run_statuses == {}
        assert adapter._active_run_tasks == {}
        # No reservation, session/run bookkeeping or live-owner handoff was ever created: denial
        # happens before the reservation counter increments (native queue helper state below).
        assert adapter._pending_agent_requests == 0
        assert adapter._active_run_agents == {}
        assert adapter._run_owners == {}
        assert adapter._shutdown_interruptible_agents == {}


# ---------------------------------------------------------------------------
# (3) Malformed/unreadable root policy — fail closed with 503, never the raw parse error
# ---------------------------------------------------------------------------


class TestMalformedPolicyReturns503:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("endpoint", "payload"), _ADMISSION_ROUTES[:2], ids=_ADMISSION_ROUTE_IDS[:2],
    )
    @pytest.mark.parametrize(
        ("raw_yaml", "gateway_section"),
        [(None, "not-a-mapping"), ("- just\n- a\n- list\n", None)],
        ids=["gateway-not-a-mapping", "root-not-a-mapping"],
    )
    async def test_malformed_root_denies_with_503(
        self, monkeypatch, endpoint, payload, raw_yaml, gateway_section,
    ):
        _write_root_config(raw_yaml=raw_yaml, gateway_section=gateway_section)
        adapter = _make_adapter()
        app = _create_app(adapter)

        import gateway.run as gateway_run
        calls = 0
        def _unexpected_runtime_kwargs():
            nonlocal calls
            calls += 1
            raise AssertionError(f"runtime agent resolution must not be reached for {endpoint}")
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _unexpected_runtime_kwargs)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(endpoint, json=payload, headers=_auth_headers())
            body_text = await response.text()

        assert response.status == 503, body_text
        assert _DENIAL_MARKER in body_text
        assert "not-a-mapping" not in body_text
        assert calls == 0


# ---------------------------------------------------------------------------
# (4) Health stays available; authenticated cron delivery keeps enqueuing despite a required root
# ---------------------------------------------------------------------------


class TestHealthAndCronDeliveryUnaffected:
    @pytest.mark.asyncio
    async def test_health_ok_under_required_root(self):
        _write_root_config(gateway_section={"proxy_required": True})
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.get("/health")
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_authenticated_cron_delivery_enqueues_despite_required_root(
        self, monkeypatch, profiles, root_home, adapter, app,
    ):
        from cron import delivery_queue
        root_home.joinpath("config.yaml").write_text(yaml.safe_dump({
            "gateway": {"proxy_required": True},
            "profile_routes": [
                {"name": "alice-1001", "platform": "telegram", "profile": "alice", "chat_id": "1001", "enabled": True},
                {"name": "bob-2002", "platform": "telegram", "profile": "bob", "chat_id": "2002", "enabled": True},
            ],
        }), encoding="utf-8")
        hermes_constants.pin_process_hermes_home(str(root_home))
        body = {"execution_id": f"exec-{uuid.uuid4().hex[:8]}", "platform": "telegram",
                "chat_id": "1001", "thread_id": None, "content": "hello from cron"}
        try:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post("/p/alice/cron/deliveries", json=body,
                                          headers={"Authorization": f"Bearer {_key('alice')}"})
                body_text = await response.text()
                receipt = yaml.safe_load(body_text)
                operation_id = receipt.get("operation_id", receipt["execution_id"])
            with adapter._profile_scope("alice"):
                queued = delivery_queue.get_status(operation_id)
        finally:
            hermes_constants.pin_process_hermes_home(None)
        assert response.status == 202, body_text
        assert queued["content"] == body["content"]
        assert queued["status"] == "pending"


# ---------------------------------------------------------------------------
# (5) proxy_required absent/false: real _create_agent runs, synthetic terminal model serves the
#     turn — one representative SYNC lane (_run_agent, chat completions) and one ASYNC lane
#     (_execute_run, /v1/runs).
# ---------------------------------------------------------------------------


class TestAbsentOrFalsePolicyReachesRealCreateAgent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("gateway_section", [None, {"proxy_required": False}],
                              ids=["absent", "explicit-false"])
    async def test_sync_chat_completions_reaches_synthetic_terminal_model(self, monkeypatch, gateway_section):
        _write_root_config(gateway_section=gateway_section)
        _patch_synthetic_runtime(monkeypatch)
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]},
                headers=_auth_headers())
            body = await response.json()

        assert response.status == 200, body
        assert body["choices"][0]["message"]["content"] == "synthetic terminal reply"

    @pytest.mark.asyncio
    async def test_async_runs_reaches_synthetic_terminal_model(self, monkeypatch):
        _write_root_config(gateway_section=None)
        _patch_synthetic_runtime(monkeypatch)
        adapter = _make_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post("/v1/runs", json={"input": "hi"}, headers=_auth_headers())
            body = await response.json()
            assert response.status == 202, body
            run_id = body["id"] if "id" in body else body.get("run_id")
            assert run_id
            task = adapter._active_run_tasks.get(run_id)
            if task is not None:
                await task

        status = adapter._run_statuses.get(run_id, {})
        assert status.get("status") == "completed", status
        assert status.get("output") == "synthetic terminal reply"


# ---------------------------------------------------------------------------
# (6) Direct ``_create_agent`` boundary call — bypasses ``_admit_api_agent_request`` entirely.
# Documents whether a caller reaching _create_agent directly (e.g. through a future non-HTTP
# entry point, or the race windows below) still refuses under a denied/malformed root policy.
# Only the downstream runtime-resolution seam is stubbed — never gateway_proxy_required or
# _create_agent itself.
# ---------------------------------------------------------------------------


class TestCreateAgentDirectBoundaryRefusesUnderDeniedRoot:
    @pytest.mark.parametrize(
        ("raw_yaml", "gateway_section"),
        [(None, {"proxy_required": True}), ("- just\n- a\n- list\n", None)],
        ids=["required", "malformed"],
    )
    def test_direct_create_agent_refuses_before_runtime_resolution(self, monkeypatch, raw_yaml, gateway_section):
        _write_root_config(raw_yaml=raw_yaml, gateway_section=gateway_section)
        adapter = _make_adapter()

        import gateway.run as gateway_run
        calls = 0

        def _counted_runtime_kwargs():
            nonlocal calls
            calls += 1
            raise AssertionError("runtime agent resolution reached under a denied/malformed root policy")

        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _counted_runtime_kwargs)

        refused = False
        try:
            adapter._create_agent(session_id=f"direct-{uuid.uuid4().hex[:8]}")
        except Exception:
            refused = True
        assert refused
        assert calls == 0, (
            "a direct _create_agent call must refuse before reaching provider runtime resolution "
            "under a required/malformed root policy — currently _create_agent never rechecks the "
            "policy at all, so this documents the open gap rather than a passing guarantee")


# ---------------------------------------------------------------------------
# (7) Policy-flip-after-admission race, sync lane: root reads false at admission, then flips true
# in the narrow window between the real _bind_api_server_session and the real _create_agent that
# follows it inside _run_agent's worker closure. The wrapper below is passive: it always calls the
# REAL _bind_api_server_session first and only then mutates the actual root config.yaml — the gate
# and the session bind are never faked. The synthetic terminal-agent seam exists ONLY to make the
# reachable old (unsafe) behavior observable rather than crashing on a real provider call.
# ---------------------------------------------------------------------------


class TestPolicyFlipAfterAdmissionRaceChatCompletions:
    @pytest.mark.asyncio
    async def test_root_flips_true_between_bind_and_create_agent_fails_closed(self, monkeypatch):
        _write_root_config(gateway_section=None)
        _patch_synthetic_runtime(monkeypatch)

        import gateway.run as gateway_run
        runtime_calls = 0
        _fake_runtime_kwargs = gateway_run._resolve_runtime_agent_kwargs

        def _counting_runtime_kwargs():
            nonlocal runtime_calls
            runtime_calls += 1
            return _fake_runtime_kwargs()

        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _counting_runtime_kwargs)

        real_bind = APIServerAdapter._bind_api_server_session

        def _flipping_bind(**kwargs):
            result = real_bind(**kwargs)
            _write_root_config(gateway_section={"proxy_required": True})
            return result

        monkeypatch.setattr(APIServerAdapter, "_bind_api_server_session", staticmethod(_flipping_bind))

        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]},
                headers=_auth_headers())
            body = await response.json()

        assert response.status == 502, body
        error = body.get("error", {})
        assert error.get("code") == "agent_incomplete", body
        hermes = error.get("hermes", {})
        assert hermes.get("completed") is False, body
        assert hermes.get("failed") is True, body
        assert _DENIAL_MARKER in (error.get("message") or ""), body
        assert "choices" not in body, body
        assert "synthetic terminal reply" not in str(body), body
        assert runtime_calls == 0, (
            "no provider/agent runtime should be reached once the root policy flips true before "
            "_create_agent — currently _create_agent never rechecks, so this is reachable today")


# ---------------------------------------------------------------------------
# (8) Same race, async lane: root reads false at admission, then flips true right after the real
# _set_run_status("running") transition — recorded by _execute_run immediately before it calls the
# real _create_agent. The wrapper always calls the REAL _set_run_status first.
# ---------------------------------------------------------------------------


class TestPolicyFlipAfterAdmissionRaceRuns:
    @pytest.mark.asyncio
    async def test_root_flips_true_on_running_before_create_agent_fails_closed(self, monkeypatch):
        _write_root_config(gateway_section=None)
        _patch_synthetic_runtime(monkeypatch)

        import gateway.run as gateway_run
        runtime_calls = 0
        _fake_runtime_kwargs = gateway_run._resolve_runtime_agent_kwargs

        def _counting_runtime_kwargs():
            nonlocal runtime_calls
            runtime_calls += 1
            return _fake_runtime_kwargs()

        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _counting_runtime_kwargs)

        real_set_run_status = APIServerAdapter._set_run_status
        flipped = {"done": False}

        def _flipping_set_run_status(self, run_id, status, **fields):
            result = real_set_run_status(self, run_id, status, **fields)
            if status == "running" and not flipped["done"]:
                flipped["done"] = True
                _write_root_config(gateway_section={"proxy_required": True})
            return result

        monkeypatch.setattr(APIServerAdapter, "_set_run_status", _flipping_set_run_status)

        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post("/v1/runs", json={"input": "hi"}, headers=_auth_headers())
            body = await response.json()
            assert response.status == 202, body
            run_id = body["id"] if "id" in body else body.get("run_id")
            task = adapter._active_run_tasks.get(run_id)
            if task is not None:
                await task

        status = adapter._run_statuses.get(run_id, {})
        assert status.get("status") == "failed", status
        assert status.get("completed") is False, status
        assert status.get("failed") is True, status
        assert _DENIAL_MARKER in (status.get("error") or ""), status
        assert runtime_calls == 0, (
            "no provider/agent runtime should be reached once the root policy flips true on the "
            "running transition before _create_agent — currently /v1/runs never rechecks either")


# ---------------------------------------------------------------------------
# (9) Process-root policy vs. a served profile's own contradicting policy, using the real
# multiplex cron fixtures (profiles / root_home / adapter / app) imported from
# test_api_server_isolated_cron_delivery — gateway_proxy_required() clears any served-profile
# secret scope before reading, pinned to the process root only (proxy_admission.py). A served
# profile declaring the opposite policy in its own config.yaml must have zero effect either way.
# ---------------------------------------------------------------------------


class TestRootPolicyGovernsOverServedProfileMultiplex:
    @pytest.mark.asyncio
    async def test_root_required_denies_despite_served_profile_false(
        self, monkeypatch, profiles, root_home, adapter, app,
    ):
        root_home.joinpath("config.yaml").write_text(
            yaml.safe_dump({"gateway": {"proxy_required": True}}), encoding="utf-8")
        alice_key = _key("alice-api")
        alice_env = profiles["alice"] / ".env"
        alice_env.write_text(alice_env.read_text(encoding="utf-8") + f"API_SERVER_KEY={alice_key}\n",
                              encoding="utf-8")
        alice_config = yaml.safe_load((profiles["alice"] / "config.yaml").read_text(encoding="utf-8"))
        alice_config["gateway"] = {"proxy_required": False}
        (profiles["alice"] / "config.yaml").write_text(yaml.safe_dump(alice_config), encoding="utf-8")

        import gateway.run as gateway_run
        calls = 0

        def _unexpected_runtime_kwargs():
            nonlocal calls
            calls += 1
            raise AssertionError("runtime agent resolution must not be reached")

        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _unexpected_runtime_kwargs)

        hermes_constants.pin_process_hermes_home(str(root_home))
        try:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/p/alice/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": f"Bearer {alice_key}"})
                body_text = await response.text()
        finally:
            hermes_constants.pin_process_hermes_home(None)

        assert response.status == 403, body_text
        assert _DENIAL_MARKER in body_text
        assert calls == 0

    @pytest.mark.asyncio
    async def test_root_false_serves_real_turn_despite_served_profile_true(
        self, monkeypatch, profiles, root_home, adapter, app,
    ):
        root_home.joinpath("config.yaml").write_text(
            yaml.safe_dump({"gateway": {"proxy_required": False}}), encoding="utf-8")
        alice_key = _key("alice-api")
        alice_env = profiles["alice"] / ".env"
        alice_env.write_text(alice_env.read_text(encoding="utf-8") + f"API_SERVER_KEY={alice_key}\n",
                              encoding="utf-8")
        alice_config = yaml.safe_load((profiles["alice"] / "config.yaml").read_text(encoding="utf-8"))
        alice_config["gateway"] = {"proxy_required": True}
        (profiles["alice"] / "config.yaml").write_text(yaml.safe_dump(alice_config), encoding="utf-8")

        _patch_synthetic_runtime(monkeypatch)

        hermes_constants.pin_process_hermes_home(str(root_home))
        try:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    "/p/alice/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": f"Bearer {alice_key}"})
                body = await response.json()
        finally:
            hermes_constants.pin_process_hermes_home(None)

        assert response.status == 200, body
        assert body["choices"][0]["message"]["content"] == "synthetic terminal reply"
