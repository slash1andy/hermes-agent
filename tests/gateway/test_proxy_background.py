"""Safety contract for proxy background execution (/bg through ``gateway.proxy_required``).

  - group 1: a required root policy with no usable proxy URL must refuse a background dispatch
    before any local provider/model preparation; absent/explicit-false policy leaves the existing
    local-positive ``/bg`` path unaffected. Driven against a REAL process-root ``config.yaml``.
  - group 2: a background dispatch with a configured, required proxy must reach the real native
    receiver exactly once per task (A/B/A across two owners) and never touch local runtime prep,
    with the foreground session's transcript/pending-images left untouched.

  - group 3: explicit attached-image content forwarded through the real
    ``agent.image_routing.build_native_content_parts`` / native receiver chain (not a hand-built
    request), proving the exact data-URI bytes — not the local path — cross the wire and reach the
    final synthetic ``AIAgent`` construction, with zero local ``_enrich_message_with_vision`` /
    local runtime prep calls and the foreground session untouched.
  - group 4: a compact pre-request media denial matrix (unsupported audio, a missing local image, a
    cross-profile ("sibling") local image, and mismatched ``media_urls``/``media_types`` arrays)
    against a real standalone proxy-required home: each must produce exactly one safe refusal
    delivery with no HTTP request ever issued and no local prep/vision call.
  - group 5: a real loopback HTTP 500, a real partial SSE stream ending without ``[DONE]``, and
    adversarial native-shaped SSE terminal frames (a ``finish_reason: "stop"`` frame carrying
    failure metadata in ``hermes``/``error``, a top-level ``error`` marker alongside a claimed
    ``"stop"``, and a genuine terminal-error frame followed by a later ``"stop"`` overwrite
    attempt) each producing a background delivery that does not claim success, does not invite a
    ``/bg`` retry, does not fall back to a local turn, and never issues a second POST for the same
    task.
"""

from __future__ import annotations

import base64
import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock
from typing import Any, List
from unittest.mock import MagicMock

import pytest
import yaml

import hermes_constants
import gateway.run
from agent import secret_scope as ss
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.hooks import HookRegistry
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource

from tests.gateway.test_proxy_memory_identity import (
    _SyntheticAIAgent,
    _patch_deterministic_provider_resolver,
    _profile_home,
    _start_shared_native_server,
)


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    hermes_constants.pin_process_hermes_home(None)
    yield
    ss.set_multiplex_active(False)
    hermes_constants.pin_process_hermes_home(None)


class _RecordingAdapter(BasePlatformAdapter):
    """A real adapter (never a ``MagicMock`` — a stray truthy attribute would silently pass a
    streaming/typing assertion) recording every delivery-facing call."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent: List[str] = []
        self.typing_calls = 0
        self.raise_after_recording = False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        if self.raise_after_recording:
            raise RuntimeError("send-private-marker")
        return SendResult(success=True, message_id=f"m{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing_calls += 1

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _make_background_runner() -> GatewayRunner:
    """Bare ``GatewayRunner`` exercising the REAL ``_run_background_task`` entry point (mirrors
    ``test_background_command.py::_make_runner``, plus the proxy-dispatch attrs
    ``test_proxy_mode.py`` / ``test_proxy_memory_identity.py`` need)."""
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._session_run_generation = {}
    runner._run_still_current_fn = lambda *a, **k: (lambda: True)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    mock_store = MagicMock()
    mock_store.get_model_override.return_value = None
    runner.session_store = mock_store
    runner.hooks = HookRegistry()
    return runner


def _source(platform=Platform.TELEGRAM, **overrides) -> SessionSource:
    kwargs = dict(platform=platform, chat_id="chat-1", user_id="user-1", user_name="tester")
    kwargs.update(overrides)
    return SessionSource(**kwargs)


# ── group 1 ── root-required policy admission for /bg ──────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gateway_section,expect_local",
    [
        (None, True),
        ({"proxy_required": False}, True),
        ({"proxy_required": True}, False),
        ({"proxy_required": True, "proxy_url": ""}, False),
    ],
    ids=["absent", "explicit-false", "required-missing-url", "required-empty-url"],
)
async def test_background_admission_policy(monkeypatch, tmp_path, gateway_section, expect_local):
    """A required root policy with no usable proxy URL must refuse before any local runtime-kwargs
    resolution; absent/explicit-false policy leaves the existing local-positive path unaffected.
    Driven against a REAL process-root ``config.yaml`` (never a patch of ``gateway_proxy_required``
    or the native dispatcher)."""
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    raw_config = {"gateway": gateway_section} if gateway_section is not None else {}
    root_home = tmp_path / "admission-root"
    root_home.mkdir()
    if raw_config:
        (root_home / "config.yaml").write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    hermes_constants.pin_process_hermes_home(str(root_home))

    runtime_kwargs_calls: List[Any] = []

    def _resolve_runtime_agent_kwargs():
        runtime_kwargs_calls.append(True)
        return {"api_key": "test-key"}

    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", _resolve_runtime_agent_kwargs)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: raw_config)

    runner = _make_background_runner()
    adapter = _RecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    source = _source()

    constructed: List[Any] = []

    def _construct(**kwargs):
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "local turn ran", "messages": []}
        constructed.append(agent)
        return agent

    monkeypatch.setattr("run_agent.AIAgent", _construct)

    await runner._run_background_task("hello", source, "bg_admission_test")

    if expect_local:
        assert len(runtime_kwargs_calls) == 1
        assert len(constructed) == 1
        assert adapter.sent
        assert "local turn ran" in adapter.sent[0]
    else:
        assert len(runtime_kwargs_calls) == 0, "required policy with no usable proxy must not reach local prep"
        assert len(constructed) == 0, "required policy with no usable proxy must not reach local prep"
        assert len(adapter.sent) == 1
        assert "gateway.proxy_required" in adapter.sent[0]
        assert "✅ Background task complete" not in adapter.sent[0]


# ── group 2 ── real native authenticated loopback A->B->A ──────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("native_outcome", ["success", "native-agent-failure"], ids=["success", "native-agent-failure"])
@pytest.mark.parametrize("send_raises", [False, True], ids=["send-ok", "send-raises-after-recording"])
async def test_background_dispatches_via_real_native_proxy_a_b_a(monkeypatch, tmp_path, native_outcome, send_raises):
    """A configured, required proxy must route each background dispatch to the real native
    receiver (owner-a/b native homes, client-a/b dispatch homes, real
    ``profiles_to_serve``/``get_profile_dir`` and ``source.profile`` switching, process-root
    ``gateway.proxy_required: true`` pinned on client-a, reused from
    ``test_proxy_memory_identity.py``), never local runtime prep, and must leave the foreground
    session's transcript and pending-image buffer untouched."""
    _patch_deterministic_provider_resolver(monkeypatch)
    # The shared resolver stub's signature (no args) doesn't match how the background/turn path
    # actually calls it (`_resolve_gateway_model(user_config)`); fixed here so the real seam is
    # exercised rather than an unrelated TypeError from a mismatched fixture lambda.
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda user_config=None: "hermes-agent")
    import run_agent

    constructed: List[_SyntheticAIAgent] = []

    def _construct(**kwargs):
        agent = _SyntheticAIAgent(**kwargs)
        if native_outcome == "native-agent-failure":
            def _failed_run(*, user_message, task_id, **_kwargs):
                return {
                    "final_response": f"partial:{user_message}",
                    "messages": [],
                    "completed": False,
                    "failed": True,
                    "error": "private-marker",
                }

            agent.run_conversation = _failed_run
        constructed.append(agent)
        return agent

    monkeypatch.setattr(run_agent, "AIAgent", _construct)

    local_prep_calls: List[dict] = []

    def _local_prep_sentinel(self, **kwargs):
        local_prep_calls.append(kwargs)
        raise AssertionError("local runtime prep must not run for a proxy-configured background task")

    monkeypatch.setattr(GatewayRunner, "_resolve_session_agent_runtime", _local_prep_sentinel)

    key_owner_a, key_owner_b = "synthetic-owner-a-key", "synthetic-owner-b-key"
    home_owner_a = _profile_home(tmp_path, "owner-a", extra_env={"API_SERVER_KEY": key_owner_a})
    home_owner_b = _profile_home(tmp_path, "owner-b", extra_env={"API_SERVER_KEY": key_owner_b})

    server_runner = object.__new__(GatewayRunner)
    server_runner.config = GatewayConfig(multiplex_profiles=True)
    server_runner._primary_profile_name = "default"
    app_runner, native_adapter, base_url = await _start_shared_native_server(server_runner)

    home_client_a = _profile_home(
        tmp_path, "client-a",
        extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-a", "GATEWAY_PROXY_KEY": key_owner_a})
    home_client_b = _profile_home(
        tmp_path, "client-b",
        extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-b", "GATEWAY_PROXY_KEY": key_owner_b})

    # This process's OWN launch config (pinned, real root config.yaml, not a mocked reader)
    # declares the proxy required.
    (home_client_a / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"proxy_required": True}}), encoding="utf-8")
    hermes_constants.pin_process_hermes_home(str(home_client_a))

    profiles = {
        "owner-a": home_owner_a, "owner-b": home_owner_b,
        "client-a": home_client_a, "client-b": home_client_b,
    }
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profiles[name])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in profiles)
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: list(profiles.items()))
    ss.set_multiplex_active(True)

    runner = _make_background_runner()
    runner.config = GatewayConfig(multiplex_profiles=True)
    adapter = _RecordingAdapter(platform=Platform.MATRIX)
    adapter.raise_after_recording = send_raises
    runner.adapters[Platform.MATRIX] = adapter
    runner._profile_adapters["client-a"] = {Platform.MATRIX: adapter}
    runner._profile_adapters["client-b"] = {Platform.MATRIX: adapter}
    source = _source(platform=Platform.MATRIX)

    # Seeded foreground state a background task must never consume/mutate.
    _foreground_key = "foreground-session-key-untouched"
    runner._pending_images = {_foreground_key: ["seeded-image.png"]}
    _pending_images_snapshot = dict(runner._pending_images)
    runner.session_store.load_transcript = MagicMock(return_value=[{"role": "user", "content": "seeded history"}])

    # Observe the real native result translation without replacing the API route or its response.
    from gateway.platforms import api_server_openai_routes as native_routes
    result_flags = Mock(wraps=native_routes._result_flags)
    finish_reason = Mock(wraps=native_routes._finish_reason)
    hermes_extras = Mock(wraps=native_routes._hermes_extras)
    monkeypatch.setattr(native_routes, "_result_flags", result_flags)
    monkeypatch.setattr(native_routes, "_finish_reason", finish_reason)
    monkeypatch.setattr(native_routes, "_hermes_extras", hermes_extras)

    try:
        source.profile = "client-a"
        await runner._run_background_task("first background task", source, "bg_task_a")
        source.profile = "client-b"
        await runner._run_background_task("second background task", source, "bg_task_b")
        source.profile = "client-a"
        await runner._run_background_task("third background task", source, "bg_task_a_2")

        # -- local runtime prep must never run for a proxy-configured background dispatch --
        assert local_prep_calls == []
        assert len(constructed) == 3, (
            "each background dispatch must reach the real native receiver exactly once"
        )
        assert [agent.platform for agent in constructed] == ["api_server"] * 3
        assert [agent.session_id for agent in constructed] == ["bg_task_a", "bg_task_b", "bg_task_a_2"]
        agent_a1, agent_b, agent_a2 = constructed
        assert agent_a1.owning_home == agent_a2.owning_home == str(home_owner_a)
        assert agent_b.owning_home == str(home_owner_b)
        assert agent_a1.owning_home != agent_b.owning_home
        assert agent_a1.session_db is agent_a2.session_db
        assert agent_a1.session_db is not agent_b.session_db
        # -- no foreground session-key / memory-identity leakage into an isolated bg task --
        assert all(agent.gateway_session_key is None for agent in constructed)

        # -- foreground pending-images/history untouched by any background dispatch --
        assert runner._pending_images == _pending_images_snapshot
        runner.session_store.load_transcript.assert_not_called()

        # -- exactly one final delivery per task, no typing, no interim stream sends --
        assert adapter.typing_calls == 0
        assert len(adapter.sent) == 3, "a final send failure must not trigger a second delivery"
        if native_outcome == "success":
            for content in adapter.sent:
                assert "✅ Background task complete" in content
            assert len(result_flags.call_args_list) == 3
            assert all(call.args[0].get("completed", True) is True
                       and not call.args[0].get("partial") and not call.args[0].get("failed")
                       and call.args[0].get("error") is None for call in result_flags.call_args_list)
            assert len(finish_reason.call_args_list) == 3
            assert all(call.args[:3] == (True, False, False) for call in finish_reason.call_args_list)
            assert hermes_extras.call_count == 0
        else:
            assert all("✅ Background task complete" not in content for content in adapter.sent)
            assert all("private-marker" not in content for content in adapter.sent)
            assert all("/bg" not in content and "/agents" not in content for content in adapter.sent)
            assert len(result_flags.call_args_list) == 3
            assert all(call.args[0].get("completed") is False
                       and call.args[0].get("failed") is True
                       and call.args[0].get("error") == "private-marker"
                       for call in result_flags.call_args_list)
            assert len(finish_reason.call_args_list) == 3
            assert all(call.args[:4] == (False, False, True, "private-marker")
                       for call in finish_reason.call_args_list)
            assert len(hermes_extras.call_args_list) == 3
            assert all(call.args[:4] == (False, False, True, "private-marker") and call.args[4] == "error"
                       for call in hermes_extras.call_args_list)
    finally:
        await app_runner.cleanup()
        await native_adapter.disconnect()


# ── group 3 ── real native image-content forwarding (positive receiver) ────────────────────────


_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _ImageObservingAIAgent:
    """Labelled synthetic seam identical in shape to ``_SyntheticAIAgent`` (only the final
    ``AIAgent`` construction/execution is faked) but also records the exact ``user_message`` the
    real native receiver resolved, to prove an attached image survived end-to-end (not just that
    the outbound request carried it)."""

    def __init__(self, **kwargs):
        from hermes_constants import get_hermes_home

        self.session_id = kwargs.get("session_id")
        self.platform = kwargs.get("platform")
        self.owning_home = str(get_hermes_home())
        self.received_user_message: Any = None

    def run_conversation(self, *, user_message, task_id, **_kwargs):
        self.received_user_message = user_message
        return {"final_response": "image turn ran", "messages": [], "completed": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("native_builder_fault", [False, True], ids=["native-image-positive", "native-builder-failure"])
async def test_background_dispatch_forwards_native_image_content_through_real_proxy(
    monkeypatch, tmp_path, native_builder_fault
):
    """A background dispatch with an attached local image must forward the REAL native content
    parts (``agent.image_routing.build_native_content_parts``'s base64 data URI, never the local
    file path) through the real client dispatch, wire, and native receiver -- only the final
    ``AIAgent`` construction is a labelled synthetic seam. No local vision enrichment or local
    runtime prep may run, and the foreground session's images/history stay untouched."""
    _patch_deterministic_provider_resolver(monkeypatch)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda user_config=None: "hermes-agent")
    import run_agent

    constructed: List[_ImageObservingAIAgent] = []

    def _construct(**kwargs):
        agent = _ImageObservingAIAgent(**kwargs)
        constructed.append(agent)
        return agent

    monkeypatch.setattr(run_agent, "AIAgent", _construct)

    if native_builder_fault:
        def _named_native_builder_failure(*_args, **_kwargs):
            raise RuntimeError("private-marker")

        monkeypatch.setattr("agent.image_routing.build_native_content_parts", _named_native_builder_failure)

    async def _forbidden_vision(self, *_a, **_k):
        raise AssertionError("local vision enrichment must not run for a proxy-configured background task")

    def _forbidden_prep(self, **_kwargs):
        raise AssertionError("local runtime prep must not run for a proxy-configured background task")

    monkeypatch.setattr(GatewayRunner, "_enrich_message_with_vision", _forbidden_vision)
    monkeypatch.setattr(GatewayRunner, "_resolve_session_agent_runtime", _forbidden_prep)

    key_owner_a = "synthetic-owner-image-key"
    home_owner_a = _profile_home(tmp_path, "owner-image", extra_env={"API_SERVER_KEY": key_owner_a})

    server_runner = object.__new__(GatewayRunner)
    server_runner.config = GatewayConfig(multiplex_profiles=True)
    server_runner._primary_profile_name = "default"
    app_runner, native_adapter, base_url = await _start_shared_native_server(server_runner)

    home_client_a = _profile_home(
        tmp_path, "client-image",
        extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-image", "GATEWAY_PROXY_KEY": key_owner_a})
    (home_client_a / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"proxy_required": True}}), encoding="utf-8")
    hermes_constants.pin_process_hermes_home(str(home_client_a))

    # The client's OWN image_cache -- the real on-disk location ``get_image_cache_dir()`` resolves
    # to under this home (``cache/images/``), never a scratch tmp_path location.
    image_path = home_client_a / "cache" / "images" / "img_native_test.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(_TINY_PNG)
    expected_data_url = f"data:image/png;base64,{base64.b64encode(_TINY_PNG).decode('ascii')}"

    profiles = {"owner-image": home_owner_a, "client-image": home_client_a}
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profiles[name])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in profiles)
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: list(profiles.items()))
    ss.set_multiplex_active(True)

    runner = _make_background_runner()
    runner.config = GatewayConfig(multiplex_profiles=True)
    adapter = _RecordingAdapter(platform=Platform.MATRIX)
    runner.adapters[Platform.MATRIX] = adapter
    runner._profile_adapters["client-image"] = {Platform.MATRIX: adapter}
    source = _source(platform=Platform.MATRIX)
    source.profile = "client-image"

    # Seeded foreground state a background image dispatch must never consume/mutate.
    runner._pending_images = {"foreground-key": ["seeded-image.png"]}
    _pending_images_snapshot = dict(runner._pending_images)
    runner.session_store.load_transcript = MagicMock(return_value=[{"role": "user", "content": "seeded history"}])

    # Passively observe the real outbound wire JSON -- the original ``ClientSession.post`` still
    # runs, this only records the body it was called with.
    import aiohttp

    original_post = aiohttp.ClientSession.post
    captured_bodies: List[Any] = []

    def _observing_post(self, url, **kwargs):
        captured_bodies.append(kwargs.get("json"))
        return original_post(self, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "post", _observing_post)

    try:
        started = await runner._handle_background_command(MessageEvent(
            text="/bg describe this image", source=source, message_id="bg-command-image",
            media_urls=[str(image_path)], media_types=["image/png"],
        ))
        match = re.search(r"bg_\d{6}_[0-9a-f]{6}", started)
        assert match, f"background acknowledgement must expose its generated task id: {started!r}"
        tasks = tuple(runner._background_tasks)
        assert tasks, "the real /bg handler must track the spawned task"
        await asyncio.gather(*tasks)

        if native_builder_fault:
            assert constructed == []
            assert captured_bodies == []
            assert len(adapter.sent) == 1
            assert "✅ Background task complete" not in adapter.sent[0]
            assert "private-marker" not in adapter.sent[0]
        else:
            assert len(constructed) == 1, "the real native receiver must construct exactly one agent"
            assert constructed[0].platform == "api_server"
            assert constructed[0].owning_home == str(home_owner_a)
            assert constructed[0].session_id == match.group(0)
            assert len(captured_bodies) == 1
            sent_content = captured_bodies[0]["messages"][-1]["content"]
            assert isinstance(sent_content, list), "an attached image must forward as native content parts"
            sent_image_urls = [
                p["image_url"]["url"] for p in sent_content
                if isinstance(p, dict) and p.get("type") == "image_url"
            ]
            assert sent_image_urls == [expected_data_url]
            assert str(image_path) not in json.dumps(sent_content), (
                "the wire body must carry the encoded image bytes, never the local file path"
            )

            received = constructed[0].received_user_message
            assert isinstance(received, list), "the native receiver must retain the image, not just the caption text"
            received_image_urls = [
                p["image_url"]["url"] for p in received
                if isinstance(p, dict) and p.get("type") == "image_url"
            ]
            assert received_image_urls == [expected_data_url]

            assert adapter.sent and "✅ Background task complete" in adapter.sent[0]
        assert len(adapter.sent) == 1
        assert runner._pending_images == _pending_images_snapshot
        runner.session_store.load_transcript.assert_not_called()
    finally:
        await app_runner.cleanup()
        await native_adapter.disconnect()


# ── group 4 ── pre-request media denial matrix ──────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label",
    ["unsupported-audio", "missing-image", "sibling-profile-image", "inconsistent-arrays"],
)
async def test_background_media_denial_matrix(monkeypatch, tmp_path, label):
    """Every profile from ``_profile_home`` is already a canonical ``tmp/profiles/<name>`` real
    home. A pre-request media problem must refuse before any HTTP call and before any local
    prep/vision call -- exactly one safe refusal delivery, never a silently-dropped attachment and
    never a claimed completion."""
    root = _profile_home(tmp_path, "standalone-media")
    (root / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"proxy_required": True, "proxy_url": "http://127.0.0.1:1/unused"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(gateway.run, "_hermes_home", root)
    hermes_constants.pin_process_hermes_home(str(root))

    own_image = root / "cache" / "images" / "own.png"
    own_image.parent.mkdir(parents=True, exist_ok=True)
    own_image.write_bytes(_TINY_PNG)

    if label == "unsupported-audio":
        audio_path = root / "cache" / "audio" / "clip.ogg"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"OggS\x00fake-audio-bytes")
        media_urls, media_types = [str(audio_path)], ["audio/ogg; private-marker"]
    elif label == "missing-image":
        media_urls, media_types = [str(root / "cache" / "images" / "does-not-exist.png")], ["image/png"]
    elif label == "sibling-profile-image":
        sibling = _profile_home(tmp_path, "sibling-media")
        sibling_image = sibling / "cache" / "images" / "sibling.png"
        sibling_image.parent.mkdir(parents=True, exist_ok=True)
        sibling_image.write_bytes(_TINY_PNG)
        media_urls, media_types = [str(sibling_image)], ["image/png"]
    else:  # inconsistent-arrays -- two urls, one declared type
        media_urls, media_types = [str(own_image), str(own_image)], ["image/png"]

    import aiohttp

    post_calls: List[Any] = []

    def _tracking_post(self, url, **kwargs):
        post_calls.append(kwargs.get("json"))
        raise RuntimeError("HTTP must not be attempted for a pre-request-denied media attachment")

    monkeypatch.setattr(aiohttp.ClientSession, "post", _tracking_post)

    async def _forbidden_vision(self, *_a, **_k):
        raise AssertionError("local vision enrichment must not run for a proxy-configured background task")

    def _forbidden_prep(self, **_kwargs):
        raise AssertionError("local runtime prep must not run for a proxy-configured background task")

    monkeypatch.setattr(GatewayRunner, "_enrich_message_with_vision", _forbidden_vision)
    monkeypatch.setattr(GatewayRunner, "_resolve_session_agent_runtime", _forbidden_prep)

    runner = _make_background_runner()
    runner.config = SimpleNamespace(multiplex_profiles=False)
    adapter = _RecordingAdapter(platform=Platform.MATRIX)
    runner.adapters[Platform.MATRIX] = adapter
    source = _source(platform=Platform.MATRIX)

    await runner._run_background_task(
        "please look at this attachment", source, f"bg_media_denied_{label}",
        media_urls=media_urls, media_types=media_types,
    )

    assert post_calls == [], "no HTTP request must be issued for a denied media attachment"
    assert len(adapter.sent) == 1, "exactly one safe refusal delivery, no silent drop, no retry"
    notification = adapter.sent[0]
    assert "✅ Background task complete" not in notification
    assert "private-marker" not in notification


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "http-500", "sse-eof", "stop-with-hermes-error", "top-level-error-with-stop",
        "error-then-stop-overwrite", "hermes-non-dict", "completed-zero", "malformed-choices",
    ],
    ids=[
        "http-500", "sse-eof", "stop-with-hermes-error", "top-level-error-with-stop",
        "error-then-stop-overwrite", "hermes-non-dict", "completed-zero", "malformed-choices",
    ],
)
async def test_background_proxy_transport_fault_is_one_non_success_delivery(monkeypatch, tmp_path, fault):
    """Transport faults stay failures at the real background boundary: no local execution, retry,
    success claim, or private upstream diagnostic leaks into the one final delivery."""
    from aiohttp import web

    private_marker = "private-marker"
    requests = []

    async def _fault_handler(request):
        requests.append(await request.json())
        if fault == "http-500":
            return web.Response(status=500, text=private_marker)
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"partial visible"}}]}\n\n')
        if fault == "sse-eof":
            return response
        if fault == "stop-with-hermes-error":
            # Native-shaped adversarial terminal frame: claims "stop" while the same frame's
            # ``hermes``/``error`` extras say the run actually failed -- the parser must not
            # ignore the failure metadata just because finish_reason reads "stop".
            final = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "error": {"message": private_marker, "type": "agent_error"},
                "hermes": {
                    "completed": False, "partial": False, "failed": True,
                    "error": private_marker, "error_code": "agent_error",
                },
            }
            await response.write(f"data: {json.dumps(final)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response
        if fault == "top-level-error-with-stop":
            # A top-level ``error`` marker alongside a claimed "stop": the marker alone must be
            # enough to refuse, without needing the ``hermes`` extras block too.
            final = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "error": {"message": private_marker, "type": "agent_error"},
            }
            await response.write(f"data: {json.dumps(final)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response
        if fault == "error-then-stop-overwrite":
            # A genuine terminal-error frame followed by a later "stop" frame attempting to
            # overwrite it: a later frame must never erase an earlier terminal failure signal.
            error_chunk = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                "error": {"message": private_marker, "type": "agent_error"},
                "hermes": {
                    "completed": False, "partial": False, "failed": True,
                    "error": private_marker, "error_code": "agent_error",
                },
            }
            await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
            overwrite_chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            await response.write(f"data: {json.dumps(overwrite_chunk)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response
        if fault == "hermes-non-dict":
            final = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "hermes": private_marker,
            }
            await response.write(f"data: {json.dumps(final)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response
        if fault == "completed-zero":
            final = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "hermes": {"completed": 0, "partial": False, "failed": False},
            }
            await response.write(f"data: {json.dumps(final)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response
        if fault == "malformed-choices":
            error_chunk = {
                "choices": [None],
                "error": {"message": private_marker, "type": "agent_error"},
            }
            await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
            await response.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n')
            await response.write(b"data: [DONE]\n\n")
            return response
        raise AssertionError(f"unhandled fault id: {fault}")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", _fault_handler)
    app_runner = web.AppRunner(app)
    await app_runner.setup()
    site = web.TCPSite(app_runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    root = _profile_home(tmp_path, "standalone")
    (root / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"proxy_required": True, "proxy_url": f"http://127.0.0.1:{port}"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(gateway.run, "_hermes_home", root)
    hermes_constants.pin_process_hermes_home(str(root))

    runner = _make_background_runner()
    runner.config = SimpleNamespace(multiplex_profiles=False)
    adapter = _RecordingAdapter(platform=Platform.MATRIX)
    runner.adapters[Platform.MATRIX] = adapter
    source = _source(platform=Platform.MATRIX)
    runner._pending_images = {"foreground": ["keep-me.png"]}
    pending_snapshot = {key: list(value) for key, value in runner._pending_images.items()}
    local_runtime_calls = []

    def _local_runtime_sentinel(**_kwargs):
        local_runtime_calls.append(True)
        raise AssertionError("local runtime preparation must not run")

    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", _local_runtime_sentinel)

    try:
        await runner._run_background_task("fault prompt", source, f"fault-{fault}")
    finally:
        await app_runner.cleanup()

    assert len(requests) == 1
    assert requests[0]["stream"] is True
    assert requests[0]["messages"][-1] == {"role": "user", "content": "fault prompt"}
    assert len(adapter.sent) == 1
    notification = adapter.sent[0]
    assert "✅ Background task complete" not in notification
    assert "private-marker" not in notification
    assert "/bg" not in notification and "/agents" not in notification
    if fault in {"http-500", "sse-eof"}:
        assert "unknown" in notification.lower(), "HTTP/EOF must remain explicitly unknown, not confirmed failed"
    assert local_runtime_calls == []
    assert runner._pending_images == pending_snapshot
