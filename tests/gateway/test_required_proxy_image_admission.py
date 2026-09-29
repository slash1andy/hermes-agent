"""Required-proxy admission for the INBOUND image-preparation boundary
(``gateway/run_inbound.py::_enrich_inbound_images`` / ``_decide_image_input_mode`` /
``_enrich_message_with_vision``), the seam that runs BEFORE ``_run_agent_inner``'s own
``gateway.proxy_required`` recheck.
Per ``proxy-inbound-image-contract.md``: image ingress intended for a remote executor must not
resolve the listener's model/provider/capabilities or call its vision tool before proxy
admission. Every test drives the REAL ``GatewayRunner`` inbound-prep methods and a REAL pinned
process-root ``config.yaml``; only the final vision tool (``tools.vision_tools.vision_analyze_tool``)
and the provider-resolution boundary (``GatewayRunner._resolve_session_agent_runtime``) are
labelled synthetic seams, recorded (never silently swallowed).

The local-positive group is a preserved regression. Denials must retain the original text and
buffer the complete image set for remote execution rather than silently dropping the images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pytest
import yaml

import hermes_constants
from agent import secret_scope as ss
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource, build_session_key


@pytest.fixture(autouse=True)
def _reset_root(monkeypatch):
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    ss.set_multiplex_active(False)
    hermes_constants.pin_process_hermes_home(None)
    yield
    ss.set_multiplex_active(False)
    hermes_constants.pin_process_hermes_home(None)


def _pin_root(monkeypatch, tmp_path, name: str, raw_config: dict | str) -> Path:
    root = tmp_path / name
    root.mkdir()
    config_text = raw_config if isinstance(raw_config, str) else yaml.safe_dump(raw_config)
    (root / "config.yaml").write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", root)
    hermes_constants.pin_process_hermes_home(str(root))
    return root


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")})
    runner.adapters = {}
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="private", user_id="u1", user_name="tester",
    )


def _record_resolver(calls: List[Any]):
    def _resolver(**kwargs):
        calls.append(kwargs)
        return "stub-model", {"provider": "stub-provider", "base_url": "", "api_key": ""}
    return _resolver


def _record_vision(calls: List[Any], *, marker: str = "fabricated-local-vision-marker", on_call=None):
    async def _tool(*, image_url, user_prompt="", **_kwargs):
        calls.append(image_url)
        if on_call:
            on_call()
        return json.dumps({"success": True, "analysis": marker})
    return _tool


# ── group 1: admission-denying / undecided root configs must skip ALL local image work ──────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_config,extra_env",
    [
        ({"gateway": {"proxy_required": True}, "agent": {"image_input_mode": "text"}},
         {"GATEWAY_PROXY_URL": "http://127.0.0.1:9/synthetic-configured"}),
        ({"gateway": {"proxy_required": True}, "agent": {"image_input_mode": "text"}}, {}),
        ("gateway:\n  proxy_required: [\nprivate-canary: do-not-read\n", {}),
        ({"gateway": [], "agent": {"image_input_mode": "text"}}, {}),
        ({"gateway": {"proxy_required": False}, "agent": {"image_input_mode": "text"}},
         {"GATEWAY_PROXY_URL": "http://127.0.0.1:9/synthetic-configured"}),
    ],
    ids=["required-with-url", "required-missing-url", "invalid-yaml", "malformed-gateway-section", "optional-configured-proxy"],
)
async def test_admission_denying_or_proxy_configured_roots_skip_local_image_work(
    monkeypatch, tmp_path, raw_config, extra_env,
):
    root = _pin_root(monkeypatch, tmp_path, "admission-root", raw_config)
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)

    runner = _make_runner()
    resolver_calls: List[Any] = []
    vision_calls: List[Any] = []
    runner._resolve_session_agent_runtime = _record_resolver(resolver_calls)
    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _record_vision(vision_calls))

    source = _source()
    session_key = build_session_key(source)
    image = str(tmp_path / "a.png")
    text = await runner._enrich_inbound_images(source, session_key, "original text", [image])

    assert resolver_calls == []
    assert vision_calls == []
    assert text == "original text"
    assert runner._consume_pending_native_image_paths(session_key) == [image]
    assert root.exists()


# ── group 2: local (non-proxy) path is unaffected — preserved regression ────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,expect_vision,proxy_required",
    [("text", True, False), ("text", True, None), ("native", False, None)],
    ids=["text-positive-root-false", "text-positive-root-absent", "native-positive"],
)
async def test_local_positive_path_unaffected(monkeypatch, tmp_path, mode, expect_vision, proxy_required):
    gateway = {} if proxy_required is None else {"proxy_required": proxy_required}
    _pin_root(monkeypatch, tmp_path, "local-root", {"gateway": gateway, "agent": {"image_input_mode": mode}})
    runner = _make_runner()
    resolver_calls: List[Any] = []
    vision_calls: List[Any] = []
    runner._resolve_session_agent_runtime = _record_resolver(resolver_calls)
    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _record_vision(vision_calls))

    source = _source()
    session_key = build_session_key(source)
    image = str(tmp_path / "a.png")
    text = await runner._enrich_inbound_images(source, session_key, "original text", [image])

    if expect_vision:
        assert vision_calls == [image]
        assert "fabricated-local-vision-marker" in text
        assert runner._consume_pending_native_image_paths(session_key) == []
    else:
        assert vision_calls == []
        assert text == "original text"
        assert runner._consume_pending_native_image_paths(session_key) == [image]


# ── group 3: late flip mid-decision, direct helper call, multi-image per-item recheck ───────────


@pytest.mark.asyncio
async def test_late_flip_during_awaited_decision_denies_after_return(monkeypatch, tmp_path):
    """The root policy flips false -> true while ``_decide_image_input_mode`` is awaited on its
    worker thread, so the late denial must preserve text and buffer the image."""
    root = _pin_root(monkeypatch, tmp_path, "lateflip-root", {"gateway": {"proxy_required": False}, "agent": {"image_input_mode": "text"}})

    def _flip_then_resolve(**kwargs):
        (root / "config.yaml").write_text(
            yaml.safe_dump({"gateway": {"proxy_required": True}, "agent": {"image_input_mode": "text"}}),
            encoding="utf-8",
        )
        return "stub-model", {"provider": "stub-provider", "base_url": "", "api_key": ""}

    runner = _make_runner()
    runner._resolve_session_agent_runtime = _flip_then_resolve
    vision_calls: List[Any] = []
    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _record_vision(vision_calls))

    source = _source()
    session_key = build_session_key(source)
    images = [str(tmp_path / name) for name in ("a.png", "b.png")]
    text = await runner._enrich_inbound_images(source, session_key, "original text", images)

    assert text == "original text"
    assert vision_calls == []
    assert runner._consume_pending_native_image_paths(session_key) == images


@pytest.mark.asyncio
async def test_thread_entry_flip_denies_before_actual_local_decision(monkeypatch, tmp_path):
    """A required flip between thread scheduling and the real decision must block resolution."""
    root = _pin_root(monkeypatch, tmp_path, "thread-entry-root", {"gateway": {"proxy_required": False}, "agent": {"image_input_mode": "text"}})
    runner = _make_runner()
    resolver_calls: List[Any] = []
    runner._resolve_session_agent_runtime = _record_resolver(resolver_calls)
    actual_decide = runner._decide_image_input_mode

    def _flip_then_decide(*args, **kwargs):
        (root / "config.yaml").write_text(
            yaml.safe_dump({"gateway": {"proxy_required": True}, "agent": {"image_input_mode": "text"}}),
            encoding="utf-8",
        )
        return actual_decide(*args, **kwargs)

    runner._decide_image_input_mode = _flip_then_decide
    vision_calls: List[Any] = []
    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _record_vision(vision_calls))

    source = _source()
    session_key = build_session_key(source)
    images = [str(tmp_path / name) for name in ("a.png", "b.png")]
    text = await runner._enrich_inbound_images(source, session_key, "original text", images)

    assert resolver_calls == []
    assert vision_calls == []
    assert text == "original text"
    assert runner._consume_pending_native_image_paths(session_key) == images


@pytest.mark.asyncio
async def test_direct_local_vision_helper_denied_under_required_root(monkeypatch, tmp_path):
    """The direct local-vision helper must refuse under a required root, like its background caller."""
    _pin_root(monkeypatch, tmp_path, "direct-root", {"gateway": {"proxy_required": True}})
    runner = _make_runner()
    vision_calls: List[Any] = []
    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _record_vision(vision_calls))

    from gateway.proxy_admission import ProxyPolicyError

    image = str(tmp_path / "a.png")
    with pytest.raises(ProxyPolicyError, match="proxy"):
        await runner._enrich_message_with_vision("original text", [image])

    assert vision_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["direct", "ordinary"], ids=["direct", "ordinary"])
async def test_multi_image_first_vision_flip_denies_second_local_call(monkeypatch, tmp_path, path):
    """After the first vision call flips the root, both local entry points deny the second image."""
    root = _pin_root(monkeypatch, tmp_path, "multi-root", {"gateway": {"proxy_required": False}, "agent": {"image_input_mode": "text"}})

    vision_calls: List[Any] = []

    def _flip_root():
        (root / "config.yaml").write_text(
            yaml.safe_dump({"gateway": {"proxy_required": True}, "agent": {"image_input_mode": "text"}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        _record_vision(vision_calls, on_call=(lambda: _flip_root() if len(vision_calls) == 1 else None)),
    )

    runner = _make_runner()
    image_a = str(tmp_path / "a.png")
    image_b = str(tmp_path / "b.png")
    from gateway.proxy_admission import ProxyPolicyError

    if path == "direct":
        with pytest.raises(ProxyPolicyError, match="proxy"):
            await runner._enrich_message_with_vision("original text", [image_a, image_b])
    else:
        runner._resolve_session_agent_runtime = _record_resolver([])
        source = _source()
        session_key = build_session_key(source)
        text = await runner._enrich_inbound_images(source, session_key, "original text", [image_a, image_b])
        assert text == "original text"
        assert "fabricated-local-vision-marker" not in text
        assert runner._consume_pending_native_image_paths(session_key) == [image_a, image_b]

    assert vision_calls == [image_a]


# ── group 4: root-vs-served contradictory scope — process root always governs ───────────────────


@pytest.mark.asyncio
async def test_root_wins_over_contradictory_served_profile_config(monkeypatch, tmp_path):
    """Process root pins required=true; a contradictory served profile cannot enable local work."""
    root = _pin_root(monkeypatch, tmp_path, "contradiction-root", {"gateway": {"proxy_required": True}})
    served = root / "profiles" / "served"
    served.mkdir(parents=True)
    (served / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"proxy_required": False}, "agent": {"image_input_mode": "text"}}),
        encoding="utf-8",
    )

    runner = _make_runner()
    resolver_calls: List[Any] = []
    vision_calls: List[Any] = []
    runner._resolve_session_agent_runtime = _record_resolver(resolver_calls)
    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _record_vision(vision_calls))

    source = _source()
    session_key = build_session_key(source)
    image = str(tmp_path / "a.png")
    with _profile_runtime_scope(served):
        text = await runner._enrich_inbound_images(source, session_key, "original text", [image])

    assert resolver_calls == []
    assert vision_calls == []
    assert text == "original text"


@pytest.mark.asyncio
async def test_root_false_allows_local_text_over_contradictory_served_profile(monkeypatch, tmp_path):
    """An explicit root false remains the governing local-positive policy."""
    _pin_root(monkeypatch, tmp_path, "local-contradiction-root", {"gateway": {"proxy_required": False}, "agent": {"image_input_mode": "text"}})
    served = tmp_path / "served"
    served.mkdir()
    (served / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"proxy_required": True}, "agent": {"image_input_mode": "text"}}),
        encoding="utf-8",
    )

    runner = _make_runner()
    vision_calls: List[Any] = []
    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _record_vision(vision_calls))
    image = str(tmp_path / "a.png")
    with _profile_runtime_scope(served):
        text = await runner._enrich_inbound_images(_source(), build_session_key(_source()), "original text", [image])

    assert vision_calls == [image]
    assert "fabricated-local-vision-marker" in text
