"""Real quick-command exec admission across root policy and profile scope."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import pytest
import yaml

import hermes_constants
from agent import secret_scope as ss
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


_SAFE_COMMAND = "printf quick-command-output"
_REFUSAL_WORDING = "quick command"


def _write_config(home: Path, gateway: object) -> None:
    value = {"gateway": gateway}
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(value), encoding="utf-8")


def _source(profile: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=f"chat-{profile}",
        user_id="user-1",
        chat_type="dm",
        profile=profile,
    )


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner.config.quick_commands = {
        "probe": {"type": "exec", "command": _SAFE_COMMAND}
    }
    runner._draining = False
    return runner


def _event(profile: str) -> MessageEvent:
    return MessageEvent(text="/probe", source=_source(profile), message_id=f"m-{profile}")


def _count_effect_sinks(monkeypatch):
    from tools.environments import local

    counts = {"env": 0, "shell": 0}
    real_env = local.build_subprocess_env
    real_shell = asyncio.create_subprocess_shell

    def counted_env(*args, **kwargs):
        counts["env"] += 1
        return real_env(*args, **kwargs)

    async def counted_shell(*args, **kwargs):
        counts["shell"] += 1
        return await real_shell(*args, **kwargs)

    monkeypatch.setattr(local, "build_subprocess_env", counted_env)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", counted_shell)
    return counts


def _assert_safe_refusal(result: str, *forbidden: str) -> None:
    lowered = result.lower()
    assert _REFUSAL_WORDING in lowered
    assert "refus" in lowered or "disabled" in lowered or "unsupported" in lowered
    for text in forbidden:
        assert text.lower() not in lowered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "root_gateway,profile_gateway,forbidden",
    [
        ({"proxy_required": True}, {"proxy_url": "http://proxy.invalid/v1"}, ("proxy.invalid",)),
        ({"proxy_required": True}, {}, ("probe",)),
        ({"proxy_required": "true"}, {"proxy_url": "http://proxy.invalid/v1"}, ("true",)),
        ("not-a-mapping", {"proxy_url": "http://proxy.invalid/v1"}, ("not-a-mapping",)),
        ({"proxy_required": False}, {"proxy_url": 17}, ("17",)),
    ],
    ids=["required-configured", "required-url-absent", "root-nonbool", "rootsection-malformed", "scoped-url-malformed"],
)
async def test_quick_exec_refuses_policy_or_resolution_faults_before_effects(
    monkeypatch, tmp_path, root_gateway, profile_gateway, forbidden
):
    process_home = tmp_path / "process"
    profile_home = process_home / "profiles" / "client-a"
    _write_config(process_home, root_gateway)
    _write_config(profile_home, profile_gateway)
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    hermes_constants.pin_process_hermes_home(str(process_home))
    ss.set_multiplex_active(True)
    counts = _count_effect_sinks(monkeypatch)
    runner = _runner()
    runner.config.multiplex_profiles = True
    try:
        source = _source("client-a")
        assert runner._resolve_profile_home_for_source(source) == profile_home
        result = await runner._hm_dispatch_quick_and_plugin_commands(
            _event("client-a"), source, "probe"
        )
        assert result[0] is True
        _assert_safe_refusal(str(result[1]), *forbidden)
        assert counts == {"env": 0, "shell": 0}

        from gateway.run import _async_profile_runtime_scope

        async with _async_profile_runtime_scope(profile_home):
            direct = await runner._hm_run_exec_quick_command("probe", _SAFE_COMMAND)
        _assert_safe_refusal(direct, *forbidden)
        assert counts == {"env": 0, "shell": 0}
    finally:
        assert os.environ.get("HERMES_HOME") == str(process_home)
        hermes_constants.pin_process_hermes_home(None)
        ss.set_multiplex_active(False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "root_required,expected_profile",
    [(None, "client-b"), (False, "client-b"), (True, None)],
    ids=[
        "absent-root-allows-absent-sibling",
        "optional-root-allows-absent-sibling",
        "required-root-denies-siblings",
    ],
)
async def test_quick_exec_profile_scope_a_b_a_preserves_root_policy_and_restores(
    monkeypatch, tmp_path, root_required, expected_profile
):
    process_home = tmp_path / "process"
    profile_a = process_home / "profiles" / "client-a"
    profile_b = process_home / "profiles" / "client-b"
    root_gateway = {} if root_required is None else {"proxy_required": root_required}
    _write_config(process_home, root_gateway)
    _write_config(profile_a, {"proxy_required": False, "proxy_url": "http://proxy.invalid/v1"})
    _write_config(profile_b, {"proxy_required": False})
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    hermes_constants.pin_process_hermes_home(str(process_home))
    ss.set_multiplex_active(True)
    counts = _count_effect_sinks(monkeypatch)
    runner = _runner()
    runner.config.multiplex_profiles = True
    outputs = {}
    try:
        for profile in ("client-a", "client-b", "client-a"):
            source = _source(profile)
            expected_home = process_home / "profiles" / profile
            assert runner._resolve_profile_home_for_source(source) == expected_home
            result = await runner._hm_dispatch_quick_and_plugin_commands(
                _event(profile), source, "probe"
            )
            assert result[0] is True
            outputs.setdefault(profile, []).append(str(result[1]))
            if profile == expected_profile:
                assert result[1] == "quick-command-output"
            else:
                _assert_safe_refusal(str(result[1]), "proxy.invalid", "probe")

            assert os.environ["HERMES_HOME"] == str(process_home)
            assert hermes_constants.get_hermes_home() == process_home
            assert ss.current_secret_scope() is None

        assert set(outputs) == {"client-a", "client-b"}
        assert outputs["client-a"] == [outputs["client-a"][0], outputs["client-a"][0]]
        all_outputs = sum(outputs.values(), [])
        if expected_profile:
            assert "quick-command-output" in all_outputs
        else:
            assert "quick-command-output" not in all_outputs
        assert counts == (
            {"env": 1, "shell": 1} if expected_profile else {"env": 0, "shell": 0}
        )
        assert os.environ["HERMES_HOME"] == str(process_home)
        assert hermes_constants.get_hermes_home() == process_home
    finally:
        hermes_constants.pin_process_hermes_home(None)
        ss.set_multiplex_active(False)
