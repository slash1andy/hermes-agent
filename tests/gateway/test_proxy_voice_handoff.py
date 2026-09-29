"""Wiring the accepted native audio transcription receiver (``gateway/platforms/api_server_audio.py``,
PR #19) into the shared gateway STT path (``gateway/run_inbound.py::_enrich_message_with_transcription``
and callers). Today that path always calls ``tools.transcription_tools.transcribe_audio`` locally,
with no read of ``gateway.proxy_required``/the configured proxy at all -- this module is
test-first RED coverage for the still-missing wiring described in
``proxy-voice-handoff-contract.md``, not a description of already-landed behavior.

Group 1 (``test_native_voice_handoff_child_process``) spawns the real native receiver as a
genuinely separate OS process (reusing ``tests/gateway/proxy_executor_fixture.py`` /
``test_proxy_executor_process.py``'s real ``subprocess`` child, never an invented IPC broker) and
drives the EXISTING gateway-side entry points (``_prepare_profile_scoped_inbound_message_text``,
``_transcribe_pending_audio_event_once``, ``_echo_pending_stt_transcripts_once``,
``_prepare_clarify_reply_text``) directly against it: ordinary enrichment + echo, pending-event
reuse (one STT call, echo-once), a clarify reply (raw transcript, no quotes), owner isolation
(A/B/A), and wrong-key rejection with zero STT admission on the child. Only the child's final
``tools.transcription_tools.transcribe_audio`` seam is a labelled synthetic (a deterministic
function of the uploaded bytes' own sha256, reused from ``proxy_executor_fixture.py``); auth, the
real multipart parse, the real profile-prefix middleware and the real per-request admission
recheck all run unmodified. The parent process pins its OWN root policy required=true while the
child's root stays false (mirrors ``test_proxy_executor_process.py``), and traps the parent's own
local STT/fallback/duration-probe seams so a silent local fallback cannot masquerade as remote
dispatch.

Group 2 (``test_voice_handoff_policy_matrix``) is a compact single-process policy matrix: optional
local stays local, while proxy-owned shapes reach no local STT seams.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
import yaml
import aiohttp.web

import hermes_constants
import tools.transcription_tools as transcription_tools
from agent import secret_scope as ss
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS, _profile_runtime_scope
from gateway.run import _probe_audio_duration as _NATIVE_PROBE_AUDIO_DURATION
from gateway.session import SessionSource, build_session_key

from tests.gateway.test_proxy_background import _RecordingAdapter
from tests.gateway.test_proxy_executor_process import (
    _CHILD_TERMINATE_TIMEOUT,
    _await_child_ready,
    _read_evidence,
    _spawn_child,
)
from tests.gateway.test_proxy_memory_identity import _client_runner, _profile_home, _source


def _expected_transcript(data: bytes) -> str:
    """Mirrors ``proxy_executor_fixture.py::_install_stt_synthetic_seam``'s deterministic mapping
    from the exact uploaded bytes to a transcript -- computed independently here so a match proves
    the real bytes crossed the wire, not a coincidence."""
    return f"synthetic-transcript:{hashlib.sha256(data).hexdigest()[:16]}"


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    hermes_constants.pin_process_hermes_home(None)
    yield
    ss.set_multiplex_active(False)
    hermes_constants.pin_process_hermes_home(None)


@pytest.fixture(autouse=True)
def _trap_local_stt(monkeypatch):
    """Trap the local STT/fallback/duration-probe seams the parent process could reach: a failing
    dict return (never a raised exception) so callers that DO still reach local code exercise their
    real failure-handling path instead of an opaque crash, while the counters prove whether the
    seam was ever reached at all."""
    transcribe_calls: List[str] = []
    fallback_calls: List[str] = []
    probe_calls: List[str] = []

    def _trap_transcribe(path, model=None, source=None):
        transcribe_calls.append(path)
        return {"success": False, "error": "local transcription must not run under this policy"}

    def _trap_fallback(path, model=None):
        fallback_calls.append(path)
        return {"success": False, "error": "local fallback must not run under this policy"}

    async def _trap_probe(path):
        probe_calls.append(path)
        return None

    monkeypatch.setattr(transcription_tools, "transcribe_audio", _trap_transcribe)
    monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", _trap_fallback)
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_probe_audio_duration", _trap_probe)
    return SimpleNamespace(transcribe=transcribe_calls, fallback=fallback_calls, probe=probe_calls)


def _voice_event(source: SessionSource, media_urls: List[str], *, message_id: str, text: str = "") -> MessageEvent:
    return MessageEvent(
        text=text, source=source, message_id=message_id, message_type=MessageType.VOICE,
        media_urls=list(media_urls), media_types=["audio/wav"] * len(media_urls),
    )


# ── group 1 ── real separate-OS-process native receiver, existing gateway entry points ──────────


@pytest.mark.asyncio
async def test_native_voice_handoff_child_process(tmp_path, monkeypatch, _trap_local_stt):
    evidence_path = tmp_path / "evidence.jsonl"
    key_owner_a, key_owner_b = "child-owner-a-stt-key", "child-owner-b-stt-key"
    home_owner_a = _profile_home(
        tmp_path, "owner-a",
        extra_env={"API_SERVER_KEY": key_owner_a, "STT_TEST_MARKER": "owner-a-marker"})
    home_owner_b = _profile_home(
        tmp_path, "owner-b",
        extra_env={"API_SERVER_KEY": key_owner_b, "STT_TEST_MARKER": "owner-b-marker"})

    child_root = _profile_home(tmp_path, "child-root")
    (child_root / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"proxy_required": False}}), encoding="utf-8")

    stderr_path = tmp_path / "child-stderr.log"
    proc = _spawn_child(child_root, evidence_path, home_owner_a, home_owner_b, stderr_path)
    try:
        port = await _await_child_ready(proc, stderr_path)
        base_url = f"http://127.0.0.1:{port}"

        home_client_a = _profile_home(
            tmp_path, "client-a",
            extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-a", "GATEWAY_PROXY_KEY": key_owner_a})
        home_client_b = _profile_home(
            tmp_path, "client-b",
            extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-b", "GATEWAY_PROXY_KEY": key_owner_b})
        home_client_a_wrongkey = _profile_home(
            tmp_path, "client-a-wrongkey",
            extra_env={"GATEWAY_PROXY_URL": f"{base_url}/p/owner-a", "GATEWAY_PROXY_KEY": "wrong-key"})

        parent_root = _profile_home(tmp_path, "parent-root")
        (parent_root / "config.yaml").write_text(
            yaml.safe_dump({"gateway": {"proxy_required": True}}), encoding="utf-8")
        hermes_constants.pin_process_hermes_home(str(parent_root))

        profiles = {
            "client-a": home_client_a, "client-b": home_client_b,
            "client-a-wrongkey": home_client_a_wrongkey,
        }
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profiles[name])
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in profiles)
        monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: list(profiles.items()))
        ss.set_multiplex_active(True)

        from gateway.proxy_admission import gateway_proxy_required
        assert gateway_proxy_required() is True

        client_runner = _client_runner()
        adapter_a, adapter_b = _RecordingAdapter(platform=Platform.MATRIX), _RecordingAdapter(platform=Platform.MATRIX)
        client_runner._profile_adapters["client-a"] = {Platform.MATRIX: adapter_a}
        client_runner._profile_adapters["client-b"] = {Platform.MATRIX: adapter_b}

        source_a, source_b = _source(), _source()
        source_a.profile, source_b.profile = "client-a", "client-b"
        key_a = build_session_key(source_a, profile="owner-a")
        key_b = build_session_key(source_b, profile="owner-b")

        audio_a1 = home_client_a / "cache" / "audio" / "clip-a1.wav"
        audio_a1.parent.mkdir(parents=True, exist_ok=True)
        bytes_a1 = b"riff-fixture-owner-a-clip-1-0123456789"
        audio_a1.write_bytes(bytes_a1)
        expected_a1 = _expected_transcript(bytes_a1)

        # -- ordinary enrichment + echo, through the real existing entry point --
        event_a1 = _voice_event(source_a, [str(audio_a1)], message_id="a1")
        prepared_a1 = await client_runner._prepare_profile_scoped_inbound_message_text(
            event=event_a1, source=source_a, history=[], session_key=key_a,
        )
        assert prepared_a1 is not None
        assert f'"{expected_a1}"' in prepared_a1
        assert adapter_a.sent == [f'🎙️ "{expected_a1}"']

        # -- owner-b, a separate scoped credential and a separate audio clip --
        audio_b1 = home_client_b / "cache" / "audio" / "clip-b1.wav"
        audio_b1.parent.mkdir(parents=True, exist_ok=True)
        bytes_b1 = b"riff-fixture-owner-b-clip-1-9876543210"
        audio_b1.write_bytes(bytes_b1)
        expected_b1 = _expected_transcript(bytes_b1)
        event_b1 = _voice_event(source_b, [str(audio_b1)], message_id="b1")
        prepared_b1 = await client_runner._prepare_profile_scoped_inbound_message_text(
            event=event_b1, source=source_b, history=[], session_key=key_b,
        )
        assert prepared_b1 is not None
        assert f'"{expected_b1}"' in prepared_b1
        assert expected_a1 not in prepared_b1
        assert adapter_b.sent == [f'🎙️ "{expected_b1}"']

        # -- owner-a again: distinct clip, owner isolation holds across an A/B/A sequence --
        audio_a2 = home_client_a / "cache" / "audio" / "clip-a2.wav"
        bytes_a2 = b"riff-fixture-owner-a-clip-2-abcdefghij"
        audio_a2.write_bytes(bytes_a2)
        expected_a2 = _expected_transcript(bytes_a2)
        event_a2 = _voice_event(source_a, [str(audio_a2)], message_id="a2")
        prepared_a2 = await client_runner._prepare_profile_scoped_inbound_message_text(
            event=event_a2, source=source_a, history=[], session_key=key_a,
        )
        assert prepared_a2 is not None
        assert f'"{expected_a2}"' in prepared_a2
        assert adapter_a.sent == [f'🎙️ "{expected_a1}"', f'🎙️ "{expected_a2}"']

        # -- pending-event reuse: one STT call cached on the event, echoed at most once --
        audio_pending = home_client_a / "cache" / "audio" / "clip-pending.wav"
        bytes_pending = b"riff-fixture-owner-a-pending-clip-zzzz"
        audio_pending.write_bytes(bytes_pending)
        expected_pending = _expected_transcript(bytes_pending)
        event_pending = _voice_event(source_a, [str(audio_pending)], message_id="pending")
        text_first, transcripts_first = await client_runner._transcribe_pending_audio_event_once(event_pending, "")
        text_second, transcripts_second = await client_runner._transcribe_pending_audio_event_once(event_pending, "")
        assert transcripts_first == transcripts_second == [expected_pending]
        assert text_first == text_second
        adapter_pending = _RecordingAdapter(platform=Platform.MATRIX)
        await client_runner._echo_pending_stt_transcripts_once(
            event_pending, adapter_pending, source_a, transcripts_first)
        await client_runner._echo_pending_stt_transcripts_once(
            event_pending, adapter_pending, source_a, transcripts_second)
        assert adapter_pending.sent == [f'🎙️ "{expected_pending}"']

        # -- clarify reply: the RAW transcript, never quote-wrapped --
        audio_clarify = home_client_a / "cache" / "audio" / "clip-clarify.wav"
        bytes_clarify = b"riff-fixture-owner-a-clarify-clip-yyy"
        audio_clarify.write_bytes(bytes_clarify)
        expected_clarify = _expected_transcript(bytes_clarify)
        event_clarify = _voice_event(source_a, [str(audio_clarify)], message_id="clarify")
        clarify_reply = await client_runner._prepare_clarify_reply_text(event_clarify)
        assert clarify_reply == expected_clarify
        assert f'"{expected_clarify}"' != clarify_reply

        # -- wrong key: rejected at the native receiver, zero STT admission on the child --
        source_wrong = _source()
        source_wrong.profile = "client-a-wrongkey"
        audio_wrong = home_client_a_wrongkey / "cache" / "audio" / "clip-wrong.wav"
        audio_wrong.parent.mkdir(parents=True, exist_ok=True)
        audio_wrong.write_bytes(b"riff-fixture-owner-a-wrongkey-clip-www")
        records_before_wrongkey = _read_evidence(evidence_path)
        event_wrong = _voice_event(source_wrong, [str(audio_wrong)], message_id="wrongkey")
        await client_runner._prepare_profile_scoped_inbound_message_text(
            event=event_wrong, source=source_wrong, history=[], session_key=key_a,
        )
        assert _read_evidence(evidence_path) == records_before_wrongkey

        # -- every successful STT record was written by the CHILD's own separate OS process,
        #    on the child's own OWNER-scoped credential and root policy (false), never the
        #    parent's local seams --
        records = _read_evidence(evidence_path)
        assert len(records) == 5, "a1, b1, a2, pending, clarify -- wrong-key must add none"
        assert {r["pid"] for r in records} == {proc.pid}
        assert proc.pid != os.getpid()
        assert {r["home"] for r in records} == {str(home_owner_a), str(home_owner_b)}
        assert all(r["own_proxy_required"] is False for r in records)
        markers = {r["home"]: r["stt_test_marker"] for r in records}
        assert markers[str(home_owner_a)] == "owner-a-marker"
        assert markers[str(home_owner_b)] == "owner-b-marker"
        received_bytes = {base64.b64decode(r["bytes_b64"]) for r in records}
        assert received_bytes == {bytes_a1, bytes_a2, bytes_b1, bytes_pending, bytes_clarify}

        # -- the parent's own local STT/fallback/duration-probe seams were never reached: the
        #    required-proxy parent exclusively used the configured native receiver --
        assert _trap_local_stt.transcribe == []
        assert _trap_local_stt.fallback == []
        assert _trap_local_stt.probe == []
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=_CHILD_TERMINATE_TIMEOUT)
        except Exception:
            proc.kill()
            await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=_CHILD_TERMINATE_TIMEOUT)
        finally:
            assert proc.stdout is not None
            proc.stdout.close()
        assert proc.poll() is not None


# ── group 2 ── compact single-process policy matrix (no network): local seams stay reachable ────
# ── only for a genuinely unconfigured/optional-local policy; every proxy-owning shape traps zero ──


def _write_process_root(monkeypatch, home: Path, *, gateway_section: Any) -> None:
    raw = {} if gateway_section is None else {"gateway": gateway_section}
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    hermes_constants.pin_process_hermes_home(str(home))


def _bare_runner(*, stt_enabled: bool = True) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False, stt_enabled=stt_enabled)
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gateway_section,stt_enabled,expect_local",
    [
        (None, True, True),
        ({"proxy_required": True}, True, False),
        ({"proxy_required": "true"}, True, False),
        ({"proxy_required": True}, False, False),
    ],
    ids=[
        "optional-local-no-proxy-stays-local",
        "required-no-url-must-not-reach-local",
        "malformed-root-must-not-reach-local",
        "stt-disabled-under-required-proxy-no-local-probe",
    ],
)
async def test_voice_handoff_policy_matrix(
    monkeypatch, tmp_path, _trap_local_stt, gateway_section, stt_enabled, expect_local,
):
    """A genuinely unconfigured, non-required root leaves the pre-existing local-positive STT path
    unaffected (contract: "Preserve local optional behavior"). Every root shape that owns STT
    through a proxy -- required-with-no-URL, a malformed (non-boolean) policy value, or
    disabled-STT under a required root -- must reach zero local transcription/fallback/duration-
    probe calls; it must never silently keep running the pre-existing unconditional local call."""
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    root = tmp_path / "policy-root"
    root.mkdir()
    _write_process_root(monkeypatch, root, gateway_section=gateway_section)

    runner = _bare_runner(stt_enabled=stt_enabled)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fixture-audio")
    enriched_text, transcripts = await runner._enrich_message_with_transcription("hi", [str(clip)])

    if expect_local:
        assert _trap_local_stt.transcribe == [str(clip)]
    else:
        assert _trap_local_stt.transcribe == []
        assert _trap_local_stt.fallback == []
        assert _trap_local_stt.probe == []
        assert transcripts == []


@pytest.mark.asyncio
async def test_late_local_worker_admission_rechecks_native_root(tmp_path, monkeypatch, _trap_local_stt):
    root = tmp_path / "root"
    root.mkdir()
    _write_process_root(monkeypatch, root, gateway_section=None)
    clip = tmp_path / "late.wav"
    clip.write_bytes(b"not-a-wav")
    reached = []

    original = asyncio.to_thread

    async def worker_wrapper(suppliedfunc, *args, **kwargs):
        def inner(*inner_args, **inner_kwargs):
            reached.append(True)
            _write_process_root(monkeypatch, root, gateway_section={"proxy_required": True})
            return suppliedfunc(*inner_args, **inner_kwargs)

        return await original(inner, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", worker_wrapper)
    await _bare_runner()._enrich_message_with_transcription("hi", [str(clip)])
    assert reached == [True]
    assert _trap_local_stt.transcribe == []
    assert _trap_local_stt.fallback == []


@pytest.mark.asyncio
async def test_late_primary_failure_does_not_enter_local_fallback(tmp_path, monkeypatch, _trap_local_stt):
    root = tmp_path / "root"
    root.mkdir()
    _write_process_root(monkeypatch, root, gateway_section=None)
    clip = tmp_path / "primary.wav"
    clip.write_bytes(b"fixture-audio")
    primary = []

    def failing_primary(path, model=None, source=None):
        primary.append(True)
        _write_process_root(monkeypatch, root, gateway_section={"proxy_required": True})
        return {"success": False, "error": "fixture failure"}

    monkeypatch.setattr(transcription_tools, "transcribe_audio", failing_primary)
    await _bare_runner()._enrich_message_with_transcription("hi", [str(clip)])
    assert primary == [True]
    assert _trap_local_stt.fallback == []


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_ok", [True, False], ids=["primary-success", "fallback-success"])
async def test_optional_local_success_preserves_transcript_and_user_text(
    tmp_path, monkeypatch, primary_ok,
):
    root = tmp_path / "absent-root"
    root.mkdir()
    _write_process_root(monkeypatch, root, gateway_section=None)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fixture-audio")
    calls = []

    def primary(path, model=None, source=None):
        calls.append("primary")
        return {"success": primary_ok, "transcript": "primary text"} if primary_ok else {"success": False}

    def fallback(path, model=None):
        calls.append("fallback")
        return {"success": True, "transcript": "fallback text"}

    monkeypatch.setattr(transcription_tools, "transcribe_audio", primary)
    monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", fallback)
    result, transcripts = await _bare_runner()._enrich_message_with_transcription(
        "original user text", [str(clip)]
    )

    expected = "primary text" if primary_ok else "fallback text"
    assert transcripts == [expected]
    assert result.splitlines() == [f'"{expected}"', "", "original user text"]
    assert calls == (["primary"] if primary_ok else ["primary", "fallback"])


@pytest.mark.asyncio
async def test_disabled_stt_native_probe_rechecks_before_ffprobe(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _write_process_root(monkeypatch, root, gateway_section=None)
    clip = tmp_path / "broken.wav"
    clip.write_bytes(b"broken-wav")
    wave_reads = []
    subprocess_calls = []

    def broken_wave_open(*args, **kwargs):
        wave_reads.append(True)
        _write_process_root(monkeypatch, root, gateway_section={"proxy_required": True})
        raise wave.Error("fixture wave read failure")

    monkeypatch.setattr(wave, "open", broken_wave_open)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *a, **k: subprocess_calls.append(True))
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_probe_audio_duration", _NATIVE_PROBE_AUDIO_DURATION)
    result, transcripts = await _bare_runner(stt_enabled=False)._enrich_message_with_transcription(
        "hi", [str(clip)]
    )
    assert wave_reads == [True]
    assert subprocess_calls == []
    assert transcripts == []
    assert "fixture" not in result


@asynccontextmanager
async def _voice_proxy_server(case):
    requests = []

    async def transcriptions(request):
        requests.append(request)
        await request.read()
        if case == "redirect":
            raise aiohttp.web.HTTPFound("/redirect-target")
        if case == "malformed":
            return aiohttp.web.Response(text='{"text":', content_type="application/json")
        if case == "shape":
            return aiohttp.web.json_response({"text": 7})
        if case == "oversized":
            return aiohttp.web.Response(
                text='{"text":"' + "x" * (_GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS + 1) + '"}',
                content_type="application/json",
            )
        return aiohttp.web.json_response({"text": "remote transcript"})

    async def redirect_target(request):
        requests.append(request)
        await request.read()
        return aiohttp.web.json_response({"text": "redirect target transcript"})

    app = aiohttp.web.Application()
    app.router.add_route("*", "/v1/audio/transcriptions", transcriptions)
    app.router.add_route("*", "/redirect-target", redirect_target)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", requests
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["disabled", "redirect", "malformed", "shape", "oversized"])
async def test_optional_proxy_native_http_negative_matrix(tmp_path, monkeypatch, _trap_local_stt, case):
    root = tmp_path / "root"
    root.mkdir()
    _write_process_root(monkeypatch, root, gateway_section=None)
    clip = tmp_path / "remote.wav"
    clip.write_bytes(b"fixture-audio")
    async with _voice_proxy_server("redirect" if case == "redirect" else case) as (url, requests):
        runner = _bare_runner()
        with _profile_runtime_scope(
            root, prepared_secret_scope={"GATEWAY_PROXY_URL": url, "GATEWAY_PROXY_KEY": "scoped-key"}
        ):
            if case == "disabled":
                runner.config.stt_enabled = False
            result, transcripts = await runner._enrich_message_with_transcription("hi", [str(clip)])
    assert _trap_local_stt.transcribe == []
    assert _trap_local_stt.fallback == []
    assert transcripts == []
    if case == "disabled":
        assert requests == []
    elif case == "redirect":
        assert len(requests) == 1
    else:
        assert len(requests) == 1
        assert "remote transcript" not in result
    assert str(clip) not in result
    assert "scoped-key" not in result


@pytest.mark.asyncio
async def test_optional_proxy_missing_scoped_key_never_opens_clip(tmp_path, monkeypatch, _trap_local_stt):
    root = tmp_path / "root"
    root.mkdir()
    _write_process_root(monkeypatch, root, gateway_section=None)
    named = _profile_home(tmp_path, "named")
    (named / "config.yaml").write_text(yaml.safe_dump({}), encoding="utf-8")
    ss.set_multiplex_active(True)
    clip = tmp_path / "secret.wav"
    clip.write_bytes(b"fixture-audio")
    opened = []
    real_open = open

    def capture_open(path, *args, **kwargs):
        if os.fspath(path) == str(clip):
            opened.append(True)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", capture_open)
    monkeypatch.setenv("GATEWAY_PROXY_KEY", "ambient-decoy")
    async with _voice_proxy_server("success") as (url, requests):
        runner = _bare_runner()
        with _profile_runtime_scope(named, prepared_secret_scope={"GATEWAY_PROXY_URL": url}):
            result, transcripts = await runner._enrich_message_with_transcription("hi", [str(clip)])
    assert opened == []
    assert requests == []
    assert _trap_local_stt.fallback == []
    assert transcripts == []
    assert "ambient-decoy" not in result
