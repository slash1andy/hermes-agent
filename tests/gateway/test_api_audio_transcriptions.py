"""HTTP contract tests for the native audio transcription receiver."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

import gateway.proxy_admission as proxy_admission
import tools.transcription_common as transcription_common
import tools.transcription_tools as transcription_tools
from tests.gateway.test_api_proxy_admission import (
    _auth_headers as _admission_auth_headers,
    _create_app,
    _make_adapter,
    _write_root_config,
)
from tests.gateway.test_api_server_isolated_cron_delivery import _key

pytest_plugins = ("tests.gateway.test_api_server_isolated_cron_delivery",)


@pytest.fixture
def audio_profiles(profiles):
    return profiles


@pytest.fixture
def audio_root_home(root_home):
    return root_home


@pytest.fixture
def audio_adapter(adapter):
    return adapter


@pytest.fixture
def audio_app(app):
    return app

ENDPOINT = "/v1/audio/transcriptions"
FIELD = "file"
_API_KEY = "synthetic-audio-api-key-not-real-0000"
_DENIAL_MARKER = "gateway.proxy_required"
_SYNTHETIC_AUDIO = b"synthetic-riff-wav-bytes-not-real-audio-0123456789"

_UNREACHABLE_SEAMS = []


@pytest.fixture(autouse=True)
def _assert_unreachable_seams():
    _UNREACHABLE_SEAMS.clear()
    yield
    assert all(not seam.calls for seam in _UNREACHABLE_SEAMS)


def _auth_headers(key: Optional[str] = _API_KEY) -> Dict[str, str]:
    return _admission_auth_headers(key)


def _multipart_form(
    *, field: str = FIELD, filename: str = "clip.wav", content: bytes = _SYNTHETIC_AUDIO,
    content_type: str = "audio/wav", extra: Optional[Dict[str, str]] = None,
    omit_file: bool = False, second_file: bool = False,
) -> FormData:
    form = FormData()
    if not omit_file:
        form.add_field(field, content, filename=filename, content_type=content_type)
    if second_file:
        form.add_field(field, content, filename="second.wav", content_type=content_type)
    for key, value in (extra or {}).items():
        form.add_field(key, value)
    return form


class _CapturingTranscribe:
    """Final real seam (success side): records the exact file bytes + args it received."""

    def __init__(self, transcript: str = "synthetic transcript", provider: str = "local") -> None:
        self.calls: List[Dict[str, Any]] = []
        self.transcript = transcript
        self.provider = provider

    def __call__(self, file_path: str, model: Optional[str] = None, source: Optional[str] = None
                 ) -> Dict[str, Any]:
        data = Path(file_path).read_bytes()
        self.calls.append({"file_path": file_path, "bytes": data, "model": model, "source": source})
        return {"success": True, "transcript": self.transcript, "provider": self.provider}


class _FailingTranscribe:
    """Final real seam (failure side): never leaks its error text into an HTTP body."""

    def __init__(self, error: str = "synthetic provider failure PRIVATE_TOKEN=fictional-audio-token") -> None:
        self.calls: List[str] = []
        self.error = error

    def __call__(self, file_path: str, model: Optional[str] = None, source: Optional[str] = None
                 ) -> Dict[str, Any]:
        self.calls.append(file_path)
        return {"success": False, "transcript": "", "error": self.error}


class _PrivateProviderFailure(RuntimeError):
    pass


class _UnreachableSeam:
    """Fails loudly if a denied/disabled/short-circuited request still reaches the provider."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: List[tuple] = []
        _UNREACHABLE_SEAMS.append(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append((args, kwargs))
        raise AssertionError(f"{self.label} must not be reached for this request")


class _BlockingTranscribe:
    """Blocks on a real ``threading.Event`` so cancellation timing is deterministic, never a sleep."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.captured_path: Optional[str] = None
        self.captured_bytes: Optional[bytes] = None

    def __call__(self, file_path: str, model: Optional[str] = None, source: Optional[str] = None
                 ) -> Dict[str, Any]:
        self.captured_path = file_path
        self.captured_bytes = Path(file_path).read_bytes()
        self.started.set()
        self.release.wait(timeout=10)
        self.finished.set()
        return {"success": True, "transcript": "post-cancel transcript", "provider": "local"}


async def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Bounded poll for a cross-thread filesystem/state side effect with no production hook to
    await directly; capped so a stalled implementation fails the test instead of hanging it."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


# ---------------------------------------------------------------------------
# (1) Auth before body parsing; required/malformed/served-root policy exactly like the other
# shared-admission agent routes.
# ---------------------------------------------------------------------------


class TestAdmissionAndAuth:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("gateway_section", [None, {"proxy_required": True}, {"proxy_required": False}],
                              ids=["absent", "required", "not-required"])
    async def test_missing_or_wrong_key_is_401_before_body_parsing(self, monkeypatch, gateway_section):
        _write_root_config(gateway_section=gateway_section)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            missing = await cli.post(ENDPOINT, data=_multipart_form())
            assert missing.status == 401
            wrong = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers("not-the-key"))
            assert wrong.status == 401
        assert adapter._pending_agent_requests == 0

    @pytest.mark.asyncio
    async def test_required_root_denies_before_dispatch_403(self, monkeypatch):
        _write_root_config(gateway_section={"proxy_required": True})
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        monkeypatch.setattr(
            transcription_tools, "transcribe_audio_local_fallback", _UnreachableSeam("local_fallback"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body_text = await response.text()
        assert response.status == 403, body_text
        assert _DENIAL_MARKER in body_text
        assert adapter._pending_agent_requests == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("raw_yaml", "gateway_section"),
        [(None, "not-a-mapping"), ("- just\n- a\n- list\n", None)],
        ids=["gateway-not-a-mapping", "root-not-a-mapping"],
    )
    async def test_malformed_root_denies_with_503(self, monkeypatch, raw_yaml, gateway_section):
        _write_root_config(raw_yaml=raw_yaml, gateway_section=gateway_section)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body_text = await response.text()
        assert response.status == 503, body_text
        assert _DENIAL_MARKER in body_text
        assert "not-a-mapping" not in body_text
        assert adapter._pending_agent_requests == 0

    @pytest.mark.asyncio
    async def test_root_policy_governs_over_served_profile_contradiction(
        self, monkeypatch, audio_profiles, audio_root_home, audio_adapter, audio_app,
    ):
        import hermes_constants
        import yaml as _yaml

        audio_root_home.joinpath("config.yaml").write_text(
            _yaml.safe_dump({"gateway": {"proxy_required": True}}), encoding="utf-8")
        alice_key = _key("alice-audio-api")
        alice_env = audio_profiles["alice"] / ".env"
        alice_env.write_text(alice_env.read_text(encoding="utf-8") + f"API_SERVER_KEY={alice_key}\n",
                              encoding="utf-8")
        alice_config = _yaml.safe_load((audio_profiles["alice"] / "config.yaml").read_text(encoding="utf-8"))
        alice_config["gateway"] = {"proxy_required": False}
        (audio_profiles["alice"] / "config.yaml").write_text(_yaml.safe_dump(alice_config), encoding="utf-8")
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))

        hermes_constants.pin_process_hermes_home(str(audio_root_home))
        try:
            async with TestClient(TestServer(audio_app)) as cli:
                response = await cli.post(
                    f"/p/alice{ENDPOINT}", data=_multipart_form(),
                    headers={"Authorization": f"Bearer {alice_key}"})
                body_text = await response.text()
        finally:
            hermes_constants.pin_process_hermes_home(None)
        assert response.status == 403, body_text
        assert _DENIAL_MARKER in body_text


# ---------------------------------------------------------------------------
# (2) Strict multipart contract: exactly one file field, no paths/URLs/model selectors/extra
# fields, no executable filename extension, no empty/oversized upload.
# ---------------------------------------------------------------------------


class TestStrictMultipartContract:
    @staticmethod
    def _raw_multipart(*, boundary: str, parts: List[bytes], closing: bool = True) -> bytes:
        body = b"".join(b"--" + boundary.encode() + b"\r\n" + part + b"\r\n" for part in parts)
        return body + (b"--" + boundary.encode() + b"--\r\n" if closing else b"")

    @staticmethod
    def _file_part(*, filename: str = "clip.wav", content: bytes = _SYNTHETIC_AUDIO) -> bytes:
        return (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode() + content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("content_type", "body"),
        [
            ("multipart/form-data; boundary=raw-boundary", b"--raw-boundary\r\n" +
             b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n\r\n' +
             _SYNTHETIC_AUDIO + b"\r\n"),
            ("multipart/form-data; boundary=raw-boundary", b"--raw-boundary\r\n" +
             b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n\r\n' +
             _SYNTHETIC_AUDIO + b"\r\n" +
             b"--raw-boundary\r\nContent-Type: text/plain\r\n\r\nignored\r\n--raw-boundary--\r\n"),
            ("application/multipart/form-data-evil; boundary=raw-boundary", b"--raw-boundary--\r\n"),
            ("multipart/form-data; boundary=raw-boundary", b"--raw-boundary\r\n" +
             b'Content-Disposition: form-data; name="file"; name="other"; filename="clip.wav"\r\n\r\n' +
             _SYNTHETIC_AUDIO + b"\r\n--raw-boundary--\r\n"),
            ("multipart/form-data; boundary=raw-boundary", b"--raw-boundary\r\n" +
             b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n' +
             b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n\r\n' +
             _SYNTHETIC_AUDIO + b"\r\n--raw-boundary--\r\n"),
            ("multipart/form-data; boundary=raw-boundary", b"--raw-boundary\r\n" +
             b'Content-Disposition: form-dataevil; name="file"; filename="clip.wav"\r\n\r\n' +
             _SYNTHETIC_AUDIO + b"\r\n--raw-boundary--\r\n"),
            ("multipart/form-data; boundary=raw-boundary", None),
        ],
        ids=["missing-closing-boundary", "extra-non-form-data-part", "wrong-top-level-content-type",
             "duplicate-name-param", "duplicate-content-disposition", "evil-content-disposition",
             "pathlike-filename"],
    )
    async def test_raw_multipart_regressions_denied_400(self, monkeypatch, content_type, body):
        _write_root_config(gateway_section=None)
        boundary = "raw-boundary"
        if body is None:
            body = self._raw_multipart(boundary=boundary, parts=[self._file_part(filename="/tmp/clip.wav")])
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        from aiohttp.payload import BytesPayload
        payload = BytesPayload(body, content_type=content_type)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=payload, headers=_auth_headers())
        assert response.status == 400
        assert adapter._pending_agent_requests == 0

    @pytest.mark.asyncio
    async def test_padded_filename_denied_400(self, monkeypatch):
        _write_root_config(gateway_section=None)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        boundary = "padded-boundary"
        body = self._raw_multipart(boundary=boundary, parts=[self._file_part(filename=" clip.wav ")])
        from aiohttp.payload import BytesPayload
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                ENDPOINT, data=BytesPayload(body, content_type=f"multipart/form-data; boundary={boundary}"),
                headers=_auth_headers())
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_base64_transfer_encoding_denied_400(self, monkeypatch):
        _write_root_config(gateway_section=None)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        boundary = "cte-boundary"
        body = self._raw_multipart(boundary=boundary, parts=[
            self._file_part() .replace(b"Content-Type: audio/wav", b"Content-Transfer-Encoding: base64\r\nContent-Type: audio/wav")
        ])
        from aiohttp.payload import BytesPayload
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                ENDPOINT, data=BytesPayload(body, content_type=f"multipart/form-data; boundary={boundary}"),
                headers=_auth_headers())
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_no_configured_key_is_401_without_work(self, monkeypatch):
        _write_root_config(gateway_section=None)
        seam = _UnreachableSeam("transcribe_audio")
        monkeypatch.setattr(transcription_tools, "transcribe_audio", seam)
        adapter = _make_adapter(None)  # type: ignore[arg-type]
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
        assert response.status == 401
        assert seam.calls == []
        assert adapter._pending_agent_requests == 0

    @pytest.mark.asyncio
    async def test_missing_file_field_denied_400(self, monkeypatch):
        _write_root_config(gateway_section=None)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                ENDPOINT, data=_multipart_form(omit_file=True), headers=_auth_headers())
        assert response.status == 400
        assert adapter._pending_agent_requests == 0

    @pytest.mark.asyncio
    async def test_duplicate_file_field_denied_400(self, monkeypatch):
        _write_root_config(gateway_section=None)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                ENDPOINT, data=_multipart_form(second_file=True), headers=_auth_headers())
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_empty_file_denied_400(self, monkeypatch):
        _write_root_config(gateway_section=None)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                ENDPOINT, data=_multipart_form(content=b""), headers=_auth_headers())
        assert response.status == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("extra_field", ["path", "url", "model", "provider", "session_id"],
                              ids=["path", "url", "model", "provider", "session_id"])
    async def test_extra_or_unknown_field_denied_400(self, monkeypatch, extra_field):
        _write_root_config(gateway_section=None)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                ENDPOINT, data=_multipart_form(extra={extra_field: "irrelevant-value"}),
                headers=_auth_headers())
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_unsupported_filename_extension_denied_400(self, monkeypatch):
        _write_root_config(gateway_section=None)
        assert ".exe" not in transcription_common.SUPPORTED_FORMATS
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                ENDPOINT, data=_multipart_form(filename="payload.exe"), headers=_auth_headers())
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_oversized_total_content_denied_400(self, monkeypatch):
        """Enforces via the EXISTING native remote-audio cap (``transcription_common.MAX_FILE_SIZE``),
        monkeypatched small so the oversized payload built here stays tiny."""
        _write_root_config(gateway_section=None)
        monkeypatch.setattr(transcription_common, "MAX_FILE_SIZE", 1024)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        oversized = b"a" * 2048
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                ENDPOINT, data=_multipart_form(content=oversized), headers=_auth_headers())
        assert response.status == 413

    @pytest.mark.asyncio
    async def test_oversized_wire_body_denied_without_large_allocation(self, monkeypatch):
        _write_root_config(gateway_section=None)
        try:
            receiver = importlib.import_module("gateway.platforms.api_server_audio")
        except ModuleNotFoundError:
            receiver = None
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        if receiver is None:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            assert response.status == 413
            return

        monkeypatch.setattr(receiver, "MAX_AUDIO_UPLOAD_WIRE_BYTES", 1024)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        boundary = "audio-cap-boundary"

        async def chunks():
            yield f"--{boundary}\r\n".encode()
            yield b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
            yield b"Content-Type: audio/wav\r\n\r\n"
            yield b"x" * 512
            yield b"y" * 513
            yield f"\r\n--{boundary}--\r\n".encode()

        from aiohttp.payload import AsyncIterablePayload
        payload = AsyncIterablePayload(
            chunks(), content_type=f"multipart/form-data; boundary={boundary}")
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=payload, headers=_auth_headers())
        assert response.status == 413


# ---------------------------------------------------------------------------
# (3) Provider success / fallback / failure — the final seam is exactly
# ``tools.transcription_tools.transcribe_audio`` / ``transcribe_audio_local_fallback``.
# ---------------------------------------------------------------------------


class TestProviderSeamOutcomes:
    @pytest.mark.asyncio
    async def test_success_returns_only_text_key(self, monkeypatch):
        _write_root_config(gateway_section=None)
        stub = _CapturingTranscribe(transcript="hello from the synthetic provider")
        monkeypatch.setattr(transcription_tools, "transcribe_audio", stub)
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body = await response.json()
        assert response.status == 200, body
        assert body == {"text": "hello from the synthetic provider"}
        assert len(stub.calls) == 1
        assert stub.calls[0]["bytes"] == _SYNTHETIC_AUDIO

    @pytest.mark.asyncio
    async def test_silence_is_valid_empty_text(self, monkeypatch):
        _write_root_config(gateway_section=None)
        stub = _CapturingTranscribe(transcript="")
        monkeypatch.setattr(transcription_tools, "transcribe_audio", stub)
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body = await response.json()
        assert response.status == 200, body
        assert body == {"text": ""}

    @pytest.mark.asyncio
    async def test_configured_provider_failure_recovers_via_local_fallback(self, monkeypatch):
        _write_root_config(gateway_section=None)
        failing = _FailingTranscribe()
        fallback = _CapturingTranscribe(transcript="recovered via local fallback")
        monkeypatch.setattr(transcription_tools, "transcribe_audio", failing)
        monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", fallback)
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body = await response.json()
        assert response.status == 200, body
        assert body == {"text": "recovered via local fallback"}
        assert len(failing.calls) == 1
        assert len(fallback.calls) == 1
        assert not Path(failing.calls[0]).exists(), "upload must be cleaned up after processing ends"

    @pytest.mark.asyncio
    async def test_provider_and_fallback_failure_is_fixed_error_no_leak(self, monkeypatch):
        _write_root_config(gateway_section=None)
        failing = _FailingTranscribe(error="fictional provider token PRIVATE/tmp/should/not/leak")

        def _fallback_also_fails(file_path, model=None):
            return {"success": False, "transcript": "", "error": "fallback also failed"}

        monkeypatch.setattr(transcription_tools, "transcribe_audio", failing)
        monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", _fallback_also_fails)
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body_text = await response.text()
        assert response.status == 502, body_text
        assert "PRIVATE" not in body_text
        assert "sk-leak-me-never" not in body_text
        assert "/tmp" not in body_text
        assert len(failing.calls) == 1
        assert not Path(failing.calls[0]).exists(), "upload must be cleaned up even on failure"

    @pytest.mark.asyncio
    async def test_provider_private_exception_is_fixed_502_without_leak(self, monkeypatch, caplog):
        _write_root_config(gateway_section=None)
        sentinel = "PRIVATE_AUDIO_SENTINEL_7f2a"

        def _raise(*args, **kwargs):
            raise _PrivateProviderFailure(sentinel)

        monkeypatch.setattr(transcription_tools, "transcribe_audio", _raise)
        monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", _UnreachableSeam("local_fallback"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body_text = await response.text()
        assert response.status == 502
        assert sentinel not in body_text
        assert sentinel not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("result", [None, [], "not-a-result"])
    async def test_malformed_provider_result_is_502_without_fallback(self, monkeypatch, result):
        _write_root_config(gateway_section=None)
        calls = []

        def _provider(*args, **kwargs):
            calls.append(True)
            return result

        monkeypatch.setattr(transcription_tools, "transcribe_audio", _provider)
        fallback = _UnreachableSeam("local_fallback")
        monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", fallback)
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
        assert response.status == 502
        assert calls == [True]
        assert fallback.calls == []

    @pytest.mark.asyncio
    async def test_definitive_false_provider_result_permits_native_fallback(self, monkeypatch):
        _write_root_config(gateway_section=None)
        primary = _FailingTranscribe(error="provider declined")
        fallback = _CapturingTranscribe(transcript="native fallback")
        monkeypatch.setattr(transcription_tools, "transcribe_audio", primary)
        monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", fallback)
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body = await response.json()
        assert response.status == 200
        assert body == {"text": "native fallback"}
        assert len(fallback.calls) == 1


# ---------------------------------------------------------------------------
# (4) STT disabled — fixed disabled outcome, no provider or subprocess ever runs.
# ---------------------------------------------------------------------------


class TestSTTDisabled:
    @pytest.mark.asyncio
    async def test_stt_disabled_short_circuits_before_any_provider(self, monkeypatch):
        _write_root_config(raw_yaml="stt:\n  enabled: false\n")
        monkeypatch.setattr(transcription_tools, "transcribe_audio", _UnreachableSeam("transcribe_audio"))
        monkeypatch.setattr(
            transcription_tools, "transcribe_audio_local_fallback", _UnreachableSeam("local_fallback"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body_text = await response.text()
        assert response.status == 503, body_text
        assert "stt" in body_text.lower()
        assert adapter._pending_agent_requests == 0


# ---------------------------------------------------------------------------
# (5) Late policy recheck at the actual threaded processing entry, and again before any fallback —
# wraps the REAL ``gateway.proxy_admission.admit_local_api_agent_creation`` (never faked itself);
# both document the target contract and are expected to fail until the production leaf rechecks.
# ---------------------------------------------------------------------------


class TestLatePolicyRecheck:
    @pytest.mark.asyncio
    async def test_root_flips_true_before_threaded_processing_fails_closed(self, monkeypatch):
        _write_root_config(gateway_section=None)
        real_admit = proxy_admission.admit_local_api_agent_creation
        calls = 0

        def _flipping_admit():
            nonlocal calls
            calls += 1
            return real_admit()

        monkeypatch.setattr(proxy_admission, "admit_local_api_agent_creation", _flipping_admit)
        real_to_thread = asyncio.to_thread
        entered = False
        seam = _UnreachableSeam("transcribe_audio")

        async def _flip_at_worker_entry(func, *args, **kwargs):
            nonlocal entered
            _write_root_config(gateway_section={"proxy_required": True})
            entered = True
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", _flip_at_worker_entry)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", seam)
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body_text = await response.text()
        assert response.status == 403, body_text
        assert _DENIAL_MARKER in body_text
        assert entered
        assert calls >= 2
        assert seam.calls == []

    @pytest.mark.asyncio
    async def test_stt_flips_disabled_at_worker_entry(self, monkeypatch):
        _write_root_config(raw_yaml="stt:\n  enabled: true\n")
        real_to_thread = asyncio.to_thread
        seam = _UnreachableSeam("transcribe_audio")

        async def _flip_at_worker_entry(func, *args, **kwargs):
            _write_root_config(raw_yaml="stt:\n  enabled: false\n")
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", _flip_at_worker_entry)
        monkeypatch.setattr(transcription_tools, "transcribe_audio", seam)
        monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", _UnreachableSeam("local_fallback"))
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
        assert response.status == 503
        assert seam.calls == []

    @pytest.mark.asyncio
    async def test_root_flips_true_before_fallback_fails_closed(self, monkeypatch):
        _write_root_config(gateway_section=None)
        primary = _FailingTranscribe()
        fallback_calls = 0

        def _primary(file_path, model=None, source=None):
            _write_root_config(gateway_section={"proxy_required": True})
            return primary(file_path, model=model, source=source)

        def _fallback(file_path, model=None):
            nonlocal fallback_calls
            fallback_calls += 1
            return {"success": True, "transcript": "must not run"}

        monkeypatch.setattr(transcription_tools, "transcribe_audio", _primary)
        monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", _fallback)
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers())
            body_text = await response.text()
        assert response.status == 403, body_text
        assert _DENIAL_MARKER in body_text
        assert len(primary.calls) == 1
        assert fallback_calls == 0


# ---------------------------------------------------------------------------
# (6) Cancelled client request: live upload bytes and pending work survive until processing
# completes; cleanup only then; no pending-count underflow.
# ---------------------------------------------------------------------------


class TestCancellationLifecycle:
    @pytest.mark.asyncio
    async def test_cancelled_client_keeps_file_until_processing_completes(self, monkeypatch):
        _write_root_config(gateway_section=None)
        stub = _BlockingTranscribe()
        monkeypatch.setattr(transcription_tools, "transcribe_audio", stub)
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        async with TestClient(TestServer(app, handler_cancellation=True)) as cli:
            task = asyncio.ensure_future(
                cli.post(ENDPOINT, data=_multipart_form(), headers=_auth_headers()))
            try:
                started = await asyncio.to_thread(stub.started.wait, 5)
                assert started, "processing never reached the provider seam"
                assert stub.captured_path is not None
                assert stub.captured_bytes == _SYNTHETIC_AUDIO
                assert Path(stub.captured_path).exists()

                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

                assert Path(stub.captured_path).exists()
                assert adapter._pending_agent_requests == 1

                stub.release.set()
                finished = await asyncio.to_thread(stub.finished.wait, 5)
                assert finished, "background processing must still run to completion after cancellation"
                cleaned = await _wait_until(lambda: not Path(stub.captured_path).exists())
            finally:
                stub.release.set()
                if not task.done():
                    task.cancel()
        assert cleaned, "upload must be unlinked once processing ends"
        drained = await _wait_until(lambda: adapter._pending_agent_requests == 0)
        assert drained
        assert adapter._pending_agent_requests == 0

    @pytest.mark.asyncio
    async def test_body_read_barrier_reserves_pending_work(self, monkeypatch):
        _write_root_config(gateway_section=None)
        stub = _CapturingTranscribe()
        monkeypatch.setattr(transcription_tools, "transcribe_audio", stub)
        adapter = _make_adapter(_API_KEY)
        app = _create_app(adapter)
        boundary = "barrier-boundary"
        release = asyncio.Event()
        first_chunk_sent = asyncio.Event()

        async def chunks():
            yield f"--{boundary}\r\n".encode()
            first_chunk_sent.set()
            await release.wait()
            yield (
                b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
                b"Content-Type: audio/wav\r\n\r\n" + _SYNTHETIC_AUDIO +
                f"\r\n--{boundary}--\r\n".encode()
            )

        from aiohttp.payload import AsyncIterablePayload
        payload = AsyncIterablePayload(chunks(), content_type=f"multipart/form-data; boundary={boundary}")
        async with TestClient(TestServer(app)) as cli:
            task = asyncio.ensure_future(cli.post(ENDPOINT, data=payload, headers=_auth_headers()))
            try:
                await asyncio.wait_for(first_chunk_sent.wait(), timeout=5)
                assert await _wait_until(lambda: adapter._pending_agent_requests == 1)
                assert adapter.active_agent_work_count() == 1
                release.set()
                response = await task
                assert response.status == 200
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
        assert adapter._pending_agent_requests == 0


# ---------------------------------------------------------------------------
# (7) A -> B -> A across two profile homes: exact upload bytes reach the seam, and the generated
# temporary file lives only under the OWNING profile's home — never a sibling's.
# ---------------------------------------------------------------------------


class TestOwningProfileScopedBytesABA:
    @pytest.mark.asyncio
    async def test_exact_bytes_and_owning_home_across_two_profiles(
        self, monkeypatch, audio_profiles, audio_root_home, audio_adapter, audio_app,
    ):
        alice_key = _key("alice-audio-aba")
        bob_key = _key("bob-audio-aba")
        for name, key in (("alice", alice_key), ("bob", bob_key)):
            env_path = audio_profiles[name] / ".env"
            env_path.write_text(env_path.read_text(encoding="utf-8") + f"API_SERVER_KEY={key}\n",
                                 encoding="utf-8")

        captured: List[Dict[str, Any]] = []
        import hermes_constants
        from agent.secret_scope import get_secret

        def _capturing(file_path, model=None, source=None):
            data = Path(file_path).read_bytes()
            owner = Path(hermes_constants.get_hermes_home()).resolve()
            secret = get_secret("API_SERVER_KEY")
            transcript = f"transcript-for-{owner.name}-{secret}"
            captured.append({"file_path": file_path, "bytes": data, "owner": owner,
                             "secret": secret, "transcript": transcript})
            return {"success": True, "transcript": transcript, "provider": "local"}

        monkeypatch.setattr(transcription_tools, "transcribe_audio", _capturing)

        audio_root_home.joinpath("config.yaml").write_text("gateway: {}\n", encoding="utf-8")
        hermes_constants.pin_process_hermes_home(str(audio_root_home))
        try:
            async with TestClient(TestServer(audio_app)) as cli:
                payload_a1 = b"alice-payload-one-" + uuid.uuid4().bytes
                resp_a1 = await cli.post(
                    f"/p/alice{ENDPOINT}", data=_multipart_form(content=payload_a1),
                    headers={"Authorization": f"Bearer {alice_key}"})
                body_a1 = await resp_a1.json()
                assert resp_a1.status == 200, body_a1

                payload_b = b"bob-payload-" + uuid.uuid4().bytes
                resp_b = await cli.post(
                    f"/p/bob{ENDPOINT}", data=_multipart_form(content=payload_b),
                    headers={"Authorization": f"Bearer {bob_key}"})
                body_b = await resp_b.json()
                assert resp_b.status == 200, body_b

                payload_a2 = b"alice-payload-two-" + uuid.uuid4().bytes
                resp_a2 = await cli.post(
                    f"/p/alice{ENDPOINT}", data=_multipart_form(content=payload_a2),
                    headers={"Authorization": f"Bearer {alice_key}"})
                body_a2 = await resp_a2.json()
                assert resp_a2.status == 200, body_a2

                # bob's key must never authorize alice's endpoint or vice versa.
                cross = await cli.post(
                    f"/p/alice{ENDPOINT}", data=_multipart_form(content=payload_a1),
                    headers={"Authorization": f"Bearer {bob_key}"})
                assert cross.status == 401
        finally:
            hermes_constants.pin_process_hermes_home(None)

        assert len(captured) == 3
        assert captured[0]["bytes"] == payload_a1
        assert captured[1]["bytes"] == payload_b
        assert captured[2]["bytes"] == payload_a2
        alice_home = audio_profiles["alice"].resolve()
        bob_home = audio_profiles["bob"].resolve()
        assert [item["owner"] for item in captured] == [alice_home, bob_home, alice_home]
        assert [item["secret"] for item in captured] == [alice_key, bob_key, alice_key]
        assert [body_a1["text"], body_b["text"], body_a2["text"]] == [
            item["transcript"] for item in captured]
        assert Path(captured[0]["file_path"]).resolve().is_relative_to(alice_home)
        assert Path(captured[1]["file_path"]).resolve().is_relative_to(bob_home)
        assert Path(captured[2]["file_path"]).resolve().is_relative_to(alice_home)


class TestDeletedProfileWorkerRace:
    @pytest.mark.asyncio
    async def test_deleted_profile_is_not_resurrected_before_native_worker(self, monkeypatch,
                                                                            audio_profiles,
                                                                            audio_root_home,
                                                                            audio_app,
                                                                            tmp_path):
        import hermes_constants

        alice_key = _key("alice-audio-delete-race")
        alice_home = audio_profiles["alice"].resolve()
        assert alice_home.is_relative_to(tmp_path.resolve())
        alice_env = alice_home / ".env"
        alice_env.write_text(alice_env.read_text(encoding="utf-8") +
                             f"API_SERVER_KEY={alice_key}\n", encoding="utf-8")
        audio_root_home.joinpath("config.yaml").write_text("gateway: {}\n", encoding="utf-8")

        provider = _UnreachableSeam("transcribe_audio")
        fallback = _UnreachableSeam("local_fallback")
        monkeypatch.setattr(transcription_tools, "transcribe_audio", provider)
        monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", fallback)

        real_to_thread = asyncio.to_thread
        removed_before_worker = False

        async def _delete_before_worker(func, *args, **kwargs):
            nonlocal removed_before_worker
            if getattr(func, "__name__", "") == "_worker_process_audio":
                shutil.rmtree(alice_home)
                hermes_constants.mark_named_profile_deleted(alice_home)
                removed_before_worker = True
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", _delete_before_worker)
        previous_pin = hermes_constants._PINNED_PROCESS_HERMES_HOME
        hermes_constants.pin_process_hermes_home(str(audio_root_home))
        try:
            async with TestClient(TestServer(audio_app)) as cli:
                response = await cli.post(
                    f"/p/alice{ENDPOINT}", data=_multipart_form(),
                    headers={"Authorization": f"Bearer {alice_key}"})
                body_text = await response.text()
        finally:
            hermes_constants.pin_process_hermes_home(previous_pin)

        assert removed_before_worker
        assert response.status == 502
        assert str(alice_home) not in body_text
        assert "FileNotFoundError" not in body_text
        assert not provider.calls
        assert not fallback.calls
        assert not alice_home.exists()
        assert hermes_constants.named_profile_is_deleted(alice_home)
        assert not (alice_home / "cache").exists()
