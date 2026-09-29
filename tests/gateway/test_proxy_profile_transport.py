"""Native loopback coverage for one gateway proxy route per profile."""

from contextlib import asynccontextmanager
import base64
import json
from types import SimpleNamespace

import aiohttp.web
import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource, build_session_key


@asynccontextmanager
async def _server(status=200, response="fixture response"):
    requests = []

    async def completions(request):
        requests.append((request.path, dict(request.headers), await request.json()))
        if status != 200:
            return aiohttp.web.Response(status=status, text=response)
        body = (
            'data: {"choices":[{"delta":{"content":'
            f'{json.dumps(response)}'
            '}}]}\n\ndata: [DONE]\n\n'
        )
        # The fixture body is deliberately plain text; it is not a model/provider call.
        return aiohttp.web.Response(text=body, content_type="text/event-stream")

    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", requests
    finally:
        await runner.cleanup()


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._run_still_current_fn = lambda *args: (lambda: True)
    return runner


def _source():
    return SessionSource(
        platform=Platform.MATRIX, chat_id="!room:loopback", user_id="@user:loopback",
        user_name="fixture-user", chat_type="group",
    )


def _scope(home, url, key):
    return _profile_runtime_scope(
        home, prepared_secret_scope={"GATEWAY_PROXY_URL": url, "GATEWAY_PROXY_KEY": key}
    )


@pytest.mark.asyncio
async def test_profile_proxy_routes_a_b_a_with_scoped_auth_and_restoration(tmp_path, monkeypatch):
    """The real gateway dispatch changes proxy URL/key with each bound profile scope, and forwards
    each owner's native ``build_session_key`` conversation identity unchanged, independently of the
    rotating transcript ``X-Hermes-Session-Id``."""
    import run_agent

    def no_local_inference(*args, **kwargs):
        raise AssertionError("AIAgent must not be constructed by gateway proxy dispatch")

    monkeypatch.setattr(run_agent, "AIAgent", no_local_inference)
    runner = _runner()
    source = _source()
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir(); home_b.mkdir()
    # Distinct native profile-qualified conversation keys for the SAME source: proves the header
    # carries the per-owner memory identity, not something derived from source/session_id alone.
    key_a = build_session_key(source, profile="owner-a")
    key_b = build_session_key(source, profile="owner-b")
    assert key_a != key_b

    async with _server(response="fixture A") as (url_a, requests_a):
        async with _server(response="fixture B") as (url_b, requests_b):
            from agent.secret_scope import get_secret

            with _scope(home_a, url_a, "synthetic-key-A"):
                result_a1 = await runner._run_agent(
                    "current A1", "fixture system", [], source, "session-fixture", session_key=key_a,
                )
                with _scope(home_b, url_b, "synthetic-key-B"):
                    result_b = await runner._run_agent(
                        "current B", "fixture system", [], source, "session-fixture", session_key=key_b,
                    )
                    assert get_secret("GATEWAY_PROXY_KEY") == "synthetic-key-B"
                assert get_secret("GATEWAY_PROXY_KEY") == "synthetic-key-A"
                result_a2 = await runner._run_agent(
                    "current A2", "fixture system", [], source, "session-fixture-next", session_key=key_a,
                )
            results = [result_a1, result_b, result_a2]

    assert [result["final_response"] for result in results] == ["fixture A", "fixture B", "fixture A"]
    assert [item[0] for item in requests_a] == ["/v1/chat/completions", "/v1/chat/completions"]
    assert [item[0] for item in requests_b] == ["/v1/chat/completions"]
    for requests, key, session_key, messages, session_ids in (
        (requests_a, "synthetic-key-A", key_a, ["current A1", "current A2"],
         ["session-fixture", "session-fixture-next"]),
        (requests_b, "synthetic-key-B", key_b, ["current B"], ["session-fixture"]),
    ):
        for (_, headers, body), message, expected_session_id in zip(requests, messages, session_ids):
            assert headers["Authorization"] == f"Bearer {key}"
            # Transcript identity may rotate independently of the owner-qualified memory key.
            assert headers["X-Hermes-Session-Id"] == expected_session_id
            assert headers["X-Hermes-Session-Key"] == session_key
            assert body["messages"][-1] == {"role": "user", "content": message}


@pytest.mark.asyncio
async def test_profile_proxy_omits_session_key_header_when_absent(tmp_path):
    """Absent-key compatibility control: a caller that supplies no ``session_key`` must not have the
    gateway invent or coarsen one — ``X-Hermes-Session-Key`` stays absent from the outbound request."""
    runner = _runner()
    source = _source()
    home = tmp_path / "profile"
    home.mkdir()

    async with _server(response="fixture no-key") as (url, requests):
        with _scope(home, url, "synthetic-key"):
            result = await runner._run_agent(
                "current", "fixture system", [], source, "session-no-key",
            )

    assert result["final_response"] == "fixture no-key"
    assert len(requests) == 1
    _, headers, _ = requests[0]
    assert "X-Hermes-Session-Key" not in headers


@pytest.mark.asyncio
async def test_profile_proxy_503_does_not_fall_back_to_a_or_local(tmp_path, monkeypatch):
    import run_agent

    monkeypatch.setattr(
        run_agent, "AIAgent",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local inference fallback")),
    )
    runner = _runner()
    source = _source()
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir(); home_b.mkdir()

    async with _server(response="fixture A") as (url_a, requests_a):
        async with _server(503, "fixture B unavailable") as (url_b, requests_b):
            with _scope(home_b, url_b, "synthetic-key-B"):
                result = await runner._run_agent(
                    "current B", "fixture system", [], source, "session-503"
                )

    assert "503" in result["final_response"]
    assert "fixture B unavailable" in result["final_response"]
    assert len(requests_b) == 1
    assert requests_a == []


@pytest.mark.asyncio
async def test_proxy_preserves_buffered_native_image_data_url_and_session_isolation(tmp_path):
    runner = _runner()
    home = tmp_path / "profile"
    home.mkdir()
    # Valid 1x1 PNG: the native builder reads this file and emits its data URL.
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(tiny_png)
    image_b.write_bytes(tiny_png)
    source_a = _source()
    source_b = SessionSource(
        platform=Platform.MATRIX, chat_id="!other-room:loopback", user_id="@user:loopback",
        user_name="fixture-user", chat_type="group",
    )
    session_a = "session-image-a"
    session_b = "session-image-b"
    runner._session_state(session_a).persistent.native_image_paths = [str(image_a)]
    runner._session_state(session_b).persistent.native_image_paths = [str(image_b)]
    image_data_a = f"data:image/png;base64,{base64.b64encode(tiny_png).decode('ascii')}"

    async with _server(response="fixture image response") as (url, requests):
        with _scope(home, url, "synthetic-key"):
            result_a = await runner._run_agent(
                "look at this", "fixture system", [], source_a, "session-image", session_key=session_a,
            )
            assert runner._session_state(session_b).persistent.native_image_paths == [str(image_b)]
            result_b = await runner._run_agent(
                "look at this too", "fixture system", [], source_b, "session-image-b", session_key=session_b,
            )
            result_text = await runner._run_agent(
                "text only", "fixture system", [], source_a, "session-image", session_key=session_a,
            )

    assert [result["final_response"] for result in (result_a, result_b, result_text)] == [
        "fixture image response",
        "fixture image response",
        "fixture image response",
    ]
    content_a = requests[0][2]["messages"][-1]["content"]
    assert content_a == [
        {"type": "text", "text": f"look at this\n\n[Image attached at: {image_a}]"},
        {"type": "image_url", "image_url": {"url": image_data_a}},
    ]
    assert requests[1][2]["messages"][-1]["content"][1]["type"] == "image_url"
    assert requests[2][2]["messages"][-1] == {"role": "user", "content": "text only"}
    assert runner._session_state(session_a).persistent.native_image_paths == []
    assert runner._session_state(session_b).persistent.native_image_paths == []
