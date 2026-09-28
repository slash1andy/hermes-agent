"""Post-stream media delivery is explicit-only (#20834).

``GatewayRunner._deliver_media_from_response`` runs AFTER streaming has sent
the visible reply. At that point a bare local filesystem path in the response
text is either text the user already saw, or stale inspected/tool content —
it is NOT an attachment request. Only explicit ``MEDIA:`` directives may
trigger post-stream uploads.

The non-streaming path (``gateway/platforms/base.py``) keeps its bare-path
auto-detect (``extract_local_files``) — that path controls what text is sent
and can strip the path from the visible reply, so auto-attach is intentional
there. This file pins the asymmetry.
"""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse
from unittest.mock import AsyncMock

import pytest

from agent.secret_scope import (
    current_secret_scope,
    current_secret_scope_home,
    is_multiplex_active,
    set_multiplex_active,
)
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _event():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123CHAN",
        chat_type="group",
        thread_id=None,
    )
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=source,
        message_id="171.000001",
    )


def _fake_runner(thread_meta):
    return SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
        _media_delivery_scope_for_source=lambda source: nullcontext(),
    )


def _adapter():
    return SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="imgs")),
    )


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


@pytest.mark.asyncio
async def test_bare_local_path_in_streamed_reply_is_not_uploaded(tmp_path, monkeypatch):
    """The #20834 shape: visible reply contains a bare path (from inspected
    content), no MEDIA: directive — nothing may be uploaded post-stream."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "mockup.png")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        f"The design lives at {media_file} if you want to look later.",
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_document.assert_not_awaited()
    adapter.send_video.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_media_tag_still_delivers_post_stream(tmp_path, monkeypatch):
    """Explicit MEDIA: directives keep working after the #20834 fix."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "chart.png")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        f"Here is the chart.\nMEDIA:{media_file}",
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    images_kwargs = adapter.send_multiple_images.await_args.kwargs
    assert images_kwargs["chat_id"] == "C123CHAN"
    assert str(media_file) in images_kwargs["images"][0][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["a", "b"])
async def test_multiplexed_post_stream_media_uses_native_profile_scope(
    tmp_path, monkeypatch, profile,
):
    """A post-stream delivery call must re-enter the source profile before filtering media."""
    root = tmp_path / "hermes"
    profiles = root / "profiles"
    root.mkdir()
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    for name in ("a", "b"):
        home = profiles / name
        (home / "cache" / "images").mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "cache" / "images").mkdir(parents=True)

    own = profiles / profile / "cache" / "images" / "own-output.png"
    sibling = profiles / ("b" if profile == "a" else "a") / "cache" / "images" / "own-output.png"
    root_decoy = root / "cache" / "images" / "own-output.png"
    own.write_bytes(b"own-profile-bytes")
    sibling.write_bytes(b"sibling-profile-bytes")
    root_decoy.write_bytes(b"root-decoy-bytes")

    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("gateway.platforms.base._HERMES_ROOT", root)
    previous_multiplex = is_multiplex_active()
    previous_scope = current_secret_scope()
    previous_home = current_secret_scope_home()
    set_multiplex_active(True)
    adapter = _adapter()
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
    event = MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C123CHAN",
            chat_type="group",
            thread_id=None,
            profile=profile,
        ),
        message_id="171.000001",
    )

    try:
        # Deliberately no _profile_runtime_scope: this is the post-turn call site.
        await runner._deliver_media_from_response(
            f"MEDIA:{root_decoy}\nMEDIA:{sibling}\nMEDIA:{own}", event, adapter,
        )
    finally:
        set_multiplex_active(previous_multiplex)

    assert current_secret_scope() == previous_scope
    assert current_secret_scope_home() == previous_home
    adapter.send_multiple_images.assert_awaited_once()
    sent_uri = adapter.send_multiple_images.await_args.kwargs["images"][0][0]
    sent_path = unquote(urlparse(sent_uri).path)
    assert sent_path == str(own)
    assert Path(sent_path).read_bytes() == b"own-profile-bytes"
    assert str(sibling) not in sent_uri
    assert str(root_decoy) not in sent_uri


