"""Own-profile local image validation + native content-part building for ``/bg`` proxy dispatch
(``GatewayRunner._run_background_task_inner``). Kept out of ``run_turn.py`` — pre-request media
policy for one narrow path, not general turn execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

_SUPPORTED_IMAGE_PREFIX = "image/"


def validate_background_media(media_urls: List[str], media_types: List[str]) -> Optional[str]:
    """Deny reason (a short user-facing string) if ``media_urls``/``media_types`` must not be
    forwarded, else ``None``.

    Own-profile local images only: the two arrays must align, every declared type must be a
    supported image MIME, and every path must resolve to an existing file inside this process's
    own Hermes home — a sibling profile's file is refused exactly like a missing one. Runs
    entirely before any HTTP call and before the native content builder touches a file.
    """
    if len(media_urls) != len(media_types):
        return "media attachment count does not match declared types"

    from hermes_constants import get_hermes_home
    own_home = Path(get_hermes_home()).resolve()

    for raw_path, media_type in zip(media_urls, media_types):
        if not str(media_type or "").startswith(_SUPPORTED_IMAGE_PREFIX):
            return "unsupported media type"
        try:
            resolved = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return "attached image could not be read"
        if resolved != own_home and own_home not in resolved.parents:
            return "attached image is not owned by this profile"

    from gateway.platforms.base import BasePlatformAdapter
    checked = BasePlatformAdapter.filter_media_delivery_paths([(url, False) for url in media_urls])
    if len(checked) != len(media_urls):
        return "attached image failed delivery-path validation"

    return None


def build_background_native_content(prompt: str, image_paths: List[str]) -> Optional[list]:
    """Native OpenAI-style ``content`` parts for ``prompt`` + local ``image_paths`` (real
    ``agent.image_routing.build_native_content_parts`` — no local vision inference, no custom
    encoding). The builder's generated text part carries local-filesystem hints (``[Image
    attached at: <path>]``); that part is replaced with the original prompt (or a neutral image
    prompt when empty) so the wire body never carries a local path. Returns ``None`` when the
    builder skipped any path — an attachment must never be silently dropped.
    """
    from agent.image_routing import build_native_content_parts

    parts, skipped = build_native_content_parts(prompt, image_paths)
    if skipped:
        return None
    text = (prompt or "").strip() or "What do you see in this image?"
    return [
        {"type": "text", "text": text} if part.get("type") == "text" else part
        for part in parts
    ]
