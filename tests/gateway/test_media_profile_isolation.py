"""Regression coverage for profile isolation in outbound MEDIA path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.secret_scope import set_multiplex_active
from gateway import media_policy
from gateway.platforms import base
from gateway.platforms.base import BasePlatformAdapter, validate_media_delivery_path
from gateway.run import _profile_runtime_scope
from hermes_constants import get_hermes_home


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    ("active", "candidate", "expected"),
    [
        ("root", "root", True),
        ("root", "profile", False),
        ("profile", "profile", True),
        ("profile", "root", False),
        ("root", "profile_alias", False),
    ],
)
def test_media_delivery_never_crosses_profile_cache_boundary(
    tmp_path, monkeypatch, strict, active, candidate, expected
):
    """The active profile may deliver its cache, but not another profile's cache."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "b"
    root_cache = root / "cache" / "images"
    profile_cache = profile / "cache" / "images"
    root_cache.mkdir(parents=True)
    profile_cache.mkdir(parents=True)
    (root / "config.yaml").write_text("gateway: {}\n", encoding="utf-8")
    (profile / "config.yaml").write_text("gateway: {}\n", encoding="utf-8")

    root_file = root_cache / "root.png"
    profile_file = profile_cache / "profile.png"
    root_file.write_bytes(b"root-private-cache")
    profile_file.write_bytes(b"profile-private-cache")
    profile_alias = root_cache / "profile-alias.png"
    profile_alias.symlink_to(profile_file)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1" if strict else "0")
    # An operator root must not turn a sibling profile cache into an implicit shared root.
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(profile_cache))
    monkeypatch.setattr(base, "_HERMES_ROOT", root)
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (root_cache,))
    monkeypatch.setattr(media_policy, "media_delivery_strict", lambda: strict)
    monkeypatch.setattr(media_policy, "media_delivery_allow_dirs", lambda: str(profile_cache))
    monkeypatch.setattr("gateway.media_fetch.fetch_remote_media", lambda _path: None)

    paths = {
        "root": root_file,
        "profile": profile_file,
        "profile_alias": profile_alias,
    }
    active_home = root if active == "root" else profile

    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(root, prepared_secret_scope={}):
            assert get_hermes_home() == root
            with _profile_runtime_scope(profile, prepared_secret_scope={}):
                assert get_hermes_home() == profile
            assert get_hermes_home() == root

            with _profile_runtime_scope(active_home, prepared_secret_scope={}):
                assert get_hermes_home() == active_home
                media, cleaned = BasePlatformAdapter.extract_media(f"MEDIA:{paths[candidate]}")
                assert cleaned == ""
                filtered = BasePlatformAdapter.filter_media_delivery_paths(media)
                native = validate_media_delivery_path(str(paths[candidate]))

                assert bool(filtered) is expected
                assert (native is not None) is expected
                if expected:
                    assert filtered[0][0] == native
    finally:
        set_multiplex_active(False)


def _configure_media_policy(monkeypatch, root, *, strict=False, safe_roots=()):
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1" if strict else "0")
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", "")
    monkeypatch.setattr(base, "_HERMES_ROOT", root)
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", tuple(safe_roots))
    monkeypatch.setattr(media_policy, "media_delivery_strict", lambda: strict)
    monkeypatch.setattr(media_policy, "media_delivery_allow_dirs", lambda: "")
    monkeypatch.setattr("gateway.media_fetch.fetch_remote_media", lambda _path: None)


def test_named_profile_scope_a_b_a_rejects_opposite_profile_same_filename(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile_a = root / "profiles" / "a"
    profile_b = root / "profiles" / "b"
    file_a = profile_a / "cache" / "images" / "same.png"
    file_b = profile_b / "cache" / "images" / "same.png"
    file_a.parent.mkdir(parents=True)
    file_b.parent.mkdir(parents=True)
    file_a.write_bytes(b"a")
    file_b.write_bytes(b"b")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _configure_media_policy(monkeypatch, root)

    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(profile_a, prepared_secret_scope={}):
            assert validate_media_delivery_path(str(file_a)) == str(file_a)
            assert validate_media_delivery_path(str(file_b)) is None
        with _profile_runtime_scope(profile_b, prepared_secret_scope={}):
            assert validate_media_delivery_path(str(file_b)) == str(file_b)
            assert validate_media_delivery_path(str(file_a)) is None
        with _profile_runtime_scope(profile_a, prepared_secret_scope={}):
            assert validate_media_delivery_path(str(file_a)) == str(file_a)
            assert validate_media_delivery_path(str(file_b)) is None
    finally:
        set_multiplex_active(False)


def test_unscoped_multiplex_private_root_delivery_is_denied(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    media = root / "cache" / "images" / "private.png"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"private")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _configure_media_policy(monkeypatch, root, safe_roots=(root / "cache",))

    set_multiplex_active(True)
    try:
        assert validate_media_delivery_path(str(media)) is None
    finally:
        set_multiplex_active(False)


def test_non_multiplex_legacy_permissive_delivery_is_retained(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    media = tmp_path / "generated" / "result.png"
    media.parent.mkdir()
    media.write_bytes(b"result")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _configure_media_policy(monkeypatch, root)

    set_multiplex_active(False)
    assert validate_media_delivery_path(str(media)) == str(media)


def test_own_profile_credentials_are_rejected(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "a"
    credentials = profile / ".env"
    credentials.parent.mkdir(parents=True)
    credentials.write_text("TOKEN=secret\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _configure_media_policy(monkeypatch, root)

    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(profile, prepared_secret_scope={}):
            assert validate_media_delivery_path(str(credentials)) is None
    finally:
        set_multiplex_active(False)


def test_symlinked_profile_dir_outside_root_is_attributed_to_active_profile(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    external_profile = tmp_path / "profile-storage" / "b"
    media = external_profile / "cache" / "images" / "outside.png"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"outside")
    (root / "profiles").mkdir(parents=True)
    try:
        (root / "profiles" / "b").symlink_to(external_profile, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"profile symlinks unavailable: {exc}")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _configure_media_policy(monkeypatch, root)

    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(external_profile, prepared_secret_scope={}):
            assert validate_media_delivery_path(str(media)) == str(media)
        with _profile_runtime_scope(root, prepared_secret_scope={}):
            assert validate_media_delivery_path(str(media)) is None
    finally:
        set_multiplex_active(False)


def test_unavailable_profile_enumeration_does_not_assign_child_to_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    media = root / "profiles" / "b" / "cache" / "images" / "image.png"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"profile")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _configure_media_policy(monkeypatch, root, safe_roots=(root / "cache",))
    monkeypatch.setattr(base, "_profile_dirs", lambda: [])

    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(root, prepared_secret_scope={}):
            assert validate_media_delivery_path(str(media)) is None
    finally:
        set_multiplex_active(False)


def test_failed_root_resolution_denies_multiplex_media(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    media = root / "cache" / "images" / "image.png"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"root")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _configure_media_policy(monkeypatch, root, safe_roots=(root / "cache",))
    resolve_path = base._resolve_path

    def fail_root(path, *, strict=False, expand=False):
        if Path(path) == root:
            return None
        return resolve_path(path, strict=strict, expand=expand)

    monkeypatch.setattr(base, "_resolve_path", fail_root)
    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(root, prepared_secret_scope={}):
            assert validate_media_delivery_path(str(media)) is None
    finally:
        set_multiplex_active(False)
