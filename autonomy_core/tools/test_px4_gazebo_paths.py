from pathlib import Path

import pytest

from autonomy_core.tools import px4_gazebo_paths as paths


def _make_worlds_dir(px4_root: Path) -> Path:
    worlds_dir = px4_root / paths.PX4_WORLDS_RELATIVE
    worlds_dir.mkdir(parents=True)
    return worlds_dir


def test_explicit_worlds_directory_has_highest_priority(tmp_path):
    explicit = tmp_path / "explicit-worlds"
    explicit.mkdir()
    environment_root = tmp_path / "environment-px4"
    _make_worlds_dir(environment_root)

    resolved = paths.resolve_px4_worlds_dir(
        worlds_dir=explicit,
        environ={paths.PX4_ROOT_ENV: str(environment_root)},
    )

    assert resolved == explicit.resolve()


def test_environment_px4_root_resolves_worlds_directory(tmp_path):
    px4_root = tmp_path / "PX4-Autopilot"
    worlds_dir = _make_worlds_dir(px4_root)

    resolved = paths.resolve_px4_worlds_dir(
        environ={paths.PX4_ROOT_ENV: str(px4_root)},
    )

    assert resolved == worlds_dir.resolve()


def test_standard_and_nested_ubuntu_checkouts_are_discovered(tmp_path):
    nested_root = tmp_path / "PX4-Autopilot" / "PX4-Autopilot"
    nested_worlds = _make_worlds_dir(nested_root)

    resolved = paths.resolve_px4_worlds_dir(environ={}, home=tmp_path)

    assert resolved == nested_worlds.resolve()


def test_missing_worlds_directory_reports_configuration_options(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        paths.resolve_px4_worlds_dir(environ={}, home=tmp_path)

    message = str(exc_info.value)
    assert paths.PX4_ROOT_ENV in message
    assert paths.PX4_WORLDS_DIR_ENV in message
    assert str(tmp_path / "PX4-Autopilot") in message


def test_explicit_world_sdf_does_not_require_px4_checkout(tmp_path):
    world_sdf = tmp_path / "custom.sdf"
    world_sdf.write_text("<sdf/>", encoding="utf-8")

    resolved = paths.resolve_world_sdf(
        world_sdf=world_sdf,
        environ={},
        home=tmp_path / "missing-home",
    )

    assert resolved == world_sdf.resolve()


def test_world_name_uses_environment_and_strips_sdf_suffix(tmp_path):
    px4_root = tmp_path / "PX4-Autopilot"
    worlds_dir = _make_worlds_dir(px4_root)
    expected = worlds_dir / "portable_world.sdf"
    expected.write_text("<sdf/>", encoding="utf-8")

    resolved = paths.resolve_world_sdf(
        px4_root=px4_root,
        environ={paths.PX4_WORLD_ENV: "portable_world.sdf"},
    )

    assert resolved == expected.resolve()


def test_conflicting_explicit_roots_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="only one"):
        paths.resolve_px4_worlds_dir(
            worlds_dir=tmp_path / "worlds",
            px4_root=tmp_path / "PX4-Autopilot",
        )


@pytest.mark.parametrize("invalid_name", ("nested/world", r"nested\world", ".", ".."))
def test_world_name_rejects_paths(invalid_name):
    with pytest.raises(ValueError, match="name, not a path"):
        paths.configured_world_name(invalid_name, environ={})
