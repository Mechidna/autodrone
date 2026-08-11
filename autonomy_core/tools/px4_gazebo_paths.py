"""Portable discovery for PX4 Gazebo world assets.

Operational tools should accept an explicit path first, then consult the
environment, and only then inspect conventional Ubuntu checkout locations.
This module deliberately does not create directories or clone PX4.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


PX4_ROOT_ENV = "PX4_AUTOPILOT_ROOT"
PX4_WORLDS_DIR_ENV = "PX4_GZ_WORLDS_DIR"
PX4_WORLD_ENV = "PX4_GZ_WORLD"
WORLD_ENV = "WORLD"
PX4_WORLDS_RELATIVE = Path("Tools") / "simulation" / "gz" / "worlds"
DEFAULT_WORLD_NAME = "gate_test_1500mm_blue_random"


def expand_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(value)))).resolve()


def configured_world_name(
    world_name: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return a validated Gazebo world name from CLI input or the environment."""

    values = os.environ if environ is None else environ
    selected = world_name or values.get(PX4_WORLD_ENV) or values.get(WORLD_ENV)
    selected = str(selected or DEFAULT_WORLD_NAME).strip()
    if not selected:
        raise ValueError("Gazebo world name must not be empty.")
    if (
        Path(selected).name != selected
        or "/" in selected
        or "\\" in selected
        or selected in {".", ".."}
    ):
        raise ValueError(
            f"Gazebo world name must be a name, not a path: {selected!r}. "
            "Use --world-sdf for an explicit file."
        )
    if selected.endswith(".sdf"):
        selected = selected[:-4]
    if not selected:
        raise ValueError("Gazebo world name must not be empty.")
    return selected


def px4_worlds_dir_from_root(px4_root: str | os.PathLike[str]) -> Path:
    return expand_path(px4_root) / PX4_WORLDS_RELATIVE


def px4_root_from_worlds_dir(
    worlds_dir: str | os.PathLike[str],
) -> Path | None:
    worlds_path = expand_path(worlds_dir)
    relative_parts = PX4_WORLDS_RELATIVE.parts
    if worlds_path.parts[-len(relative_parts) :] != relative_parts:
        return None
    root = worlds_path
    for _part in relative_parts:
        root = root.parent
    return root


def resolve_px4_worlds_dir(
    *,
    worlds_dir: str | os.PathLike[str] | None = None,
    px4_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    home: str | os.PathLike[str] | None = None,
    require_exists: bool = True,
) -> Path:
    """Resolve the PX4 Gazebo worlds directory without machine-specific paths."""

    if worlds_dir is not None and px4_root is not None:
        raise ValueError("Specify only one of worlds_dir and px4_root.")

    values = os.environ if environ is None else environ
    candidates: list[Path] = []

    if worlds_dir is not None:
        candidates.append(expand_path(worlds_dir))
    elif px4_root is not None:
        candidates.append(px4_worlds_dir_from_root(px4_root))
    elif values.get(PX4_WORLDS_DIR_ENV):
        candidates.append(expand_path(values[PX4_WORLDS_DIR_ENV]))
    elif values.get(PX4_ROOT_ENV):
        candidates.append(px4_worlds_dir_from_root(values[PX4_ROOT_ENV]))
    else:
        home_path = expand_path(home) if home is not None else Path.home().resolve()
        candidates.extend(
            (
                home_path / "PX4-Autopilot" / PX4_WORLDS_RELATIVE,
                home_path
                / "PX4-Autopilot"
                / "PX4-Autopilot"
                / PX4_WORLDS_RELATIVE,
            )
        )

    unique_candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    for candidate in unique_candidates:
        if candidate.is_dir():
            return candidate

    if not require_exists:
        return unique_candidates[0]

    attempted = "\n  - ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(
        "PX4 Gazebo worlds directory was not found. Tried:\n"
        f"  - {attempted}\n"
        f"Set {PX4_ROOT_ENV}, set {PX4_WORLDS_DIR_ENV}, or pass an explicit path."
    )


def resolve_world_sdf(
    *,
    world_name: str | None = None,
    world_sdf: str | os.PathLike[str] | None = None,
    worlds_dir: str | os.PathLike[str] | None = None,
    px4_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    home: str | os.PathLike[str] | None = None,
    require_exists: bool = True,
) -> Path:
    """Resolve one world SDF, preserving an explicit file as highest priority."""

    if world_sdf is not None:
        selected = expand_path(world_sdf)
    else:
        selected_name = configured_world_name(world_name, environ=environ)
        selected = resolve_px4_worlds_dir(
            worlds_dir=worlds_dir,
            px4_root=px4_root,
            environ=environ,
            home=home,
            require_exists=require_exists,
        ) / f"{selected_name}.sdf"

    if require_exists and not selected.is_file():
        raise FileNotFoundError(
            f"Gazebo world SDF was not found: {selected}. "
            "Pass --world-sdf or verify the selected PX4 worlds directory."
        )
    return selected
