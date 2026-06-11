"""Import-safe guard helpers for future competition runtime adapters.

This module is intentionally passive. It does not wire itself into AutonomyAPI,
px4_runner, perception, logging, or Gazebo diagnostics. Future competition
adapters/runners must call these helpers before perception updates or
command-enabled starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


GAZEBO_TRUTH_POSE_SOURCE = "gazebo_truth_sim_only"

COMMAND_ENABLED_MODES = frozenset(
    {
        "command_live",
        "race",
    }
)

GAZEBO_TRUTH_KEY_PARTS = (
    "gazebo_pose",
    "gazebo_model",
    "gazebo_camera",
    "gazebo_truth",
    "latest_gazebo_pose",
)


class CompetitionGuardError(RuntimeError):
    """Raised when competition mode attempts to use Gazebo-only truth data."""


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, bytearray)):
        return len(value) > 0
    return True


def _path_join(parent: str, child: Any) -> str:
    key = str(child)
    return f"{parent}.{key}" if parent else key


def _assert_no_gazebo_truth_fields(value: Any, *, context: str, path: str = "") -> None:
    if value is None:
        return

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = _path_join(path, key)
            key_lower = str(key).lower()
            if any(part in key_lower for part in GAZEBO_TRUTH_KEY_PARTS):
                if _is_nonempty(child):
                    raise CompetitionGuardError(
                        f"{context} contains Gazebo truth field {child_path!r}"
                    )
                continue
            _assert_no_gazebo_truth_fields(child, context=context, path=child_path)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        for index, child in enumerate(value):
            _assert_no_gazebo_truth_fields(
                child,
                context=context,
                path=_path_join(path, index),
            )
        return

    if isinstance(value, str) and value == GAZEBO_TRUTH_POSE_SOURCE:
        raise CompetitionGuardError(
            f"{context} contains forbidden pose source {GAZEBO_TRUTH_POSE_SOURCE!r}"
        )


@dataclass(frozen=True)
class CompetitionGuard:
    """Small competition-mode guard for future adapter/runnable boundaries."""

    competition_mode: bool = True

    def assert_perception_world_pose_source(
        self,
        perception_world_pose_source: Any,
        *,
        context: str = "competition runtime",
    ) -> None:
        if not self.competition_mode:
            return

        if str(perception_world_pose_source) == GAZEBO_TRUTH_POSE_SOURCE:
            raise CompetitionGuardError(
                f"{context} cannot use perception_world_pose_source="
                f"{GAZEBO_TRUTH_POSE_SOURCE!r}"
            )

    def assert_no_gazebo_pose(
        self,
        gazebo_pose: Any,
        *,
        context: str = "competition perception update",
    ) -> None:
        if not self.competition_mode:
            return

        if gazebo_pose is not None:
            raise CompetitionGuardError(f"{context} must pass gazebo_pose=None")

    def assert_no_gazebo_truth_fields(
        self,
        value: Any,
        *,
        context: str = "competition runtime input",
    ) -> None:
        if not self.competition_mode:
            return

        _assert_no_gazebo_truth_fields(value, context=context)

    def assert_competition_safe(
        self,
        *,
        perception_world_pose_source: Any = None,
        gazebo_pose: Any = None,
        image_metadata: Any = None,
        runner_inputs: Any = None,
        mode: str = "",
        command_enabled: bool = False,
    ) -> None:
        """Validate future competition adapter start/update inputs."""

        if not self.competition_mode:
            return

        if perception_world_pose_source is not None:
            self.assert_perception_world_pose_source(
                perception_world_pose_source,
                context="competition runtime",
            )
        self.assert_no_gazebo_pose(gazebo_pose)
        self.assert_no_gazebo_truth_fields(
            image_metadata,
            context="competition image metadata",
        )
        self.assert_no_gazebo_truth_fields(
            runner_inputs,
            context="competition runner inputs",
        )

        if command_enabled or str(mode) in COMMAND_ENABLED_MODES:
            if str(perception_world_pose_source) == GAZEBO_TRUTH_POSE_SOURCE:
                raise CompetitionGuardError(
                    "command-enabled competition mode cannot start with "
                    f"{GAZEBO_TRUTH_POSE_SOURCE!r}"
                )

    def perception_update_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Return perception kwargs with a guaranteed competition-safe gazebo_pose."""

        self.assert_no_gazebo_pose(
            kwargs.get("gazebo_pose"),
            context="competition perception update kwargs",
        )
        self.assert_no_gazebo_truth_fields(
            kwargs.get("image_pose_snapshot"),
            context="competition image_pose_snapshot",
        )
        safe_kwargs = dict(kwargs)
        safe_kwargs["gazebo_pose"] = None
        return safe_kwargs


__all__ = [
    "COMMAND_ENABLED_MODES",
    "CompetitionGuard",
    "CompetitionGuardError",
    "GAZEBO_TRUTH_POSE_SOURCE",
    "GAZEBO_TRUTH_KEY_PARTS",
]
