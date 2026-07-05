"""Competition-safe AutonomyAPI construction profile.

Phase 9A keeps the existing `autonomy_api6.AutonomyAPI` implementation, but
prevents competition code from using legacy Gazebo-truth defaults directly.
Importing this module does not import or instantiate AutonomyAPI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from autonomy_core.runtime.competition_guard import (
    CompetitionGuard,
    CompetitionGuardError,
    GAZEBO_TRUTH_POSE_SOURCE,
)


COMPETITION_SAFE_POSE_SOURCE = "mavsdk"
COMPETITION_AUTONOMY_PROFILE_NAME = "competition_safe_autonomy_profile"
COMPETITION_OFFICIAL_TRANSFORM_MODE = "competition_official_ned"

_CONTROLLED_AUTONOMY_KWARGS = frozenset(
    {
        "use_perception",
        "race_gate_count",
        "race_gate_order",
        "save_perception_debug_frames",
        "use_lookahead_gate_filter",
    }
)

_FORBIDDEN_EXTRA_KWARG_PARTS = (
    "gazebo",
    "truth",
    "pose_snapshot",
    "image_pose_snapshot",
    "use_diagnostic_far_depth_correction",
    "perception_world_pose_source",
    "perception_transform_mode",
)


class CompetitionAutonomyProfileError(RuntimeError):
    """Raised when AutonomyAPI would be constructed with unsafe defaults."""


@dataclass(frozen=True)
class CompetitionAutonomyProfile:
    """Safe construction policy for the existing AutonomyAPI facade."""

    use_perception: bool = False
    race_gate_count: Optional[int] = None
    race_gate_order: Optional[Sequence[Any]] = None
    save_perception_debug_frames: bool = False
    use_lookahead_gate_filter: bool = True
    perception_world_pose_source: str = COMPETITION_SAFE_POSE_SOURCE
    perception_transform_mode: str = COMPETITION_OFFICIAL_TRANSFORM_MODE
    use_diagnostic_far_depth_correction: bool = False
    allow_legacy_yolo_default: bool = False
    profile_name: str = COMPETITION_AUTONOMY_PROFILE_NAME
    extra_autonomy_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def constructor_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by the current AutonomyAPI constructor."""

        _validate_profile(self)
        kwargs = {
            "use_perception": bool(self.use_perception),
            "race_gate_count": self.race_gate_count,
            "race_gate_order": self.race_gate_order,
            "save_perception_debug_frames": False,
            "use_lookahead_gate_filter": bool(self.use_lookahead_gate_filter),
        }
        kwargs.update(dict(self.extra_autonomy_kwargs))
        return kwargs


def create_competition_autonomy_api(
    *,
    profile: CompetitionAutonomyProfile = CompetitionAutonomyProfile(),
    autonomy_factory: Optional[Callable[..., Any]] = None,
    autonomy_kwargs: Optional[Mapping[str, Any]] = None,
    guard: Optional[CompetitionGuard] = None,
) -> Any:
    """Create and harden AutonomyAPI only when explicitly requested."""

    active_profile = _profile_with_extra_kwargs(profile, autonomy_kwargs)
    kwargs = active_profile.constructor_kwargs()
    factory = autonomy_factory if autonomy_factory is not None else _load_autonomy_api
    autonomy = factory(**kwargs)
    apply_competition_autonomy_profile(
        autonomy,
        profile=active_profile,
        guard=guard,
    )
    validate_competition_autonomy_api(
        autonomy,
        profile=active_profile,
        guard=guard,
    )
    return autonomy


def apply_competition_autonomy_profile(
    autonomy: Any,
    *,
    profile: CompetitionAutonomyProfile = CompetitionAutonomyProfile(),
    guard: Optional[CompetitionGuard] = None,
) -> Any:
    """Apply competition-safe runtime fields to an AutonomyAPI-like object."""

    _validate_profile(profile)
    active_guard = guard or CompetitionGuard()
    try:
        active_guard.assert_perception_world_pose_source(
            profile.perception_world_pose_source,
            context="competition AutonomyAPI profile",
        )
    except CompetitionGuardError as exc:
        raise CompetitionAutonomyProfileError(str(exc)) from exc

    setattr(autonomy, "perception_world_pose_source", profile.perception_world_pose_source)
    setattr(autonomy, "perception_world_pose_source_used", profile.perception_world_pose_source)
    setattr(autonomy, "perception_transform_mode", profile.perception_transform_mode)
    setattr(autonomy, "save_perception_debug_frames", False)
    setattr(autonomy, "use_diagnostic_far_depth_correction", False)
    setattr(autonomy, "image_gazebo_pose_snapshot", None)
    setattr(autonomy, "competition_autonomy_profile_active", True)
    setattr(autonomy, "competition_autonomy_profile_name", profile.profile_name)
    setattr(
        autonomy,
        "competition_autonomy_profile_notes",
        (
            "Gazebo truth is disabled for competition mode.",
            "Perception updates must pass gazebo_pose=None.",
            "Perception updates must pass image_pose_snapshot=None.",
            f"Perception transform mode is {profile.perception_transform_mode}.",
            "Command publication is controlled outside AutonomyAPI.",
        ),
    )
    setattr(
        autonomy,
        "competition_yolo_config_source",
        (
            "legacy_autonomyapi_default_explicitly_acknowledged"
            if profile.use_perception and profile.allow_legacy_yolo_default
            else "not_loaded_in_profile"
        ),
    )
    print_startup = getattr(autonomy, "print_perception_transform_startup", None)
    if profile.use_perception and callable(print_startup):
        print_startup()
    return autonomy


def validate_competition_autonomy_api(
    autonomy: Any,
    *,
    profile: CompetitionAutonomyProfile = CompetitionAutonomyProfile(),
    guard: Optional[CompetitionGuard] = None,
) -> None:
    """Validate that an AutonomyAPI-like object is competition-safe."""

    _validate_profile(profile)
    active_guard = guard or CompetitionGuard()
    source = getattr(autonomy, "perception_world_pose_source", None)
    try:
        active_guard.assert_competition_safe(
            perception_world_pose_source=source,
            runner_inputs={
                "use_diagnostic_far_depth_correction": bool(
                    getattr(autonomy, "use_diagnostic_far_depth_correction", False)
                ),
            },
            command_enabled=False,
        )
    except CompetitionGuardError as exc:
        raise CompetitionAutonomyProfileError(str(exc)) from exc

    if str(source) == GAZEBO_TRUTH_POSE_SOURCE:
        raise CompetitionAutonomyProfileError(
            "competition AutonomyAPI profile cannot use gazebo_truth_sim_only"
        )
    if bool(getattr(autonomy, "save_perception_debug_frames", False)):
        raise CompetitionAutonomyProfileError(
            "competition AutonomyAPI profile must disable debug-frame writes"
        )
    if bool(getattr(autonomy, "use_diagnostic_far_depth_correction", False)):
        raise CompetitionAutonomyProfileError(
            "competition AutonomyAPI profile must disable far-depth correction"
        )
    if getattr(autonomy, "image_gazebo_pose_snapshot", None) is not None:
        raise CompetitionAutonomyProfileError(
            "competition AutonomyAPI profile must clear image_gazebo_pose_snapshot"
        )
    transform_mode = getattr(autonomy, "perception_transform_mode", None)
    if str(transform_mode) != profile.perception_transform_mode:
        raise CompetitionAutonomyProfileError(
            "competition AutonomyAPI profile must use "
            f"perception_transform_mode={profile.perception_transform_mode!r}"
        )


def _profile_with_extra_kwargs(
    profile: CompetitionAutonomyProfile,
    autonomy_kwargs: Optional[Mapping[str, Any]],
) -> CompetitionAutonomyProfile:
    if not autonomy_kwargs:
        return profile
    combined = {**dict(profile.extra_autonomy_kwargs), **dict(autonomy_kwargs)}
    return CompetitionAutonomyProfile(
        use_perception=profile.use_perception,
        race_gate_count=profile.race_gate_count,
        race_gate_order=profile.race_gate_order,
        save_perception_debug_frames=profile.save_perception_debug_frames,
        use_lookahead_gate_filter=profile.use_lookahead_gate_filter,
        perception_world_pose_source=profile.perception_world_pose_source,
        perception_transform_mode=profile.perception_transform_mode,
        use_diagnostic_far_depth_correction=profile.use_diagnostic_far_depth_correction,
        allow_legacy_yolo_default=profile.allow_legacy_yolo_default,
        profile_name=profile.profile_name,
        extra_autonomy_kwargs=combined,
    )


def _validate_profile(profile: CompetitionAutonomyProfile) -> None:
    guard = CompetitionGuard()
    try:
        guard.assert_perception_world_pose_source(
            profile.perception_world_pose_source,
            context="competition AutonomyAPI profile",
        )
    except CompetitionGuardError as exc:
        raise CompetitionAutonomyProfileError(str(exc)) from exc

    if bool(profile.save_perception_debug_frames):
        raise CompetitionAutonomyProfileError(
            "competition AutonomyAPI profile requires save_perception_debug_frames=False"
        )
    if bool(profile.use_diagnostic_far_depth_correction):
        raise CompetitionAutonomyProfileError(
            "competition AutonomyAPI profile requires "
            "use_diagnostic_far_depth_correction=False"
        )
    if str(profile.perception_transform_mode) != COMPETITION_OFFICIAL_TRANSFORM_MODE:
        raise CompetitionAutonomyProfileError(
            "competition AutonomyAPI profile requires "
            f"perception_transform_mode={COMPETITION_OFFICIAL_TRANSFORM_MODE!r}"
        )
    if bool(profile.use_perception) and not bool(profile.allow_legacy_yolo_default):
        raise CompetitionAutonomyProfileError(
            "real perception requires explicit YOLO configuration. The current "
            "AutonomyAPI has a legacy hardcoded YOLO weights path; set "
            "allow_legacy_yolo_default=True only for an explicit temporary "
            "dry-run acknowledgment, or add a yolo-model-path-aware factory."
        )

    extra = dict(profile.extra_autonomy_kwargs)
    controlled = sorted(_CONTROLLED_AUTONOMY_KWARGS.intersection(extra))
    if controlled:
        raise CompetitionAutonomyProfileError(
            "competition AutonomyAPI profile owns constructor fields: "
            + ", ".join(controlled)
        )
    for key, value in extra.items():
        key_lower = str(key).lower()
        if any(part in key_lower for part in _FORBIDDEN_EXTRA_KWARG_PARTS):
            if value not in (None, False, ""):
                raise CompetitionAutonomyProfileError(
                    f"forbidden competition AutonomyAPI extra kwarg: {key!r}"
                )


def _load_autonomy_api(**kwargs: Any) -> Any:
    from autonomy_core.launch.autonomy_api6 import AutonomyAPI

    return AutonomyAPI(**kwargs)


__all__ = [
    "COMPETITION_AUTONOMY_PROFILE_NAME",
    "COMPETITION_OFFICIAL_TRANSFORM_MODE",
    "COMPETITION_SAFE_POSE_SOURCE",
    "CompetitionAutonomyProfile",
    "CompetitionAutonomyProfileError",
    "apply_competition_autonomy_profile",
    "create_competition_autonomy_api",
    "validate_competition_autonomy_api",
]
