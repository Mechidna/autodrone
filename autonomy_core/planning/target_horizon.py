"""Planning horizon helpers used by AutonomyAPI wrappers."""

import time

import numpy as np


def _append_planning_lookahead_targets(
    api,
    current_pos,
    target_gates,
    target_track_ids,
    target_waypoint_types,
    max_gates_ahead,
    allow_raw_candidates,
):
    api.append_lookahead_called = True
    api.append_lookahead_input_track_ids = [
        int(getattr(tr, "id", -1))
        for tr in sorted(api.gate_memory.tracks, key=lambda tr: tr.id)
    ]
    api.append_lookahead_selected_track_ids = []
    api.append_lookahead_selected_centers = ""
    api.append_lookahead_selected_types = ""
    initial_selected_count = len(target_track_ids)
    if not api.use_planning_lookahead_tracks:
        api.tentative_lookahead_rejection_reason = "append_disabled"
        return target_gates, target_track_ids, target_waypoint_types

    remaining = int(max_gates_ahead) - len(target_gates)
    if remaining <= 0:
        api.tentative_lookahead_rejection_reason = "append_no_remaining_slots"
        return target_gates, target_track_ids, target_waypoint_types

    now = time.time()
    api.prune_raw_planning_lookahead_candidates(now=now)
    selected_ids = {tid for tid in target_track_ids if tid is not None and tid >= 0}
    existing_points = [np.asarray(g, dtype=float).reshape(3) for g in target_gates]
    api.planning_lookahead_track_ids = []
    lookahead_sources = []

    all_tracks = sorted(api.gate_memory.tracks, key=lambda tr: tr.id)
    for tr in all_tracks:
        if remaining <= 0:
            break
        track_id = api.canonical_track_id(tr.id)
        if track_id is None or track_id in selected_ids:
            if track_id is not None and track_id in selected_ids:
                api.horizon_track_decisions.setdefault(
                    track_id, "included:hard_current_or_stable"
                )
            continue
        if track_id in api.completed_track_ids_this_cycle:
            api.horizon_rejected_track_ids.append(track_id)
            api.horizon_rejection_reason = "completed_this_lap"
            api.horizon_track_decisions[track_id] = "excluded:completed_this_lap"
            continue

        is_hard_lookahead = bool(getattr(tr, "is_stable", False))
        is_committed_unstable = bool(
            getattr(tr, "committed", False)
            and not getattr(tr, "is_stable", False)
        )
        if is_hard_lookahead:
            center_source = getattr(tr, "center", None)
        else:
            center_source = getattr(tr, "filtered_center_world", None)
            if center_source is None:
                center_source = getattr(tr, "center", None)
        if center_source is None:
            api.horizon_rejected_track_ids.append(track_id)
            api.horizon_rejection_reason = "lookahead_missing_center"
            api.horizon_track_decisions[track_id] = "excluded:lookahead_missing_center"
            continue
        center, reason = api._planning_target_safe_center(center_source)
        if center is None:
            api.horizon_rejected_track_ids.append(track_id)
            api.horizon_rejection_reason = reason
            api.horizon_track_decisions[track_id] = f"excluded:{reason}"
            continue
        waypoint_type = "hard_stable"
        if is_hard_lookahead:
            if tr.hits < api.planning_lookahead_min_hits:
                api.horizon_rejected_track_ids.append(track_id)
                api.horizon_rejection_reason = "lookahead_insufficient_hits"
                api.horizon_track_decisions[track_id] = "excluded:lookahead_insufficient_hits"
                continue
            if api.is_near_completed_gate(center):
                api.horizon_rejected_track_ids.append(track_id)
                api.horizon_rejection_reason = "lookahead_completed_this_lap"
                api.horizon_track_decisions[track_id] = "excluded:lookahead_completed_this_lap"
                continue
            duplicate_radius = max(
                float(api.gate_memory.commit_radius),
                float(getattr(api.gate_memory, "duplicate_merge_radius", 0.0)),
            )
            if any(float(np.linalg.norm(center - p)) < duplicate_radius for p in existing_points):
                api.horizon_rejected_track_ids.append(track_id)
                api.horizon_rejection_reason = "lookahead_duplicate_selected"
                api.horizon_track_decisions[track_id] = "excluded:lookahead_duplicate_selected"
                continue
            if float(np.linalg.norm(center - current_pos)) > api.max_detection_range:
                api.horizon_rejected_track_ids.append(track_id)
                api.horizon_rejection_reason = "lookahead_too_far"
                api.horizon_track_decisions[track_id] = "excluded:lookahead_too_far"
                continue
        else:
            reason = api._tentative_lookahead_rejection(
                tr=tr,
                track_id=track_id,
                center=center,
                current_pos=current_pos,
                selected_ids=selected_ids,
                existing_points=existing_points,
            )
            if reason:
                api.horizon_rejected_track_ids.append(track_id)
                api.horizon_rejection_reason = reason
                api.tentative_lookahead_rejection_reason = reason
                api.horizon_track_decisions[track_id] = f"excluded:{reason}"
                continue
            waypoint_type = (
                "soft_committed_unstable"
                if is_committed_unstable
                else "soft_tentative"
            )

        target_gates.append(center.copy())
        target_track_ids.append(track_id)
        target_waypoint_types.append(waypoint_type)
        selected_ids.add(track_id)
        existing_points.append(center.copy())
        api.planning_lookahead_track_ids.append(track_id)
        api.horizon_track_decisions[track_id] = f"included:{waypoint_type}"
        if waypoint_type in ("soft_committed_unstable", "soft_tentative"):
            api.tentative_lookahead_used = True
            api.tentative_lookahead_track_ids.append(track_id)
            lookahead_sources.append(waypoint_type)
        else:
            lookahead_sources.append("hard_stable")
        remaining -= 1

    if allow_raw_candidates and remaining > 0:
        for candidate in api.raw_planning_lookahead_candidates:
            if remaining <= 0:
                break
            center, reason = api._planning_target_safe_center(candidate["center"])
            if center is None:
                api.horizon_rejection_reason = reason
                continue
            if api.is_near_completed_gate(center):
                api.horizon_rejection_reason = "raw_lookahead_completed_this_lap"
                continue
            duplicate_radius = max(
                float(api.gate_memory.commit_radius),
                float(getattr(api.gate_memory, "duplicate_merge_radius", 0.0)),
            )
            if any(float(np.linalg.norm(center - p)) < duplicate_radius for p in existing_points):
                api.horizon_rejection_reason = "raw_lookahead_duplicate_selected"
                continue
            if float(np.linalg.norm(center - current_pos)) > api.max_detection_range:
                api.horizon_rejection_reason = "raw_lookahead_too_far"
                continue

            target_gates.append(center.copy())
            target_track_ids.append(-1)
            target_waypoint_types.append("soft_tentative")
            existing_points.append(center.copy())
            lookahead_sources.append("raw_rejected_clamped")
            remaining -= 1

    appended_ids = target_track_ids[initial_selected_count:]
    appended_gates = target_gates[initial_selected_count:]
    appended_types = target_waypoint_types[initial_selected_count:]
    api.append_lookahead_selected_track_ids = list(appended_ids)
    api.append_lookahead_selected_centers = ";".join(
        f"{track_id}:{gate[0]:.2f},{gate[1]:.2f},{gate[2]:.2f}"
        for gate, track_id in zip(appended_gates, appended_ids)
    )
    api.append_lookahead_selected_types = " ".join(str(t) for t in appended_types)
    if (
        int(getattr(api, "tentative_lookahead_eligible_count", 0)) > 0
        and not api.tentative_lookahead_used
        and not api.tentative_lookahead_rejection_reason
    ):
        api.tentative_lookahead_rejection_reason = (
            api.horizon_rejection_reason
            or "eligible_tentative_not_selected"
        )
    api.planning_lookahead_used = len(lookahead_sources) > 0
    api.planning_lookahead_source = " ".join(lookahead_sources)
    return target_gates, target_track_ids, target_waypoint_types


def finalize_planning_horizon_debug(api):
    selected = {
        api.canonical_track_id(tid)
        for tid in api.active_target_track_ids + api.planning_lookahead_track_ids
        if tid is not None
    }
    parts = []
    for tr in sorted(api.gate_memory.tracks, key=lambda item: item.id):
        tid = api.canonical_track_id(tr.id)
        center = getattr(tr, "filtered_center_world", None)
        if center is None:
            center = getattr(tr, "center", np.full(3, np.nan))
        center = np.asarray(center, dtype=float).reshape(3)
        state = "stable" if getattr(tr, "is_stable", False) else "tentative"
        decision = api.horizon_track_decisions.get(tid)
        if decision is None:
            decision = (
                "included:selected_horizon"
                if tid in selected
                else "excluded:not_selected_by_race_or_lookahead_policy"
            )
        parts.append(
            f"track{tid}:state={state},center={center[0]:.2f}/{center[1]:.2f}/{center[2]:.2f},"
            f"decision={decision}"
        )
    api.planning_track_horizon_debug = ";".join(parts)
    active_center = (
        np.asarray(api.active_target_center, dtype=float).reshape(3)
        if api.active_target_center is not None
        else np.full(3, np.nan)
    )
    api.planning_cycle_debug = (
        f"gate_idx={api.current_gate_idx},active_track={api.active_target_track_id},"
        f"active={active_center[0]:.2f}/{active_center[1]:.2f}/{active_center[2]:.2f},"
        f"lookahead_ids={'/'.join(map(str,api.planning_lookahead_track_ids))},"
        f"lookahead_centers={api.append_lookahead_selected_centers},"
        f"types={api.planning_horizon_waypoint_types},"
        f"tracks={api.planning_track_horizon_debug}"
    )
    print("[PLANNING FLOW] " + api.planning_cycle_debug)


def build_waypoint_horizon_from_memory(api, current_pos, max_gates_ahead=3):
    """
    Build planning horizon from explicit race progression.

    No spatial coordinate is used to order gates. A predefined race order
    wins if supplied; otherwise committed track IDs are appended in
    discovery order and the progression cursor advances through that list.

    Returns:
        waypoints: np.ndarray shape (N,3)
        target_gates: list[np.ndarray]
        target_track_ids: list[int]
    """
    committed_tracks = api.refresh_race_order_from_memory()
    api.race_progression.update_clearance(current_pos, api.gate_memory.get_track_by_id)
    api.target_rejected_completed = False
    api.rejected_wrong_order = False
    api.rejected_duplicate = False
    api.rejected_completed_this_lap = False
    api.candidate_track_id = None
    api.candidate_center = None
    api.candidate_order_score = float("nan")
    api.lap_reset_triggered = False
    api.next_valid_target_found = False
    api.valid_candidate_count = 0
    api.selected_next_gate_track_id = None
    api.selected_next_gate_stability_score = float("nan")
    api.selected_target_source = ""
    api.horizon_build_cursor = api.race_progression.cursor
    api.horizon_available_order = []
    api.horizon_selected_track_ids = []
    api.horizon_rejected_track_ids = []
    api.horizon_rejection_reason = ""
    api.planning_lookahead_track_ids = []
    api.planning_lookahead_source = ""
    api.planning_lookahead_used = False
    api.tentative_lookahead_used = False
    api.tentative_lookahead_track_ids = []
    api.tentative_lookahead_centers = ""
    api.tentative_lookahead_rejection_reason = ""
    api.append_lookahead_called = False
    api.append_lookahead_input_track_ids = []
    api.append_lookahead_selected_track_ids = []
    api.append_lookahead_selected_centers = ""
    api.append_lookahead_selected_types = ""
    api.horizon_track_decisions = {}
    api.planning_track_horizon_debug = ""
    api.planning_cycle_debug = ""

    if len(committed_tracks) == 0:
        api.planning_horizon_waypoint_types = "start"
        api._planning_target_waypoint_types = []
        return np.array([current_pos], dtype=float), [], []

    order = api.race_progression.order()
    api.horizon_available_order = list(order)
    target_tracks = []
    target_center_overrides = {}
    selected_track_ids = set()
    first_selected_order_idx = None

    if (
        api.race_progression.cursor < len(order)
        and api.selected_current_track_id is not None
        and api.canonical_track_id(order[api.race_progression.cursor])
        != api.canonical_track_id(api.selected_current_track_id)
    ):
        blocked_id = api.canonical_track_id(order[api.race_progression.cursor])
        api.horizon_rejected_track_ids.append(blocked_id)
        api.horizon_rejection_reason = "farther_future_track_cannot_be_hard_current"
        api.current_selection_rejection_reason = (
            f"track{blocked_id}:farther_future_track_cannot_be_hard_current;"
            f"selected_current_track={api.selected_current_track_id}"
        )
        api.horizon_track_decisions[blocked_id] = (
            "excluded:farther_future_track_cannot_be_hard_current"
        )
        api.planning_horizon_waypoint_types = "start"
        api._planning_target_waypoint_types = []
        return np.array([current_pos], dtype=float), [], []

    for order_idx in range(api.race_progression.cursor, len(order)):
        if len(target_tracks) >= max_gates_ahead:
            break

        track_id = order[order_idx]
        if track_id in selected_track_ids:
            continue

        tr = api.gate_memory.get_track_by_id(track_id)
        handoff = api.pending_lookahead_handoff
        is_handoff = bool(
            handoff is not None
            and api.canonical_track_id(handoff.get("track_id")) == api.canonical_track_id(track_id)
        )
        if tr is None or (not tr.committed and not is_handoff):
            api.horizon_rejected_track_ids.append(track_id)
            api.horizon_rejection_reason = "order_track_not_committed"
            api.horizon_track_decisions[track_id] = "excluded:order_track_not_committed"
            if is_handoff:
                api.promotion_blocked_reason = "handoff_track_missing_or_not_committed"
            # Sequence integrity matters more than horizon length. With a
            # predefined race order, do not skip an unavailable next gate
            # and accidentally plan to a later gate.
            break
        if (
            api.use_lookahead_gate_filter
            and not getattr(tr, "is_stable", False)
            and track_id not in api.race_accepted_track_ids
            and not is_handoff
        ):
            api.last_perception_accepted = False
            api.last_perception_rejection_reason = "track_not_stable"
            api.active_target_admission_status = "pending_stability"
            api.horizon_rejected_track_ids.append(track_id)
            api.horizon_rejection_reason = "track_not_stable"
            api.horizon_track_decisions[track_id] = "excluded:track_not_stable"
            print(
                f"[TARGET REJECT] reason=track_not_stable "
                f"track_id={tr.id} blocked={getattr(tr, 'promotion_blocked_reason', '')}"
            )
            api.promotion_blocked_reason = (
                getattr(tr, "promotion_blocked_reason", "")
                or "track_not_stable"
            )
            break
        candidate_center = (
            np.asarray(handoff["center"], dtype=float).reshape(3)
            if is_handoff
            else np.asarray(tr.center, dtype=float).reshape(3)
        )
        valid, reason = api.validate_planning_target(candidate_center)
        if not valid:
            api.last_perception_accepted = False
            api.last_perception_rejection_reason = reason
            api.horizon_rejected_track_ids.append(track_id)
            api.horizon_rejection_reason = reason
            api.horizon_track_decisions[track_id] = f"excluded:{reason}"
            if is_handoff:
                api.promotion_blocked_reason = f"planning_target_invalid:{reason}"
            print(f"[TARGET REJECT] reason={reason} track_id={tr.id} center={tr.center}")
            break
        if api.is_near_completed_gate(candidate_center):
            api.last_perception_accepted = False
            api.last_perception_rejection_reason = "already_completed_landmark"
            api.target_rejected_completed = True
            api.rejected_completed_this_lap = True
            api.horizon_rejected_track_ids.append(track_id)
            api.horizon_rejection_reason = "already_completed_landmark"
            api.horizon_track_decisions[track_id] = "excluded:already_completed_landmark"
            if is_handoff:
                api.promotion_blocked_reason = "already_completed_landmark"
            print(f"[TARGET REJECT] reason=already_completed_landmark track_id={tr.id} center={tr.center}")
            continue
        if track_id not in api.race_accepted_track_ids and not is_handoff:
            api.last_perception_accepted = False
            api.last_perception_rejection_reason = "track_not_admitted_to_race"
            api.active_target_admission_status = "rejected"
            api.horizon_rejected_track_ids.append(track_id)
            api.horizon_rejection_reason = "track_not_admitted_to_race"
            api.horizon_track_decisions[track_id] = "excluded:track_not_admitted_to_race"
            print(f"[TARGET REJECT] reason=track_not_admitted_to_race track_id={tr.id} center={tr.center}")
            continue
        valid, reason = api.validate_candidate_target(candidate_center, current_pos, track_id=tr.id)
        if not valid:
            api.last_perception_accepted = False
            api.last_perception_rejection_reason = reason
            api.horizon_rejected_track_ids.append(track_id)
            api.horizon_rejection_reason = reason
            api.horizon_track_decisions[track_id] = f"excluded:{reason}"
            if is_handoff:
                api.promotion_blocked_reason = f"candidate_target_invalid:{reason}"
            print(f"[TARGET REJECT] reason={reason} track_id={tr.id} center={tr.center}")
            continue
        target_tracks.append(tr)
        if is_handoff:
            target_center_overrides[track_id] = candidate_center.copy()
            api.promoted_lookahead_to_active = True
            api.promoted_track_id = track_id
            api.promoted_track_center = candidate_center.copy()
            api.promotion_blocked_reason = ""
            api.active_target_admission_status = "promoted_lookahead"
            print(
                "[LOOKAHEAD HANDOFF] promoted "
                f"track_id={track_id} center={candidate_center.tolist()} "
                f"previous_ids={api.previous_horizon_track_ids} "
                f"previous_types={api.previous_horizon_waypoint_types}"
            )
        api.horizon_track_decisions[track_id] = (
            "included:hard_current" if len(target_tracks) == 1 else "included:hard_stable"
        )
        selected_track_ids.add(track_id)
        api.valid_candidate_count += 1
        if len(target_tracks) == 1:
            api.selected_next_gate_track_id = tr.id
            api.selected_next_gate_stability_score = float(getattr(tr, "stability_score", np.nan))
            api.selected_target_source = "stable_track" if getattr(tr, "is_stable", False) else "race_admitted_track"
            print(
                f"TRACK {tr.id} selected as next target: "
                f"score={api.selected_next_gate_stability_score:.2f} "
                f"source={api.selected_target_source}"
            )
        if first_selected_order_idx is None:
            first_selected_order_idx = order_idx

    target_gates = [
        target_center_overrides.get(tr.id, tr.center).copy()
        for tr in target_tracks
    ]
    target_track_ids = [tr.id for tr in target_tracks]
    target_waypoint_types = [
        "hard_current" if i == 0 else "hard_stable"
        for i in range(len(target_gates))
    ]

    if len(target_gates) == 0:
        api.planning_horizon_waypoint_types = "start"
        api._planning_target_waypoint_types = []
        return np.array([current_pos], dtype=float), [], []

    target_gates, target_track_ids, target_waypoint_types = api._append_planning_lookahead_targets(
        current_pos=np.asarray(current_pos, dtype=float).reshape(3),
        target_gates=target_gates,
        target_track_ids=target_track_ids,
        target_waypoint_types=target_waypoint_types,
        max_gates_ahead=max_gates_ahead,
        allow_raw_candidates=True,
    )
    api.update_lookahead_pipeline_debug()
    api.horizon_selected_track_ids = list(target_track_ids)
    api.planning_horizon_waypoint_types = " ".join(["start"] + target_waypoint_types)
    api._planning_target_waypoint_types = list(target_waypoint_types)

    if len(target_tracks) == 0:
        if len(target_gates) == 0:
            api.planning_horizon_waypoint_types = "start"
            api._planning_target_waypoint_types = []
            return np.array([current_pos], dtype=float), [], []

    if first_selected_order_idx is not None:
        api.race_progression.cursor = first_selected_order_idx

    if len(target_gates) > 0:
        api.last_valid_target = target_gates[0].copy()
        api.active_target_source = "memory_track"
        first_track = api.gate_memory.get_track_by_id(target_track_ids[0]) if target_track_ids[0] >= 0 else None
        if first_track is None or target_track_ids[0] not in api.race_accepted_track_ids:
            api.selected_target_source = "planning_lookahead"
        else:
            api.selected_target_source = (
                "stable_track" if getattr(first_track, "is_stable", False) else "race_admitted_track"
            )
        api.active_target_track_id = target_track_ids[0]
        api.next_valid_target_found = True

    return np.vstack([current_pos] + target_gates), target_gates, target_track_ids


def build_waypoint_horizon_from_gt(api, current_pos, max_gates_ahead=3):
    if not api.use_perception:
        max_gates_ahead = 1

    remaining_gates = api.gt_gates[api.current_gate_idx:api.current_gate_idx + max_gates_ahead]
    if len(remaining_gates) == 0:
        api.planning_horizon_waypoint_types = "start"
        api._planning_target_waypoint_types = []
        return np.array([current_pos], dtype=float), [], []

    target_gates = [np.asarray(g, dtype=float) for g in remaining_gates]
    target_track_ids = [-1] * len(target_gates)  # no memory-track IDs in GT mode
    target_waypoint_types = ["hard_current"] + ["hard_stable"] * max(0, len(target_gates) - 1)
    api.planning_horizon_waypoint_types = " ".join(["start"] + target_waypoint_types)
    api._planning_target_waypoint_types = list(target_waypoint_types)
    return np.vstack([current_pos] + target_gates), target_gates, target_track_ids


def build_waypoint_horizon(api, current_pos, max_gates_ahead=3):
    """
    Unified horizon builder.
    Returns:
        waypoints: np.ndarray shape (N,3)
        target_gates: list[np.ndarray]
        target_track_ids: list[int]
    """
    if api.use_perception:
        return api.build_waypoint_horizon_from_memory(current_pos, max_gates_ahead=max_gates_ahead)
    return api.build_waypoint_horizon_from_gt(current_pos, max_gates_ahead=max_gates_ahead)
