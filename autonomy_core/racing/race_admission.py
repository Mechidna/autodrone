"""Landmark merge and race-admission helpers used by AutonomyAPI wrappers."""

import numpy as np


def apply_landmark_merge_event(api, merge_event):
    api.duplicate_radius_used = api.gate_memory.duplicate_merge_radius
    api.pairwise_committed_track_distances = ";".join(
        f"{a}-{b}:{d:.2f}" for a, b, d in api.gate_memory.last_pairwise_distances
    )
    api.merge_candidate_pairs = ";".join(
        f"{a}-{b}:{d:.2f}" for a, b, d in api.gate_memory.last_merge_candidate_pairs
    )
    api.merge_blocked_reason = api.gate_memory.last_merge_blocked_reason
    if not merge_event or not merge_event.get("merged", False):
        api.merged_into_track_id = None
        api.duplicate_merge_reason = ""
        api.suspected_duplicate_track = False
        return

    source_id = int(merge_event["source_id"])
    target_id = int(merge_event["target_id"])
    api.track_id_aliases[source_id] = target_id
    api.merged_into_track_id = target_id
    api.duplicate_merge_reason = merge_event.get("reason", "")
    api.suspected_duplicate_track = True

    source_completed = source_id in api.completed_track_ids_this_cycle
    api.completed_track_ids_this_cycle.discard(source_id)
    if source_completed:
        api.completed_track_ids_this_cycle.add(target_id)
    api.race_accepted_track_ids = [
        api.canonical_track_id(tid) for tid in api.race_accepted_track_ids
    ]
    deduped = []
    for tid in api.race_accepted_track_ids:
        if tid is not None and tid not in deduped:
            deduped.append(tid)
    api.race_accepted_track_ids = deduped

    api.race_progression.inferred_order = [
        api.canonical_track_id(tid) for tid in api.race_progression.inferred_order
    ]
    order = []
    for tid in api.race_progression.inferred_order:
        if tid is not None and tid not in order:
            order.append(tid)
    api.race_progression.inferred_order = order
    if api.race_progression.cursor > len(order):
        api.race_progression.cursor = len(order)

    if source_id in api.active_target_track_ids:
        api.clear_active_perception_target(reason="active_target_merged_duplicate")


def refresh_landmark_merges(api):
    merge_event = api.gate_memory.merge_duplicate_committed_tracks()
    api.apply_landmark_merge_event(merge_event)
    return merge_event


def accept_track_into_race_order(api, tr):
    api.race_order_inserted = False
    api.race_order_rejected_reason = ""
    if tr is None or not tr.committed:
        api.race_order_rejected_reason = "track_not_committed"
        return False

    if api.use_lookahead_gate_filter and not getattr(tr, "is_stable", False):
        api.race_order_rejected_reason = getattr(
            tr,
            "promotion_blocked_reason",
            "track_not_stable",
        ) or "track_not_stable"
        api.rejected_track_temporary_vs_permanent = "temporary"
        api.active_target_admission_status = "pending_stability"
        print(
            f"TRACK {tr.id} blocked from promotion: "
            f"reason={api.race_order_rejected_reason}"
        )
        return False

    track_id = api.canonical_track_id(tr.id)
    api.track_id = track_id
    api.landmark_uncertainty = api.gate_memory.track_uncertainty(track_id)
    canonical_track = api.gate_memory.get_track_by_id(track_id)
    api.track_observations = 0 if canonical_track is None else canonical_track.hits
    api.rejected_track_temporary_vs_permanent = ""

    if track_id in api.race_accepted_track_ids:
        api.active_target_admission_status = "accepted"
        return True

    if api.track_observations < api.gate_memory.commit_hits:
        api.race_order_rejected_reason = "insufficient_observations"
        api.rejected_track_temporary_vs_permanent = "temporary"
        return False

    if (
        np.isfinite(api.landmark_uncertainty)
        and api.landmark_uncertainty > api.gate_memory.duplicate_merge_radius
    ):
        api.race_order_rejected_reason = f"landmark_uncertainty_too_high:{api.landmark_uncertainty:.2f}"
        api.rejected_track_temporary_vs_permanent = "temporary"
        return False

    if api.race_gate_count is not None and len(api.race_accepted_track_ids) >= api.race_gate_count:
        api.race_order_rejected_reason = "race_gate_count_reached"
        api.rejected_track_temporary_vs_permanent = "temporary"
        return False

    if api.is_near_completed_gate(tr.center, radius=api.gate_memory.duplicate_merge_radius):
        api.race_order_rejected_reason = "near_completed_unique_gate"
        api.rejected_track_temporary_vs_permanent = "temporary"
        api.suspected_duplicate_track = True
        return False

    duplicate = None
    for accepted_id in api.race_accepted_track_ids:
        accepted = api.gate_memory.get_track_by_id(accepted_id)
        if accepted is None:
            continue
        dist = float(np.linalg.norm(tr.center - accepted.center))
        if dist < api.gate_memory.duplicate_merge_radius:
            duplicate = accepted
            break
    if duplicate is not None:
        api.gate_memory.merge_track_into(track_id, duplicate.id, reason="race_order_duplicate")
        api.apply_landmark_merge_event(api.gate_memory.last_merge_event)
        api.race_order_rejected_reason = "duplicate_of_accepted_race_gate"
        api.rejected_track_temporary_vs_permanent = "merged"
        api.suspected_duplicate_track = True
        return False

    api.race_accepted_track_ids.append(track_id)
    api.race_order_inserted = True
    api.active_target_admission_status = "accepted"
    api.race_admitted_track_ids = list(api.race_accepted_track_ids)
    print(
        f"TRACK {track_id} admitted to race candidate pool; "
        "sequence index assigned by geometric progress"
    )
    return True


def assign_race_order_from_progress(api, committed_tracks):
    """Assign uncompleted tracks without letting a farther gate take the current slot."""
    current_pos = np.array([
        api.telemetry.pos["x"],
        api.telemetry.pos["y"],
        api.telemetry.pos["z"],
    ], dtype=float)
    previous_idx = int(api.current_gate_idx)
    candidates = []
    rejected = []

    for tr in committed_tracks:
        track_id = api.canonical_track_id(tr.id)
        center_source = getattr(tr, "filtered_center_world", None)
        if center_source is None:
            center_source = getattr(tr, "center", None)
        if track_id is None or center_source is None:
            continue
        center = np.asarray(center_source, dtype=float).reshape(3)
        valid, reason = api.validate_planning_target(center)
        if not valid:
            rejected.append((track_id, reason))
            continue
        if track_id in api.completed_track_ids_this_cycle or api.is_near_completed_gate(center):
            rejected.append((track_id, "completed_gate"))
            continue
        candidates.append({
            "track": tr,
            "track_id": track_id,
            "center": center,
            "distance": float(np.linalg.norm(center - current_pos)),
            "progress": float("nan"),
        })

    candidates.sort(key=lambda item: (item["distance"], item["track_id"]))
    api.current_gate_candidate_track_ids = [item["track_id"] for item in candidates]
    api.selected_current_track_id = candidates[0]["track_id"] if candidates else None
    api.rejected_current_track_ids = [
        item["track_id"] for item in candidates[1:]
    ] + [track_id for track_id, _ in rejected]

    if len(candidates) >= 2:
        nearest = candidates[0]["center"]
        farthest = max(candidates[1:], key=lambda item: item["distance"])["center"]
        course_direction = farthest - nearest
        norm = float(np.linalg.norm(course_direction))
        if norm > 1e-6:
            course_direction /= norm
            if float(np.dot(nearest - current_pos, course_direction)) < 0.0:
                course_direction *= -1.0
        else:
            course_direction = (nearest - current_pos) / max(
                float(np.linalg.norm(nearest - current_pos)), 1e-6
            )
    elif len(candidates) == 1:
        course_direction = candidates[0]["center"] - current_pos
        course_direction /= max(float(np.linalg.norm(course_direction)), 1e-6)
    else:
        course_direction = np.array([0.0, 1.0, 0.0], dtype=float)

    for item in candidates:
        item["progress"] = float(np.dot(item["center"] - current_pos, course_direction))

    if candidates:
        current = candidates[0]
        future = sorted(
            candidates[1:],
            key=lambda item: (item["progress"], item["distance"], item["track_id"]),
        )
        assigned = [current] + future
    else:
        assigned = []

    assigned_index = {
        item["track_id"]: previous_idx + rank
        for rank, item in enumerate(assigned)
    }
    api.future_lookahead_track_ids = [
        item["track_id"] for item in assigned[1:]
    ]

    existing_order = [
        api.canonical_track_id(track_id)
        for track_id in api.race_progression.inferred_order
    ]
    completed_prefix = []
    for track_id in existing_order[:previous_idx]:
        if track_id is not None and track_id not in completed_prefix:
            completed_prefix.append(track_id)

    contiguous_order = list(completed_prefix)
    missing_preceding = False
    for item in assigned:
        track_id = item["track_id"]
        if missing_preceding or track_id not in api.race_accepted_track_ids:
            missing_preceding = True
            continue
        contiguous_order.append(track_id)

    api.race_progression.inferred_order = contiguous_order
    api.race_progression.cursor = min(
        max(int(api.race_progression.cursor), previous_idx),
        len(contiguous_order),
    )

    rejection_parts = []
    for item in assigned[1:]:
        rejection_parts.append(
            f"track{item['track_id']}:farther_than_current_track{api.selected_current_track_id}"
        )
    rejection_parts.extend(f"track{track_id}:{reason}" for track_id, reason in rejected)
    api.current_selection_rejection_reason = ";".join(rejection_parts)

    debug_parts = []
    for rank, item in enumerate(assigned):
        track_id = item["track_id"]
        race_idx = assigned_index[track_id]
        if rank == 0:
            reason = "selected_nearest_valid_current_candidate"
        elif track_id in api.race_accepted_track_ids:
            reason = "future_gate_progress_order"
        else:
            reason = "future_gate_pending_race_admission"
        if track_id in api.race_accepted_track_ids and track_id not in contiguous_order:
            reason += "|withheld_until_preceding_gate_admitted"
        center = item["center"]
        debug_parts.append(
            f"track{track_id}:center={center[0]:.2f}/{center[1]:.2f}/{center[2]:.2f},"
            f"dist={item['distance']:.2f},progress={item['progress']:.2f},"
            f"score={-item['progress']:.2f},prev_active={previous_idx},"
            f"assigned={race_idx},reason={reason}"
        )
    for track_id, reason in rejected:
        debug_parts.append(
            f"track{track_id}:prev_active={previous_idx},assigned=None,reason=rejected:{reason}"
        )
    api.race_order_assignment_debug = ";".join(debug_parts)
    if debug_parts:
        print("[RACE ORDER ASSIGNMENT] " + api.race_order_assignment_debug)


def refresh_race_order_from_memory(api):
    api.refresh_landmark_merges()
    committed_tracks = api.gate_memory.get_committed_tracks()
    stable_tracks = api.gate_memory.get_stable_tracks()
    api.update_gate_filter_summary_logs()
    api.committed_track_centers_log = ";".join(
        f"{tr.id}:{tr.center[0]:.2f},{tr.center[1]:.2f},{tr.center[2]:.2f}:h{tr.hits}"
        for tr in committed_tracks
    )
    tracks_for_admission = stable_tracks if api.use_lookahead_gate_filter else committed_tracks
    for tr in tracks_for_admission:
        api.accept_track_into_race_order(tr)

    api.assign_race_order_from_progress(committed_tracks)
    valid_ids = {tr.id for tr in committed_tracks}
    order = []
    for tid in api.race_progression.inferred_order:
        tid = api.canonical_track_id(tid)
        if tid is None or tid not in valid_ids:
            continue
        if tid not in api.race_accepted_track_ids:
            continue
        if tid not in order:
            order.append(tid)
    api.race_progression.inferred_order = order
    if api.race_progression.cursor > len(order):
        api.race_progression.cursor = len(order)
    api.race_order_track_ids = api.race_progression.order()
    api.race_order_after_merge = list(api.race_order_track_ids)
    return committed_tracks
