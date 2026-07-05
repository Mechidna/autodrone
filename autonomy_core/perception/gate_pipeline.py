"""Low-risk perception pipeline debug helpers."""

import numpy as np


def _format_vec3(value):
    return "/".join(f"{x:.2f}" for x in np.asarray(value, dtype=float).reshape(3))


def build_detection_flow_debug_fields(entries):
    parts = []
    yolo_confidence = []
    quad_area_px2 = []
    old_area_confidence = []
    memory_confidence_used = []
    memory_admission_threshold = []
    memory_admission_passed = []
    pnp_camera_original = []
    pnp_camera_depth_corrected = []
    depth_correction_factor = []
    world_original = []
    world_depth_corrected = []

    for idx in sorted(entries):
        e = entries[idx]
        yolo_confidence.append(f"det{idx}:{e.get('yolo_confidence', np.nan):.3f}")
        quad_area_px2.append(f"det{idx}:{e.get('quad_area_px2', np.nan):.1f}")
        old_area_confidence.append(f"det{idx}:{e.get('old_area_confidence', np.nan):.3f}")
        memory_confidence_used.append(f"det{idx}:{e.get('memory_confidence', np.nan):.3f}")
        memory_admission_threshold.append(
            f"det{idx}:{e.get('memory_admission_threshold', np.nan):.3f}"
        )
        memory_admission_passed.append(
            f"det{idx}:{int(bool(e.get('memory_admission_passed', False)))}"
        )
        pnp_camera_original.append(
            f"det{idx}:{_format_vec3(e.get('pnp_camera_original', np.full(3, np.nan)))}"
        )
        pnp_camera_depth_corrected.append(
            f"det{idx}:{_format_vec3(e.get('pnp_camera_depth_corrected', np.full(3, np.nan)))}"
        )
        depth_correction_factor.append(
            f"det{idx}:{e.get('depth_correction_factor', 1.0):.6f}"
        )
        world_original.append(
            f"det{idx}:{_format_vec3(e.get('world_original', np.full(3, np.nan)))}"
        )
        world_depth_corrected.append(
            f"det{idx}:{_format_vec3(e.get('world_depth_corrected', np.full(3, np.nan)))}"
        )
        parts.append(
            f"det{idx}:yolo={e.get('yolo_confidence', np.nan):.3f},"
            f"area_px2={e.get('quad_area_px2', np.nan):.1f},"
            f"old_area_conf={e.get('old_area_confidence', np.nan):.3f},"
            f"mem_conf={e.get('memory_confidence', np.nan):.3f},"
            f"mem_pass={int(bool(e.get('memory_admission_passed', False)))},"
            f"pnp={int(bool(e.get('pnp')))},"
            f"cam={_format_vec3(e.get('cam', np.full(3, np.nan)))},"
            f"raw={_format_vec3(e.get('raw', np.full(3, np.nan)))},"
            f"corrected={_format_vec3(e.get('corrected', np.full(3, np.nan)))},"
            f"depth_diag_future={int(bool(e.get('diagnostic_far_depth_is_future', False)))},"
            f"depth_diag_class={e.get('diagnostic_far_depth_classification','')},"
            f"depth_factor={e.get('depth_correction_factor', 1.0):.6f},"
            f"track={e.get('track')},mem={int(bool(e.get('memory')))},"
            f"state={e.get('state','')},race={e.get('race_idx')},"
            f"role={e.get('role','')},reason={e.get('reason','')}"
        )

    return {
        "perception_detection_flow": ";".join(parts),
        "yolo_confidence": ";".join(yolo_confidence),
        "quad_area_px2": ";".join(quad_area_px2),
        "old_area_confidence": ";".join(old_area_confidence),
        "memory_confidence_used": ";".join(memory_confidence_used),
        "memory_admission_threshold": ";".join(memory_admission_threshold),
        "memory_admission_passed": ";".join(memory_admission_passed),
        "pnp_camera_original": ";".join(pnp_camera_original),
        "pnp_camera_depth_corrected": ";".join(pnp_camera_depth_corrected),
        "depth_correction_factor": ";".join(depth_correction_factor),
        "world_original": ";".join(world_original),
        "world_depth_corrected": ";".join(world_depth_corrected),
        "has_detection_flow_parts": bool(parts),
    }
