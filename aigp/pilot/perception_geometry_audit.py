from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np

from autonomy_core.core.competition_config import VADR_TS_002
from autonomy_core.core.frame_conventions import (
    body_frd_to_local_ned_rotmat,
    local_ned_to_neu,
    local_neu_to_ned,
    opencv_camera_to_mavlink_body_frd_rotmat,
)


PX4_X500_MONO_CAM_BODY_FRD_TRANSLATION_M = np.array(
    [0.12, -0.03, -0.242],
    dtype=float,
)


class PerceptionGeometryAudit:
    """Debug-only geometry residuals for PX4/sim validation.

    This class never feeds perception, planning, estimation, or control. It only
    compares live detections against configured known gates and prints compact
    residuals.
    """

    def __init__(self, config_section: Any):
        self.enabled = bool(getattr(config_section, "enabled", False))
        self.print_period_s = max(0.0, float(getattr(config_section, "print_period_s", 0.5)))
        self.max_prints = max(0, int(getattr(config_section, "max_prints", 80)))
        self.max_match_distance_m = max(
            0.0,
            float(getattr(config_section, "max_match_distance_m", 5.0)),
        )
        self.known_gate_positions_neu = self._vec3_array(
            getattr(config_section, "known_gate_positions_neu", ())
        )
        self.gate_right_axis_neu = self._unit_vec3(
            getattr(config_section, "gate_right_axis_neu", (1.0, 0.0, 0.0)),
            default=np.array([1.0, 0.0, 0.0], dtype=float),
        )
        self.gate_up_axis_neu = self._unit_vec3(
            getattr(config_section, "gate_up_axis_neu", (0.0, 0.0, 1.0)),
            default=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        self._last_print_s = -math.inf
        self._printed = 0

        if abs(float(np.dot(self.gate_right_axis_neu, self.gate_up_axis_neu))) > 0.95:
            self.enabled = False
            print(
                "[GEOM_AUDIT_CONFIG] disabled reason=gate_axes_nearly_parallel",
                flush=True,
            )
            return

        known_count = int(len(self.known_gate_positions_neu))
        print(
            "[GEOM_AUDIT_CONFIG] "
            f"enabled={int(self.enabled)} "
            f"known_gates={known_count} "
            f"print_period_s={self.print_period_s:.2f} "
            f"max_prints={self.max_prints} "
            f"max_match_distance_m={self.max_match_distance_m:.2f} "
            f"gate_right_neu={self._fmt_vec(self.gate_right_axis_neu, 2)} "
            f"gate_up_neu={self._fmt_vec(self.gate_up_axis_neu, 2)}",
            flush=True,
        )

    @property
    def active(self) -> bool:
        return bool(
            self.enabled
            and self.max_prints > 0
            and self._printed < self.max_prints
            and len(self.known_gate_positions_neu) > 0
        )

    def maybe_print(
        self,
        detections: list[dict[str, Any]],
        *,
        drone_pos_ned: np.ndarray,
        drone_rpy_rad: np.ndarray,
        camera_matrix: np.ndarray,
        camera_to_body: np.ndarray,
        camera_translation_body: np.ndarray,
        object_points_m: np.ndarray,
        frame_id: Optional[int] = None,
    ) -> None:
        if not self.active:
            return
        now_s = time.monotonic()
        if now_s - self._last_print_s < self.print_period_s:
            return

        printed_this_call = 0
        for index, detection in enumerate(detections):
            if self._printed >= self.max_prints:
                break
            result = self.evaluate_detection(
                detection,
                detection_index=index,
                drone_pos_ned=drone_pos_ned,
                drone_rpy_rad=drone_rpy_rad,
                camera_matrix=camera_matrix,
                camera_to_body=camera_to_body,
                camera_translation_body=camera_translation_body,
                object_points_m=object_points_m,
                frame_id=frame_id,
            )
            if result is None:
                continue
            print(self.format_result(result), flush=True)
            self._printed += 1
            printed_this_call += 1

        if printed_this_call:
            self._last_print_s = now_s

    def evaluate_detection(
        self,
        detection: dict[str, Any],
        *,
        detection_index: int,
        drone_pos_ned: np.ndarray,
        drone_rpy_rad: np.ndarray,
        camera_matrix: np.ndarray,
        camera_to_body: np.ndarray,
        camera_translation_body: np.ndarray,
        object_points_m: np.ndarray,
        frame_id: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        if not self.active:
            return None

        gate_world_neu = self._vec3(detection.get("gate_center_world"))
        gate_camera = self._vec3(detection.get("gate_center_camera"))
        if gate_world_neu is None or gate_camera is None:
            return None

        match_index, match_dist = self._nearest_known_gate(gate_world_neu)
        if match_index is None or match_dist > self.max_match_distance_m:
            return None

        known_gate_neu = self.known_gate_positions_neu[match_index]
        drone_pos_ned = np.asarray(drone_pos_ned, dtype=float).reshape(3)
        drone_rpy_rad = np.asarray(drone_rpy_rad, dtype=float).reshape(3)
        camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        camera_to_body = np.asarray(camera_to_body, dtype=float).reshape(3, 3)
        camera_translation_body = np.asarray(camera_translation_body, dtype=float).reshape(3)
        object_points_m = np.asarray(object_points_m, dtype=float).reshape(4, 3)

        rot_ned_body = body_frd_to_local_ned_rotmat(*drone_rpy_rad)
        expected_body = rot_ned_body.T @ (
            local_neu_to_ned(known_gate_neu) - drone_pos_ned
        )
        body_to_camera = camera_to_body.T
        expected_camera = body_to_camera @ (expected_body - camera_translation_body)

        gate_body = self._vec3(detection.get("gate_center_body_frd"))
        if gate_body is None:
            gate_body = camera_to_body @ gate_camera
        pnp_body_from_origin = camera_translation_body + gate_body
        pnp_delta_ned = rot_ned_body @ pnp_body_from_origin
        expected_delta_ned = local_neu_to_ned(known_gate_neu) - drone_pos_ned

        observed_keypoints = self._keypoints_px(detection)
        expected_keypoints = self.project_known_gate_keypoints(
            known_gate_neu=known_gate_neu,
            drone_pos_ned=drone_pos_ned,
            drone_rpy_rad=drone_rpy_rad,
            camera_matrix=camera_matrix,
            camera_to_body=camera_to_body,
            camera_translation_body=camera_translation_body,
            object_points_m=object_points_m,
        )

        observed_center = self._finite_mean_2d(observed_keypoints)
        expected_center = self._finite_mean_2d(expected_keypoints)
        center_error_px = (
            observed_center - expected_center
            if observed_center is not None and expected_center is not None
            else np.full(2, np.nan, dtype=float)
        )
        keypoint_rmse_px = self._keypoint_rmse(observed_keypoints, expected_keypoints)
        keypoint_corner_errors_px = self._keypoint_errors(observed_keypoints, expected_keypoints)
        observed_side_lengths_px = self._side_lengths(observed_keypoints)
        expected_side_lengths_px = self._side_lengths(expected_keypoints)
        keypoint_area_ratio = self._area_ratio(observed_keypoints, expected_keypoints)
        size_depth_m = self._size_depth(observed_keypoints, camera_matrix, object_points_m)
        reprojected_keypoints = self._keypoints_from_value(detection.get("reprojected_corners"))
        reprojected_center = self._finite_mean_2d(reprojected_keypoints)
        reprojected_center_error_px = (
            reprojected_center - observed_center
            if reprojected_center is not None and observed_center is not None
            else np.full(2, np.nan, dtype=float)
        )
        keypoint_conf = self._keypoint_confidence(detection)
        camera_bearing_error_deg = self._camera_bearing_error_deg(
            gate_camera,
            expected_camera,
        )
        yaw_correction_deg = self._horizontal_yaw_correction_deg(
            pnp_delta_ned,
            expected_delta_ned,
        )
        horizontal_range_ratio = self._horizontal_range_ratio(
            pnp_delta_ned,
            expected_delta_ned,
        )
        transform_sweep = self._transform_sweep(
            gate_camera=gate_camera,
            drone_pos_ned=drone_pos_ned,
            drone_rpy_rad=drone_rpy_rad,
            camera_translation_body=camera_translation_body,
            camera_to_body=camera_to_body,
            known_gate_neu=known_gate_neu,
        )
        pnp_candidate_sweep = self._pnp_candidate_sweep(
            detection,
            drone_pos_ned=drone_pos_ned,
            drone_rpy_rad=drone_rpy_rad,
            camera_translation_body=camera_translation_body,
            camera_to_body=camera_to_body,
            known_gate_neu=known_gate_neu,
        )

        return {
            "frame_id": frame_id,
            "detection_index": int(detection_index),
            "gate_index": int(match_index),
            "match_distance_m": float(match_dist),
            "world_error_neu_m": gate_world_neu - known_gate_neu,
            "world_error_norm_m": float(np.linalg.norm(gate_world_neu - known_gate_neu)),
            "body_error_m": pnp_body_from_origin - expected_body,
            "camera_error_m": gate_camera - expected_camera,
            "observed_keypoint_center_px": observed_center,
            "expected_keypoint_center_px": expected_center,
            "keypoint_center_error_px": center_error_px,
            "keypoint_rmse_px": keypoint_rmse_px,
            "keypoint_corner_errors_px": keypoint_corner_errors_px,
            "observed_side_lengths_px": observed_side_lengths_px,
            "expected_side_lengths_px": expected_side_lengths_px,
            "keypoint_area_ratio": keypoint_area_ratio,
            "keypoint_confidence": keypoint_conf,
            "reprojected_center_error_px": reprojected_center_error_px,
            "pnp_camera_m": gate_camera,
            "expected_camera_m": expected_camera,
            "pnp_depth_m": float(gate_camera[2]),
            "size_depth_m": size_depth_m,
            "camera_bearing_error_deg": camera_bearing_error_deg,
            "yaw_correction_deg": yaw_correction_deg,
            "horizontal_range_ratio": horizontal_range_ratio,
            "transform_sweep": transform_sweep,
            "pnp_candidate_sweep": pnp_candidate_sweep,
            "pnp_selected_order": str(detection.get("pnp_selected_order", "")),
            "pnp_selected_solver": str(detection.get("pnp_selected_solver", "")),
            "pnp_debug_best_order": str(detection.get("pnp_debug_best_order", "")),
            "allow_pnp_corner_reordering": bool(
                detection.get("allow_pnp_corner_reordering", False)
            ),
            "pnp_live_vs_debug_best_order_mismatch": bool(
                detection.get("pnp_live_vs_debug_best_order_mismatch", False)
            ),
            "keypoint_polygon_winding": str(detection.get("keypoint_polygon_winding", "")),
            "reprojection_error_px": self._finite_float(
                detection.get("reprojection_error"),
                math.nan,
            ),
            "rpy_deg": np.degrees(drone_rpy_rad),
        }

    def project_known_gate_keypoints(
        self,
        *,
        known_gate_neu: np.ndarray,
        drone_pos_ned: np.ndarray,
        drone_rpy_rad: np.ndarray,
        camera_matrix: np.ndarray,
        camera_to_body: np.ndarray,
        camera_translation_body: np.ndarray,
        object_points_m: np.ndarray,
    ) -> np.ndarray:
        known_gate_neu = np.asarray(known_gate_neu, dtype=float).reshape(3)
        drone_pos_ned = np.asarray(drone_pos_ned, dtype=float).reshape(3)
        drone_rpy_rad = np.asarray(drone_rpy_rad, dtype=float).reshape(3)
        camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        camera_to_body = np.asarray(camera_to_body, dtype=float).reshape(3, 3)
        camera_translation_body = np.asarray(camera_translation_body, dtype=float).reshape(3)
        object_points_m = np.asarray(object_points_m, dtype=float).reshape(4, 3)

        rot_ned_body = body_frd_to_local_ned_rotmat(*drone_rpy_rad)
        body_to_camera = camera_to_body.T
        points_px = []
        for obj in object_points_m:
            corner_neu = (
                known_gate_neu
                + float(obj[0]) * self.gate_right_axis_neu
                + float(obj[1]) * self.gate_up_axis_neu
            )
            corner_body = rot_ned_body.T @ (local_neu_to_ned(corner_neu) - drone_pos_ned)
            corner_camera = body_to_camera @ (corner_body - camera_translation_body)
            points_px.append(self._project_camera_point(corner_camera, camera_matrix))
        return np.asarray(points_px, dtype=float).reshape(4, 2)

    def format_result(self, result: dict[str, Any]) -> str:
        frame = result.get("frame_id")
        frame_text = "" if frame is None else f" frame={int(frame)}"
        return (
            "[GEOM_AUDIT] "
            f"gate={int(result['gate_index']) + 1} "
            f"det={int(result['detection_index'])}"
            f"{frame_text} "
            f"match={float(result['match_distance_m']):.2f} "
            f"world_err={self._fmt_vec(result['world_error_neu_m'], 2)} "
            f"world_norm={float(result['world_error_norm_m']):.2f} "
            f"body_err={self._fmt_vec(result['body_error_m'], 2)} "
            f"cam_err={self._fmt_vec(result['camera_error_m'], 2)} "
            f"kp_obs={self._fmt_vec(result['observed_keypoint_center_px'], 1)} "
            f"kp_exp={self._fmt_vec(result['expected_keypoint_center_px'], 1)} "
            f"kp_err={self._fmt_vec(result['keypoint_center_error_px'], 1)} "
            f"kp_rmse={float(result['keypoint_rmse_px']):.1f} "
            f"pnp_cam={self._fmt_vec(result['pnp_camera_m'], 2)} "
            f"exp_cam={self._fmt_vec(result['expected_camera_m'], 2)} "
            f"depth={float(result['pnp_depth_m']):.2f} "
            f"size_depth={float(result['size_depth_m']):.2f} "
            f"reproj={float(result['reprojection_error_px']):.2f} "
            f"rpy_deg={self._fmt_vec(result['rpy_deg'], 1)}"
            "\n"
            "[GEOM_DIAG] "
            f"gate={int(result['gate_index']) + 1} "
            f"det={int(result['detection_index'])}"
            f"{frame_text} "
            f"cam_bearing_err_deg={self._fmt_vec(result['camera_bearing_error_deg'], 1)} "
            f"yaw_corr_deg={float(result['yaw_correction_deg']):.2f} "
            f"range_ratio={float(result['horizontal_range_ratio']):.3f} "
            f"reproj_center_err_px={self._fmt_vec(result['reprojected_center_error_px'], 1)} "
            f"kconf={self._fmt_vec(result['keypoint_confidence'], 2)} "
            f"pnp={result['pnp_selected_solver']}/{result['pnp_selected_order']} "
            f"debug_best={result['pnp_debug_best_order']} "
            f"reorder={int(result['allow_pnp_corner_reordering'])} "
            f"mismatch={int(result['pnp_live_vs_debug_best_order_mismatch'])} "
            f"wind={result['keypoint_polygon_winding']} "
            f"tf={self._fmt_sweep(result['transform_sweep'])} "
            f"pnp_known={self._fmt_pnp_sweep(result['pnp_candidate_sweep'])}"
            "\n"
            "[KEYPOINT_AUDIT] "
            f"gate={int(result['gate_index']) + 1} "
            f"det={int(result['detection_index'])}"
            f"{frame_text} "
            f"corner_err_px={self._fmt_corner_errors(result['keypoint_corner_errors_px'])} "
            f"side_obs_px={self._fmt_vec(result['observed_side_lengths_px'], 1)} "
            f"side_exp_px={self._fmt_vec(result['expected_side_lengths_px'], 1)} "
            f"area_ratio={float(result['keypoint_area_ratio']):.2f}"
        )

    def _nearest_known_gate(self, gate_world_neu: np.ndarray) -> tuple[Optional[int], float]:
        if len(self.known_gate_positions_neu) == 0:
            return None, math.inf
        delta = self.known_gate_positions_neu - np.asarray(gate_world_neu, dtype=float).reshape(3)
        distances = np.linalg.norm(delta, axis=1)
        index = int(np.argmin(distances))
        return index, float(distances[index])

    @staticmethod
    def _project_camera_point(point_camera: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        point = np.asarray(point_camera, dtype=float).reshape(3)
        z = float(point[2])
        if not math.isfinite(z) or z <= 1e-9:
            return np.full(2, np.nan, dtype=float)
        k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        return np.array(
            [
                k[0, 0] * (point[0] / z) + k[0, 2],
                k[1, 1] * (point[1] / z) + k[1, 2],
            ],
            dtype=float,
        )

    @staticmethod
    def _keypoints_px(detection: dict[str, Any]) -> Optional[np.ndarray]:
        for key in ("yolo_keypoints", "ordered_corners", "raw_corners", "keypoints_px"):
            value = detection.get(key)
            out = PerceptionGeometryAudit._keypoints_from_value(value)
            if out is not None:
                return out
        return None

    @staticmethod
    def _keypoints_from_value(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim >= 2 and arr.shape[0] >= 4 and arr.shape[1] >= 2:
            out = arr[:4, :2].reshape(4, 2).copy()
            if np.all(np.isfinite(out)):
                return out
        return None

    @staticmethod
    def _keypoint_confidence(detection: dict[str, Any]) -> np.ndarray:
        for key in ("yolo_keypoints", "keypoints_px"):
            value = detection.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float)
            if arr.ndim >= 2 and arr.shape[0] >= 4 and arr.shape[1] >= 3:
                out = arr[:4, 2].reshape(4).astype(float)
                if np.all(np.isfinite(out)):
                    return out
        return np.full(4, np.nan, dtype=float)

    @staticmethod
    def _camera_bearing_error_deg(observed_camera: np.ndarray, expected_camera: np.ndarray) -> np.ndarray:
        observed = np.asarray(observed_camera, dtype=float).reshape(3)
        expected = np.asarray(expected_camera, dtype=float).reshape(3)
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(expected)):
            return np.full(2, np.nan, dtype=float)
        if abs(float(observed[2])) < 1e-9 or abs(float(expected[2])) < 1e-9:
            return np.full(2, np.nan, dtype=float)
        observed_x = math.atan2(float(observed[0]), float(observed[2]))
        expected_x = math.atan2(float(expected[0]), float(expected[2]))
        observed_y = math.atan2(float(observed[1]), float(observed[2]))
        expected_y = math.atan2(float(expected[1]), float(expected[2]))
        return np.degrees(
            np.array(
                [
                    PerceptionGeometryAudit._wrap_pi(observed_x - expected_x),
                    PerceptionGeometryAudit._wrap_pi(observed_y - expected_y),
                ],
                dtype=float,
            )
        )

    @staticmethod
    def _horizontal_yaw_correction_deg(predicted_delta_ned: np.ndarray, expected_delta_ned: np.ndarray) -> float:
        predicted = np.asarray(predicted_delta_ned, dtype=float).reshape(3)
        expected = np.asarray(expected_delta_ned, dtype=float).reshape(3)
        if not np.all(np.isfinite(predicted[:2])) or not np.all(np.isfinite(expected[:2])):
            return math.nan
        if np.linalg.norm(predicted[:2]) < 1e-9 or np.linalg.norm(expected[:2]) < 1e-9:
            return math.nan
        pred_angle = math.atan2(float(predicted[1]), float(predicted[0]))
        exp_angle = math.atan2(float(expected[1]), float(expected[0]))
        return math.degrees(PerceptionGeometryAudit._wrap_pi(exp_angle - pred_angle))

    @staticmethod
    def _horizontal_range_ratio(predicted_delta_ned: np.ndarray, expected_delta_ned: np.ndarray) -> float:
        predicted = np.asarray(predicted_delta_ned, dtype=float).reshape(3)
        expected = np.asarray(expected_delta_ned, dtype=float).reshape(3)
        predicted_norm = float(np.linalg.norm(predicted[:2]))
        expected_norm = float(np.linalg.norm(expected[:2]))
        if not math.isfinite(predicted_norm) or not math.isfinite(expected_norm) or expected_norm < 1e-9:
            return math.nan
        return predicted_norm / expected_norm

    def _transform_sweep(
        self,
        *,
        gate_camera: np.ndarray,
        drone_pos_ned: np.ndarray,
        drone_rpy_rad: np.ndarray,
        camera_translation_body: np.ndarray,
        camera_to_body: np.ndarray,
        known_gate_neu: np.ndarray,
    ) -> list[dict[str, Any]]:
        tilt = float(VADR_TS_002.camera_tilt_up_rad)
        camera_translation_body = np.asarray(camera_translation_body, dtype=float).reshape(3)
        variants = [
            ("current", camera_to_body, 0.0, camera_translation_body),
            ("spec_origin", camera_to_body, 0.0, np.zeros(3, dtype=float)),
            (
                "px4_x500_mount",
                camera_to_body,
                0.0,
                PX4_X500_MONO_CAM_BODY_FRD_TRANSLATION_M,
            ),
            ("yaw-5.7", camera_to_body, math.radians(-5.7), camera_translation_body),
            ("yaw+5.7", camera_to_body, math.radians(5.7), camera_translation_body),
            (
                "cam_x_flip",
                camera_to_body @ np.diag([-1.0, 1.0, 1.0]),
                0.0,
                camera_translation_body,
            ),
            (
                "cam_y_flip",
                camera_to_body @ np.diag([1.0, -1.0, 1.0]),
                0.0,
                camera_translation_body,
            ),
            (
                "cam_xy_flip",
                camera_to_body @ np.diag([-1.0, -1.0, 1.0]),
                0.0,
                camera_translation_body,
            ),
            (
                "tilt_down20",
                opencv_camera_to_mavlink_body_frd_rotmat(-tilt),
                0.0,
                camera_translation_body,
            ),
            (
                "tilt_0",
                opencv_camera_to_mavlink_body_frd_rotmat(0.0),
                0.0,
                camera_translation_body,
            ),
        ]
        out = []
        for name, c2b, yaw_delta, translation in variants:
            world_neu = self._world_from_camera(
                gate_camera=gate_camera,
                drone_pos_ned=drone_pos_ned,
                drone_rpy_rad=drone_rpy_rad + np.array([0.0, 0.0, yaw_delta], dtype=float),
                camera_translation_body=translation,
                camera_to_body=c2b,
            )
            err = float(np.linalg.norm(world_neu - known_gate_neu))
            out.append({
                "name": name,
                "err": err,
                "world": world_neu,
            })
        out.sort(key=lambda item: item["err"])
        mandatory_names = ("current", "spec_origin", "px4_x500_mount")
        selected = []
        for name in mandatory_names:
            match = next((item for item in out if item["name"] == name), None)
            if match is not None:
                selected.append(match)
        for item in out:
            if all(existing["name"] != item["name"] for existing in selected):
                selected.append(item)
            if len(selected) >= 5:
                break
        return selected

    def _pnp_candidate_sweep(
        self,
        detection: dict[str, Any],
        *,
        drone_pos_ned: np.ndarray,
        drone_rpy_rad: np.ndarray,
        camera_translation_body: np.ndarray,
        camera_to_body: np.ndarray,
        known_gate_neu: np.ndarray,
    ) -> Optional[dict[str, Any]]:
        candidates = detection.get("pnp_candidates")
        if not isinstance(candidates, (list, tuple)) or not candidates:
            return None
        selected_camera = self._vec3(detection.get("gate_center_camera"))
        selected_err = math.nan
        if selected_camera is not None:
            selected_world = self._world_from_camera(
                gate_camera=selected_camera,
                drone_pos_ned=drone_pos_ned,
                drone_rpy_rad=drone_rpy_rad,
                camera_translation_body=camera_translation_body,
                camera_to_body=camera_to_body,
            )
            selected_err = float(np.linalg.norm(selected_world - known_gate_neu))

        best = None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            tvec = self._vec3(candidate.get("tvec"))
            if tvec is None:
                continue
            world_neu = self._world_from_camera(
                gate_camera=tvec,
                drone_pos_ned=drone_pos_ned,
                drone_rpy_rad=drone_rpy_rad,
                camera_translation_body=camera_translation_body,
                camera_to_body=camera_to_body,
            )
            err = float(np.linalg.norm(world_neu - known_gate_neu))
            item = {
                "err": err,
                "selected_err": selected_err,
                "solver": str(candidate.get("solver", "")),
                "order": str(candidate.get("order", "")),
                "solver_candidate_index": int(candidate.get("solver_candidate_index", -1)),
                "reprojection_error": self._finite_float(candidate.get("error"), math.nan),
                "depth": self._finite_float(candidate.get("depth"), math.nan),
            }
            if best is None or err < float(best["err"]):
                best = item
        return best

    @staticmethod
    def _world_from_camera(
        *,
        gate_camera: np.ndarray,
        drone_pos_ned: np.ndarray,
        drone_rpy_rad: np.ndarray,
        camera_translation_body: np.ndarray,
        camera_to_body: np.ndarray,
    ) -> np.ndarray:
        rot_ned_body = body_frd_to_local_ned_rotmat(*np.asarray(drone_rpy_rad, dtype=float).reshape(3))
        body = np.asarray(camera_translation_body, dtype=float).reshape(3) + (
            np.asarray(camera_to_body, dtype=float).reshape(3, 3)
            @ np.asarray(gate_camera, dtype=float).reshape(3)
        )
        world_ned = np.asarray(drone_pos_ned, dtype=float).reshape(3) + rot_ned_body @ body
        return local_ned_to_neu(world_ned)

    @staticmethod
    def _wrap_pi(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _fmt_sweep(sweep: list[dict[str, Any]]) -> str:
        if not sweep:
            return "none"
        return ",".join(
            f"{item['name']}:{float(item['err']):.2f}" for item in sweep[:5]
        )

    @staticmethod
    def _fmt_corner_errors(errors: Any) -> str:
        arr = np.asarray(errors, dtype=float).reshape(4, 2)
        labels = ("tl", "tr", "br", "bl")
        return ",".join(
            f"{label}({float(err[0]):.1f},{float(err[1]):.1f})"
            for label, err in zip(labels, arr)
        )

    @staticmethod
    def _fmt_pnp_sweep(sweep: Optional[dict[str, Any]]) -> str:
        if not sweep:
            return "none"
        return (
            f"sel:{float(sweep['selected_err']):.2f} "
            f"best:{sweep['solver']}/{sweep['order']}/"
            f"{int(sweep['solver_candidate_index'])}:"
            f"{float(sweep['err']):.2f} "
            f"reproj:{float(sweep['reprojection_error']):.2f} "
            f"z:{float(sweep['depth']):.2f}"
        )

    @staticmethod
    def _finite_mean_2d(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if points is None:
            return None
        arr = np.asarray(points, dtype=float).reshape(-1, 2)
        mask = np.all(np.isfinite(arr), axis=1)
        if not np.any(mask):
            return None
        return np.mean(arr[mask], axis=0)

    @staticmethod
    def _keypoint_rmse(observed: Optional[np.ndarray], expected: np.ndarray) -> float:
        if observed is None:
            return math.nan
        obs = np.asarray(observed, dtype=float).reshape(4, 2)
        exp = np.asarray(expected, dtype=float).reshape(4, 2)
        mask = np.all(np.isfinite(obs), axis=1) & np.all(np.isfinite(exp), axis=1)
        if not np.any(mask):
            return math.nan
        return float(np.sqrt(np.mean(np.sum((obs[mask] - exp[mask]) ** 2, axis=1))))

    @staticmethod
    def _keypoint_errors(observed: Optional[np.ndarray], expected: np.ndarray) -> np.ndarray:
        if observed is None:
            return np.full((4, 2), np.nan, dtype=float)
        obs = np.asarray(observed, dtype=float).reshape(4, 2)
        exp = np.asarray(expected, dtype=float).reshape(4, 2)
        out = obs - exp
        out[~np.isfinite(out)] = math.nan
        return out

    @staticmethod
    def _side_lengths(points: Optional[np.ndarray]) -> np.ndarray:
        if points is None:
            return np.full(4, np.nan, dtype=float)
        pts = np.asarray(points, dtype=float).reshape(4, 2)
        if not np.all(np.isfinite(pts)):
            return np.full(4, np.nan, dtype=float)
        return np.array(
            [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[2] - pts[3]),
                np.linalg.norm(pts[3] - pts[0]),
            ],
            dtype=float,
        )

    @classmethod
    def _area_ratio(cls, observed: Optional[np.ndarray], expected: np.ndarray) -> float:
        obs_area = cls._quad_area(observed)
        exp_area = cls._quad_area(expected)
        if not math.isfinite(obs_area) or not math.isfinite(exp_area) or exp_area <= 1e-9:
            return math.nan
        return obs_area / exp_area

    @staticmethod
    def _quad_area(points: Optional[np.ndarray]) -> float:
        if points is None:
            return math.nan
        pts = np.asarray(points, dtype=float).reshape(4, 2)
        if not np.all(np.isfinite(pts)):
            return math.nan
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def _size_depth(
        keypoints_px: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        object_points_m: np.ndarray,
    ) -> float:
        if keypoints_px is None:
            return math.nan
        pts = np.asarray(keypoints_px, dtype=float).reshape(4, 2)
        if not np.all(np.isfinite(pts)):
            return math.nan
        obj = np.asarray(object_points_m, dtype=float).reshape(4, 3)
        gate_width_m = float(np.max(obj[:, 0]) - np.min(obj[:, 0]))
        gate_height_m = float(np.max(obj[:, 1]) - np.min(obj[:, 1]))
        k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        width_px = 0.5 * (
            np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])
        )
        height_px = 0.5 * (
            np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])
        )
        depths = []
        if width_px > 1.0 and gate_width_m > 0.0:
            depths.append(float(k[0, 0]) * gate_width_m / float(width_px))
        if height_px > 1.0 and gate_height_m > 0.0:
            depths.append(float(k[1, 1]) * gate_height_m / float(height_px))
        return float(np.mean(depths)) if depths else math.nan

    @staticmethod
    def _vec3(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size < 3:
            return None
        out = arr[:3].astype(float).copy()
        if not np.all(np.isfinite(out)):
            return None
        return out

    @classmethod
    def _vec3_array(cls, value: Any) -> np.ndarray:
        if not isinstance(value, (list, tuple)):
            return np.empty((0, 3), dtype=float)
        out = []
        for item in value:
            vec = cls._vec3(item)
            if vec is not None:
                out.append(vec)
        if not out:
            return np.empty((0, 3), dtype=float)
        return np.asarray(out, dtype=float).reshape(-1, 3)

    @staticmethod
    def _unit_vec3(value: Any, *, default: np.ndarray) -> np.ndarray:
        try:
            vec = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError):
            vec = np.asarray(default, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vec))
        if not math.isfinite(norm) or norm <= 1e-9:
            vec = np.asarray(default, dtype=float).reshape(3)
            norm = float(np.linalg.norm(vec))
        return vec / norm

    @staticmethod
    def _finite_float(value: Any, default: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if math.isfinite(out) else float(default)

    @staticmethod
    def _fmt_vec(value: Any, precision: int) -> str:
        if value is None:
            return "(nan,nan)"
        arr = np.asarray(value, dtype=float).reshape(-1)
        fmt = f"{{:.{int(precision)}f}}"
        return "(" + ",".join(fmt.format(float(v)) for v in arr) + ")"
