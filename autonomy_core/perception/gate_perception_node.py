import numpy as np


class GatePerceptionNode:
    """
    Wraps an existing GatePerception module and converts detections
    into world-frame gate center measurements.
    """

    def __init__(self, gate_perception, camera_to_body_rotmat=None):
        self.gate_perception = gate_perception
        self.last_pipeline_debug = {
            "yolo_raw_count": 0,
            "pnp_success_count": 0,
            "world_valid_count": 0,
            "processed_detection_indices": [],
        }
        if camera_to_body_rotmat is None:
            # OpenCV optical frame: x right, y down, z forward.
            # Body/world convention used here: x forward, y left, z up.
            camera_to_body_rotmat = np.array([
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ], dtype=float)

        self.R_body_camera = np.asarray(camera_to_body_rotmat, dtype=float).reshape(3, 3)

    @staticmethod
    def _yaw_to_rotmat(yaw: float) -> np.ndarray:
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

    @staticmethod
    def _rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Body-to-world rotation matrix using the same ZYX convention as the
        controller: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                  cp * cr],
        ], dtype=float)

    def detect_gate(
        self,
        frame,
        camera_matrix,
        dist_coeffs,
        drone_pos,
        drone_rpy_rad=None,
        drone_yaw_rad=None,
    ):
        """
        Returns a world-frame gate center estimate if detection is valid.

        Expected gate_perception output:
            {
                "confidence": float,
                "t": np.ndarray shape (3,1) or equivalent
            }

        If perception returns None, a string, or malformed output, return None.
        """
        perception = self.gate_perception.process(frame, camera_matrix, dist_coeffs)

        # No result
        if perception is None:
            return None

        # Sometimes perception code returns a status string like "No gate detected!"
        if isinstance(perception, str):
            return None

        # Must be dict-like
        if not isinstance(perception, dict):
            return None

        return self._perception_to_world_detection(
            perception=perception,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            drone_pos=drone_pos,
            drone_rpy_rad=drone_rpy_rad,
            drone_yaw_rad=drone_yaw_rad,
        )

    def detect_gates(
        self,
        frame,
        camera_matrix,
        dist_coeffs,
        drone_pos,
        drone_rpy_rad=None,
        drone_yaw_rad=None,
    ):
        if not hasattr(self.gate_perception, "process_all"):
            det = self.detect_gate(
                frame=frame,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                drone_pos=drone_pos,
                drone_rpy_rad=drone_rpy_rad,
                drone_yaw_rad=drone_yaw_rad,
            )
            self.last_pipeline_debug = {
                "yolo_raw_count": int(getattr(self.gate_perception, "last_yolo_detection_count", 0)),
                "pnp_success_count": 0 if det is None else 1,
                "world_valid_count": 0 if det is None else 1,
                "processed_detection_indices": [],
            }
            return [] if det is None else [det]

        perceptions = self.gate_perception.process_all(frame, camera_matrix, dist_coeffs)
        detections = []
        processed_indices = []
        for perception in perceptions:
            det = self._perception_to_world_detection(
                perception=perception,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                drone_pos=drone_pos,
                drone_rpy_rad=drone_rpy_rad,
                drone_yaw_rad=drone_yaw_rad,
            )
            if det is not None:
                detections.append(det)
                processed_indices.append(int(det.get("detection_index", len(processed_indices))))
        self.last_pipeline_debug = {
            "yolo_raw_count": int(getattr(self.gate_perception, "last_yolo_detection_count", 0)),
            "pnp_success_count": int(len(perceptions)),
            "world_valid_count": int(len(detections)),
            "processed_detection_indices": processed_indices,
        }
        return detections

    def _perception_to_world_detection(
        self,
        perception,
        camera_matrix,
        dist_coeffs,
        drone_pos,
        drone_rpy_rad=None,
        drone_yaw_rad=None,
    ):
        if perception is None or isinstance(perception, str) or not isinstance(perception, dict):
            return None

        conf = perception.get("confidence", None)
        t = perception.get("t", None)

        if conf is None or t is None:
            return None

        try:
            t = np.asarray(t, dtype=float).reshape(3, 1)
        except Exception:
            return None

        gate_camera = t.reshape(3)
        gate_body = self.R_body_camera @ gate_camera

        if drone_rpy_rad is not None:
            try:
                roll, pitch, yaw = np.asarray(drone_rpy_rad, dtype=float).reshape(3)
                R_wb = self._rpy_to_rotmat(float(roll), float(pitch), float(yaw))
            except Exception:
                return None
        elif drone_yaw_rad is not None:
            R_wb = self._yaw_to_rotmat(float(drone_yaw_rad))
        else:
            return None

        # user fix due to sdf inspection
        camera_pos_body = np.array([0.12, 0.03, 0.242])
        gate_world = np.asarray(drone_pos,dtype=float).reshape(3) + R_wb @ (camera_pos_body + gate_body)
        # end user fix
        debug = perception.get("debug", {})
        gate_size_sweep = {}
        for key, value in debug.get("gate_size_sweep", {}).items():
            cam = np.asarray(value.get("tvec", np.full(3, np.nan)), dtype=float).reshape(3)
            body = self.R_body_camera @ cam
            world = np.asarray(drone_pos, dtype=float).reshape(3) + R_wb @ (camera_pos_body + body)
            gate_size_sweep[key] = {
                "camera": cam,
                "body": body,
                "world": world,
                "reprojection_error": float(value.get("reprojection_error", np.nan)),
            }
        pnp_formulation_debug = []
        for value in debug.get("pnp_formulation_debug", []):
            cam = np.asarray(value.get("tvec", np.full(3, np.nan)), dtype=float).reshape(3)
            body = self.R_body_camera @ cam
            world = np.asarray(drone_pos, dtype=float).reshape(3) + R_wb @ (camera_pos_body + body)
            candidates = []
            for cand in value.get("candidates", []):
                cand_cam = np.asarray(cand.get("tvec", np.full(3, np.nan)), dtype=float).reshape(3)
                cand_body = self.R_body_camera @ cand_cam
                cand_world = np.asarray(drone_pos, dtype=float).reshape(3) + R_wb @ (camera_pos_body + cand_body)
                candidates.append({
                    "index": int(cand.get("index", -1)),
                    "camera": cand_cam,
                    "body": cand_body,
                    "world": cand_world,
                    "rvec": np.asarray(cand.get("rvec", np.full(3, np.nan)), dtype=float).reshape(3),
                    "normal": np.asarray(cand.get("normal", np.full(3, np.nan)), dtype=float).reshape(3),
                    "error": float(cand.get("error", np.nan)),
                    "projected_corners": cand.get("projected_corners", None),
                })
            pnp_formulation_debug.append({
                "solver": value.get("solver", ""),
                "order": value.get("order", ""),
                "chosen_candidate": value.get("chosen_candidate", None),
                "candidate_count": int(value.get("candidate_count", 0)),
                "camera": cam,
                "body": body,
                "world": world,
                "rvec": np.asarray(value.get("rvec", np.full(3, np.nan)), dtype=float).reshape(3),
                "normal": np.asarray(value.get("normal", np.full(3, np.nan)), dtype=float).reshape(3),
                "reprojection_error": float(value.get("reprojection_error", np.nan)),
                "candidates": candidates,
            })
        gate_normal_camera = np.asarray(
            debug.get("gate_normal_camera", np.array([np.nan, np.nan, np.nan])),
            dtype=float,
        ).reshape(3)
        gate_normal_body = self.R_body_camera @ gate_normal_camera
        gate_normal_world = R_wb @ gate_normal_body

        return {
            "confidence": float(conf),
            "yolo_confidence": float(
                perception.get(
                    "yolo_confidence",
                    debug.get("yolo_box_confidence", conf),
                )
            ),
            "quad_area_px2": float(
                perception.get("quad_area_px2", debug.get("quad_area_px2", np.nan))
            ),
            "quad_area_confidence": float(
                perception.get(
                    "quad_area_confidence",
                    debug.get("quad_area_confidence", conf),
                )
            ),
            "old_area_confidence": float(
                perception.get(
                    "old_area_confidence",
                    debug.get("old_area_confidence", conf),
                )
            ),
            "memory_confidence": float(
                perception.get(
                    "memory_confidence",
                    perception.get(
                        "yolo_confidence",
                        debug.get("yolo_box_confidence", conf),
                    ),
                )
            ),
            "gate_center_camera": gate_camera,
            "gate_center_body": gate_body,
            "gate_center_cam": gate_body,
            "gate_center_world": gate_world,
            "camera_to_body_matrix_used": self.R_body_camera.copy(),
            "body_to_world_method_used": "gate_perception_node_default",
            "drone_pos": np.asarray(drone_pos, dtype=float).reshape(3),
            "drone_yaw_rad": float(drone_yaw_rad) if drone_yaw_rad is not None else float(yaw),
            "gate_normal_camera": gate_normal_camera,
            "gate_normal_body": gate_normal_body,
            "gate_normal_world": gate_normal_world,
            "reprojection_error": float(debug.get("reprojection_error", np.nan)),
            "corner_reprojection_error_px": float(debug.get("corner_reprojection_error_px", np.nan)),
            "raw_corners": debug.get("raw_corners", None),
            "ordered_corners": debug.get("ordered_corners", None),
            "pnp_debug_best_ordered_corners": debug.get("pnp_debug_best_ordered_corners", None),
            "reprojected_corners": debug.get("reprojected_corners", None),
            "rvec": debug.get("rvec", None),
            "tvec": debug.get("tvec", gate_camera),
            "pnp_candidates": debug.get("pnp_candidates", []),
            "chosen_candidate": debug.get("chosen_candidate", None),
            "live_solver_name": debug.get("live_solver_name", ""),
            "pnp_fallback_reason": debug.get("pnp_fallback_reason", ""),
            "pnp_selected_order": debug.get("pnp_selected_order", ""),
            "pnp_selected_solver": debug.get("pnp_selected_solver", ""),
            "pnp_selected_score": float(debug.get("pnp_selected_score", np.nan)),
            "pnp_selected_reprojection_error": float(debug.get("pnp_selected_reprojection_error", np.nan)),
            "pnp_selected_gate_center_camera": np.asarray(
                debug.get("pnp_selected_gate_center_camera", gate_camera),
                dtype=float,
            ).reshape(3),
            "pnp_selected_reason": debug.get("pnp_selected_reason", ""),
            "pnp_candidate_summary": debug.get("pnp_candidate_summary", ""),
            "allow_pnp_corner_reordering": bool(debug.get("allow_pnp_corner_reordering", False)),
            "pnp_live_candidate_orders_allowed": debug.get("pnp_live_candidate_orders_allowed", ""),
            "pnp_debug_best_order": debug.get("pnp_debug_best_order", ""),
            "pnp_live_vs_debug_best_order_mismatch": bool(
                debug.get("pnp_live_vs_debug_best_order_mismatch", False)
            ),
            "pnp_lateral_angle": float(debug.get("pnp_lateral_angle", np.nan)),
            "image_center_offset_normalized": float(
                debug.get("image_center_offset_normalized", np.nan)
            ),
            "keypoint_polygon_signed_area": float(debug.get("keypoint_polygon_signed_area", np.nan)),
            "keypoint_polygon_winding": debug.get("keypoint_polygon_winding", ""),
            "keypoint_edge_top": float(debug.get("keypoint_edge_top", np.nan)),
            "keypoint_edge_right": float(debug.get("keypoint_edge_right", np.nan)),
            "keypoint_edge_bottom": float(debug.get("keypoint_edge_bottom", np.nan)),
            "keypoint_edge_left": float(debug.get("keypoint_edge_left", np.nan)),
            "keypoint_bbox_center": np.asarray(
                debug.get("keypoint_bbox_center", np.full(2, np.nan)),
                dtype=float,
            ).reshape(2),
            "keypoint_polygon_center": np.asarray(
                debug.get("keypoint_polygon_center", np.full(2, np.nan)),
                dtype=float,
            ).reshape(2),
            "keypoint_bbox_polygon_delta": np.asarray(
                debug.get("keypoint_bbox_polygon_delta", np.full(2, np.nan)),
                dtype=float,
            ).reshape(2),
            "raw_keypoint_polygon_signed_area": float(
                debug.get("raw_keypoint_polygon_signed_area", np.nan)
            ),
            "raw_keypoint_polygon_winding": debug.get("raw_keypoint_polygon_winding", ""),
            "gate_size_sweep": gate_size_sweep,
            "pnp_formulation_debug": pnp_formulation_debug,
            "detection_index": int(debug.get("detection_index", -1)),
            "processed_detection_index": int(debug.get("processed_detection_index", -1)),
            "yolo_bbox": debug.get("yolo_bbox", None),
            "yolo_box_confidence": float(debug.get("yolo_box_confidence", np.nan)),
            "yolo_keypoints": debug.get("yolo_keypoints", None),
            "raw": perception,
        }
