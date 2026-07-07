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
    official_camera_to_body_frd_rotmat,
)
from perception_geometry_audit import PerceptionGeometryAudit
from runtime_config import load_runtime_config


class PerceptionWrapper:
    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        camera_matrix=None,
        dist_coeffs=None,
        gate_perception=None,
        yolo_model_path: Optional[str] = None,
        yolo_conf: Optional[float] = None,
        yolo_imgsz: Optional[int] = None,
        yolo_device: Optional[int | str] = None,
        preprocess_mode: Optional[str] = None,
        config=None,
    ):
        self.config = config if config is not None else load_runtime_config()
        perception_config = self.config.perception
        self.transform_mode = str(perception_config.transform_mode)
        self.world_pose_source = str(perception_config.world_pose_source).lower()
        self.camera_matrix = self._matrix3(camera_matrix, self.config.camera.matrix)
        self.dist_coeffs = self._dist_coeffs(dist_coeffs, self.config.camera.dist_coeffs)

        if backend is None:
            backend = perception_config.backend
        self.backend = str(backend).lower()

        if gate_perception is None:
            gate_perception = self._create_gate_perception(
                yolo_model_path=yolo_model_path
                or perception_config.yolo_model_path,
                yolo_conf=yolo_conf
                if yolo_conf is not None
                else perception_config.yolo_conf,
                yolo_imgsz=yolo_imgsz
                if yolo_imgsz is not None
                else perception_config.yolo_imgsz,
                yolo_device=yolo_device
                if yolo_device is not None
                else perception_config.yolo_device,
                preprocess_mode=preprocess_mode
                or perception_config.preprocess_mode,
                yolo_keypoint_order=perception_config.yolo_keypoint_order,
                yolo_keypoint_layout=perception_config.yolo_keypoint_layout,
                gate_size_m=perception_config.gate_size_m,
            )

        self.gate_perception = gate_perception
        camera_to_body = official_camera_to_body_frd_rotmat(VADR_TS_002)
        self.camera_translation_body = np.asarray(
            self.config.camera.body_translation_m,
            dtype=float,
        ).reshape(3)
        self.perception_yaw_correction_deg = float(
            self.config.camera.yaw_correction_deg
        )
        self.perception_yaw_correction_rad = math.radians(
            self.perception_yaw_correction_deg
        )
        self.camera_to_body = np.asarray(camera_to_body, dtype=float).reshape(3, 3)
        self.object_points_m = np.asarray(
            getattr(self.gate_perception, "model_points", VADR_TS_002.gate_inner_object_points_m),
            dtype=float,
        ).reshape(-1, 3)
        self.audit_object_points_m = self.object_points_m[:4].copy()
        audit_config = self.config.perception_geometry_audit
        self.gazebo_rotation_mode = str(
            getattr(audit_config, "gazebo_rotation_mode", "transpose")
        ).lower()
        self.gazebo_optical_mode = str(
            getattr(audit_config, "gazebo_optical_mode", "physical_minus_y")
        ).lower()
        self.gazebo_camera_sim_max_pose_age_s = 0.25
        self._last_gazebo_camera_sim_warning_s = -math.inf
        self.geometry_audit = PerceptionGeometryAudit(
            audit_config
        )
        print(
            "[PERCEPTION_CONFIG] "
            f"backend={self.backend} "
            f"transform_mode={self.transform_mode} "
            f"world_pose_source={self.world_pose_source} "
            f"yolo_keypoint_order={perception_config.yolo_keypoint_order} "
            f"yolo_keypoint_layout={perception_config.yolo_keypoint_layout} "
            f"camera_mount_profile={self.config.camera.mount_profile} "
            f"perception_yaw_correction_deg={self.perception_yaw_correction_deg:.3f} "
            f"K={self._fmt_array(self.camera_matrix, precision=3)} "
            f"dist={self._fmt_array(self.dist_coeffs, precision=4)} "
            f"camera_to_body={self._fmt_array(self.camera_to_body, precision=4)} "
            f"camera_translation_body={self._fmt_array(self.camera_translation_body, precision=4)} "
            f"object_points_m={self._fmt_array(self.object_points_m, precision=3)}",
            flush=True,
        )

    @staticmethod
    def _matrix3(value, default) -> np.ndarray:
        if value is None:
            return np.asarray(default, dtype=float).reshape(3, 3).copy()
        return np.asarray(value, dtype=float).reshape(3, 3).copy()

    @staticmethod
    def _dist_coeffs(value, default) -> np.ndarray:
        if value is None:
            return np.asarray(default, dtype=float).reshape(-1).copy()
        return np.asarray(value, dtype=float).reshape(-1).copy()

    def _create_gate_perception(
        self,
        *,
        yolo_model_path: Optional[str],
        yolo_conf: float,
        yolo_imgsz: int,
        yolo_device: Optional[int | str],
        preprocess_mode: str,
        yolo_keypoint_order: str,
        yolo_keypoint_layout: str,
        gate_size_m: float,
    ):
        if self.backend in ("yolo", "pose", "yolo_pose"):
            from autonomy_core.perception.gate_perception_yolo import GatePerception

            return GatePerception(
                gate_size=float(gate_size_m),
                yolo_model_path=yolo_model_path,
                yolo_conf=float(yolo_conf),
                yolo_imgsz=int(yolo_imgsz),
                yolo_device=yolo_device,
                preprocess_mode=str(preprocess_mode),
                keypoint_order=str(yolo_keypoint_order),
                keypoint_layout=str(yolo_keypoint_layout),
            )

        if self.backend in ("orange", "hsv_orange"):
            from autonomy_core.perception.gate_perception_orange import GatePerception

            return GatePerception(gate_size=VADR_TS_002.gate_outer_square_m)

        if self.backend not in ("blue", "hsv", "hsv_blue"):
            raise ValueError(f"Unsupported perception.backend={self.backend!r}")

        from autonomy_core.perception.gate_perception import GatePerception

        return GatePerception(gate_size=VADR_TS_002.gate_outer_square_m)

    def update(
        self,
        *,
        frame,
        attitude=None,
        odometry=None,
        local_position_ned=None,
        gazebo_pose=None,
    ) -> dict[str, Any]:
        frame_data = self._frame_data(frame)
        image = frame_data.get("image")
        frame_gazebo_pose = frame_data.get("gazebo_pose")
        if isinstance(frame_gazebo_pose, dict):
            gazebo_pose = frame_gazebo_pose
        perception_wall_time = time.time()

        latest_perception = self._empty_latest_perception(
            frame_data=frame_data,
            perception_wall_time=perception_wall_time,
        )
        latest_perception["gazebo_pose"] = gazebo_pose

        if image is None:
            return latest_perception

        position_source = odometry
        drone_pos_ned = self._position_ned(position_source)
        if drone_pos_ned is None:
            position_source = local_position_ned
            drone_pos_ned = self._position_ned(position_source)
        attitude_source = attitude
        drone_rpy = self._attitude_rpy(attitude_source)
        if drone_rpy is None:
            attitude_source = odometry
            drone_rpy = self._odometry_rpy(attitude_source)

        try:
            raw_detections = self._detect_camera_only(image)
            if self._should_project_with_gazebo_camera():
                raw_detections = self._project_detections_with_gazebo_camera(
                    raw_detections,
                    gazebo_pose=gazebo_pose,
                    drone_pos_ned=drone_pos_ned,
                    drone_rpy_rad=drone_rpy,
                    frame_id=int(latest_perception.get("frame_id", -1)),
                    image_wall_time=self._float_or_none(frame_data.get("wall_time")),
                    image_ros_stamp_sec=self._int_or_none(frame_data.get("ros_stamp_sec")),
                    image_ros_stamp_nanosec=self._int_or_none(
                        frame_data.get("ros_stamp_nanosec")
                    ),
                    attitude_wall_time=self._wall_time(attitude_source),
                    position_wall_time=self._wall_time(position_source),
                )
            elif (
                self._should_project_with_mavlink_pose()
                and drone_pos_ned is not None
                and drone_rpy is not None
            ):
                raw_detections = self._project_detections_to_world(
                    raw_detections,
                    drone_pos_ned=drone_pos_ned,
                    drone_rpy_rad=drone_rpy,
                    frame_id=int(latest_perception.get("frame_id", -1)),
                    gazebo_pose=gazebo_pose,
                    image_wall_time=self._float_or_none(frame_data.get("wall_time")),
                    image_ros_stamp_sec=self._int_or_none(frame_data.get("ros_stamp_sec")),
                    image_ros_stamp_nanosec=self._int_or_none(
                        frame_data.get("ros_stamp_nanosec")
                    ),
                    attitude_wall_time=self._wall_time(attitude_source),
                    position_wall_time=self._wall_time(position_source),
                )
        except Exception as exc:
            print(f"[perception_wrapper] update failed: {exc}", flush=True)
            raw_detections = []

        latest_perception["perception_wall_time"] = time.time()
        latest_perception["detections"] = [
            self._normalize_detection(detection, index)
            for index, detection in enumerate(raw_detections)
        ]
        return latest_perception

    def _should_project_with_gazebo_camera(self) -> bool:
        return (
            self.world_pose_source == "gazebo_camera_sim"
            and str(self.config.runtime.runner_mode).lower() != "competition"
        )

    def _should_project_with_mavlink_pose(self) -> bool:
        if str(self.config.state_estimation.mode).lower() == "estimator":
            return False
        source = str(self.world_pose_source).lower()
        return source not in ("camera_only", "none", "estimator", "gazebo_camera_sim")

    def _empty_latest_perception(self, *, frame_data, perception_wall_time: float) -> dict[str, Any]:
        return {
            "frame_id": int(frame_data.get("frame_id", -1)),
            "image_sim_time_ns": frame_data.get("sim_time_ns"),
            "image_wall_time": float(frame_data.get("wall_time", 0.0)),
            "image_ros_stamp_sec": frame_data.get("ros_stamp_sec"),
            "image_ros_stamp_nanosec": frame_data.get("ros_stamp_nanosec"),
            "perception_wall_time": float(perception_wall_time),
            "camera_matrix": self.camera_matrix.copy(),
            "dist_coeffs": self.dist_coeffs.copy(),
            "camera_to_body": self.camera_to_body.copy(),
            "camera_translation_body": self.camera_translation_body.copy(),
            "perception_yaw_correction_rad": float(self.perception_yaw_correction_rad),
            "perception_yaw_correction_deg": float(self.perception_yaw_correction_deg),
            "world_frame": (
                "gazebo_camera_sim_neu"
                if self._should_project_with_gazebo_camera()
                else "mavlink_local_ned_projected_to_neu"
            ),
            "body_frame": "mavlink_body_frd",
            "transform_mode": self.transform_mode,
            "world_pose_source": self.world_pose_source,
            "gazebo_pose": frame_data.get("gazebo_pose"),
            "detections": [],
        }

    @staticmethod
    def _frame_data(frame) -> dict[str, Any]:
        if isinstance(frame, dict):
            return frame
        if frame is None:
            return {}
        return {
            "frame_id": -1,
            "image": frame,
            "shape": getattr(frame, "shape", None),
            "sim_time_ns": None,
            "wall_time": time.time(),
        }

    def _detect_camera_only(self, image) -> list[dict[str, Any]]:
        if hasattr(self.gate_perception, "process_all"):
            perceptions = self.gate_perception.process_all(
                image,
                self.camera_matrix,
                self.dist_coeffs,
            )
        else:
            result = self.gate_perception.process(
                image,
                self.camera_matrix,
                self.dist_coeffs,
            )
            perceptions = [] if result is None or isinstance(result, str) else [result]

        detections = []
        for perception in perceptions:
            if not isinstance(perception, dict):
                continue
            debug = perception.get("debug", {})
            gate_camera = self._vec3(
                perception.get("t", debug.get("tvec", None)),
                default=np.full(3, np.nan),
            )
            detections.append({
                "confidence": float(perception.get("confidence", 0.0)),
                "yolo_confidence": float(
                    perception.get(
                        "yolo_confidence",
                        debug.get("yolo_box_confidence", perception.get("confidence", 0.0)),
                    )
                ),
                "memory_confidence": float(
                    perception.get(
                        "memory_confidence",
                        perception.get(
                            "yolo_confidence",
                            debug.get("yolo_box_confidence", perception.get("confidence", 0.0)),
                        ),
                    )
                ),
                "quad_area_px2": float(
                    perception.get("quad_area_px2", debug.get("quad_area_px2", np.nan))
                ),
                "quad_area_confidence": float(
                    perception.get(
                        "quad_area_confidence",
                        debug.get("quad_area_confidence", perception.get("confidence", 0.0)),
                    )
                ),
                "old_area_confidence": float(
                    perception.get(
                        "old_area_confidence",
                        debug.get("old_area_confidence", perception.get("confidence", 0.0)),
                    )
                ),
                "gate_center_camera": gate_camera,
                "gate_center_body": self.camera_to_body @ gate_camera,
                "gate_center_body_frd": self.camera_to_body @ gate_camera,
                "gate_center_world": None,
                "gate_center_world_ned": None,
                "reprojection_error": float(debug.get("reprojection_error", np.nan)),
                "reprojected_corners": debug.get("reprojected_corners", None),
                "ordered_corners": debug.get("ordered_corners"),
                "raw_corners": debug.get("raw_corners"),
                "gate_normal_camera": debug.get("gate_normal_camera", None),
                "rvec": debug.get("rvec"),
                "tvec": debug.get("tvec", gate_camera),
                "yolo_keypoints": debug.get("yolo_keypoints", None),
                "yolo_bbox": debug.get("yolo_bbox", None),
                "pnp_candidates": debug.get("pnp_candidates", ()),
                "pnp_candidate_summary": debug.get("pnp_candidate_summary", ""),
                "pnp_formulation_debug": debug.get("pnp_formulation_debug", ()),
                "pnp_selected_order": debug.get("pnp_selected_order", ""),
                "pnp_selected_solver": debug.get("pnp_selected_solver", ""),
                "pnp_selected_score": self._float_or_default(
                    debug.get("pnp_selected_score"),
                    np.nan,
                    allow_nan=True,
                ),
                "pnp_selected_reprojection_error": self._float_or_default(
                    debug.get("pnp_selected_reprojection_error"),
                    np.nan,
                    allow_nan=True,
                ),
                "pnp_selected_reason": debug.get("pnp_selected_reason", ""),
                "pnp_debug_best_order": debug.get("pnp_debug_best_order", ""),
                "pnp_live_vs_debug_best_order_mismatch": bool(
                    debug.get("pnp_live_vs_debug_best_order_mismatch", False)
                ),
                "allow_pnp_corner_reordering": bool(
                    debug.get("allow_pnp_corner_reordering", False)
                ),
                "keypoint_polygon_winding": debug.get("keypoint_polygon_winding", ""),
                "detection_index": int(debug.get("detection_index", -1)),
                "processed_detection_index": int(debug.get("processed_detection_index", -1)),
                "camera_to_body_matrix_used": self.camera_to_body.copy(),
                "body_to_world_method_used": "camera_only",
            })
        return detections

    def _project_detections_with_gazebo_camera(
        self,
        detections: list[dict[str, Any]],
        *,
        gazebo_pose=None,
        drone_pos_ned: Optional[np.ndarray] = None,
        drone_rpy_rad: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
        image_wall_time: Optional[float] = None,
        image_ros_stamp_sec: Optional[int] = None,
        image_ros_stamp_nanosec: Optional[int] = None,
        attitude_wall_time: Optional[float] = None,
        position_wall_time: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        pose_ok, reason = self._gazebo_camera_pose_ok(gazebo_pose)
        if not pose_ok:
            self._maybe_warn_gazebo_camera_sim(
                f"gazebo_camera_sim requested but Gazebo camera pose is unavailable: {reason}"
            )
            return detections

        projected = [
            self._project_detection_to_world_gazebo_camera(
                detection,
                gazebo_pose=gazebo_pose,
                drone_pos_ned=drone_pos_ned,
                drone_rpy_rad=drone_rpy_rad,
            )
            for detection in detections
        ]

        if drone_pos_ned is not None and drone_rpy_rad is not None:
            try:
                self.geometry_audit.maybe_print(
                    projected,
                    drone_pos_ned=np.asarray(drone_pos_ned, dtype=float).reshape(3),
                    drone_rpy_rad=self._perception_rpy(drone_rpy_rad),
                    camera_matrix=self.camera_matrix,
                    camera_to_body=self.camera_to_body,
                    camera_translation_body=self.camera_translation_body,
                    object_points_m=self.audit_object_points_m,
                    frame_id=frame_id,
                    gazebo_pose=gazebo_pose,
                    image_wall_time=image_wall_time,
                    image_ros_stamp_sec=image_ros_stamp_sec,
                    image_ros_stamp_nanosec=image_ros_stamp_nanosec,
                    attitude_wall_time=attitude_wall_time,
                    position_wall_time=position_wall_time,
                )
            except Exception as exc:
                print(f"[GEOM_AUDIT] disabled_after_error={exc}", flush=True)
                self.geometry_audit.enabled = False
        return projected

    def _project_detection_to_world_gazebo_camera(
        self,
        detection: dict[str, Any],
        *,
        gazebo_pose: dict[str, Any],
        drone_pos_ned: Optional[np.ndarray] = None,
        drone_rpy_rad: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        out = dict(detection)
        gate_camera = self._vec3(out.get("gate_center_camera"), default=None)
        if gate_camera is None:
            return out

        gate_world_neu = self._world_from_camera_gazebo(gate_camera, gazebo_pose)
        if gate_world_neu is None:
            return out

        gate_body_frd = self.camera_to_body @ gate_camera
        out["gate_center_body"] = gate_body_frd.copy()
        out["gate_center_body_frd"] = gate_body_frd.copy()
        out["gate_center_world"] = gate_world_neu.copy()
        out["gate_center_world_ned"] = local_neu_to_ned(gate_world_neu)
        out["world_pose_source"] = "gazebo_camera_sim"
        out["body_to_world_method_used"] = (
            f"gazebo_camera_sim:{self.gazebo_rotation_mode}:{self.gazebo_optical_mode}"
        )
        out["camera_to_body_matrix_used"] = self.camera_to_body.copy()
        out["camera_translation_body_used"] = self.camera_translation_body.copy()
        out["perception_yaw_correction_rad"] = float(self.perception_yaw_correction_rad)
        out["perception_yaw_correction_deg"] = float(self.perception_yaw_correction_deg)
        camera_pos_world = self._vec3(
            gazebo_pose.get("gazebo_camera_pos_world"),
            default=None,
        )
        if camera_pos_world is not None:
            out["gazebo_camera_pos_world"] = camera_pos_world.copy()
            out["gazebo_camera_pos_neu"] = self._gazebo_world_to_neu(camera_pos_world)
        out["gazebo_camera_quat_world"] = self._quat4(
            gazebo_pose.get("gazebo_camera_quat_world")
        )
        out["gazebo_pose_wall_time"] = self._float_or_none(
            gazebo_pose.get("gazebo_pose_wall_time")
        )
        out["gazebo_pose_selection_method"] = str(
            gazebo_pose.get("gazebo_pose_selection_method", "")
        )
        if drone_pos_ned is not None:
            pos_ned = np.asarray(drone_pos_ned, dtype=float).reshape(3)
            out["drone_pos_ned"] = pos_ned.copy()
            out["drone_pos_neu"] = local_ned_to_neu(pos_ned)
        if drone_rpy_rad is not None:
            raw_rpy = np.asarray(drone_rpy_rad, dtype=float).reshape(3)
            out["drone_rpy_rad_mavlink"] = raw_rpy.copy()
            out["drone_rpy_rad_used"] = self._perception_rpy(raw_rpy)

        gate_normal_camera = self._vec3(out.get("gate_normal_camera"), default=None)
        if gate_normal_camera is not None:
            normal_world_neu = self._vector_from_camera_gazebo(
                gate_normal_camera,
                gazebo_pose,
            )
            if normal_world_neu is not None:
                out["gate_normal_world"] = normal_world_neu.copy()
                out["gate_normal_world_ned"] = local_neu_to_ned(normal_world_neu)

        return out

    def _project_detections_to_world(
        self,
        detections: list[dict[str, Any]],
        *,
        drone_pos_ned: np.ndarray,
        drone_rpy_rad: np.ndarray,
        frame_id: Optional[int] = None,
        gazebo_pose=None,
        image_wall_time: Optional[float] = None,
        image_ros_stamp_sec: Optional[int] = None,
        image_ros_stamp_nanosec: Optional[int] = None,
        attitude_wall_time: Optional[float] = None,
        position_wall_time: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        drone_rpy_used = self._perception_rpy(drone_rpy_rad)
        projected = [
            self._project_detection_to_world(
                detection,
                drone_pos_ned=drone_pos_ned,
                drone_rpy_rad=drone_rpy_used,
                drone_rpy_raw_rad=drone_rpy_rad,
            )
            for detection in detections
        ]
        try:
            self.geometry_audit.maybe_print(
                projected,
                drone_pos_ned=np.asarray(drone_pos_ned, dtype=float).reshape(3),
                drone_rpy_rad=drone_rpy_used,
                camera_matrix=self.camera_matrix,
                camera_to_body=self.camera_to_body,
                camera_translation_body=self.camera_translation_body,
                object_points_m=self.audit_object_points_m,
                frame_id=frame_id,
                gazebo_pose=gazebo_pose,
                image_wall_time=image_wall_time,
                image_ros_stamp_sec=image_ros_stamp_sec,
                image_ros_stamp_nanosec=image_ros_stamp_nanosec,
                attitude_wall_time=attitude_wall_time,
                position_wall_time=position_wall_time,
            )
        except Exception as exc:
            print(f"[GEOM_AUDIT] disabled_after_error={exc}", flush=True)
            self.geometry_audit.enabled = False
        return projected

    def _project_detection_to_world(
        self,
        detection: dict[str, Any],
        *,
        drone_pos_ned: np.ndarray,
        drone_rpy_rad: np.ndarray,
        drone_rpy_raw_rad: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        out = dict(detection)
        gate_camera = self._vec3(out.get("gate_center_camera"), default=None)
        if gate_camera is None:
            return out

        roll, pitch, yaw = np.asarray(drone_rpy_rad, dtype=float).reshape(3)
        rot_ned_body = body_frd_to_local_ned_rotmat(roll, pitch, yaw)
        drone_pos_ned = np.asarray(drone_pos_ned, dtype=float).reshape(3)
        gate_body_frd = self.camera_to_body @ gate_camera
        gate_world_ned = drone_pos_ned + rot_ned_body @ (
            self.camera_translation_body + gate_body_frd
        )
        gate_world_neu = local_ned_to_neu(gate_world_ned)

        out["gate_center_body"] = gate_body_frd.copy()
        out["gate_center_body_frd"] = gate_body_frd.copy()
        out["gate_center_world_ned"] = gate_world_ned.copy()
        out["gate_center_world"] = gate_world_neu.copy()
        out["drone_pos_ned"] = drone_pos_ned.copy()
        out["drone_pos_neu"] = local_ned_to_neu(drone_pos_ned)
        if drone_rpy_raw_rad is None:
            drone_rpy_raw_rad = drone_rpy_rad
        out["drone_rpy_rad_mavlink"] = np.asarray(drone_rpy_raw_rad, dtype=float).reshape(3).copy()
        out["drone_rpy_rad_used"] = np.asarray(drone_rpy_rad, dtype=float).reshape(3).copy()
        out["perception_yaw_correction_rad"] = float(self.perception_yaw_correction_rad)
        out["perception_yaw_correction_deg"] = float(self.perception_yaw_correction_deg)
        out["camera_to_body_matrix_used"] = self.camera_to_body.copy()
        out["camera_translation_body_used"] = self.camera_translation_body.copy()
        out["body_to_world_method_used"] = (
            f"mavlink_body_frd_to_local_ned_to_neu:{self.transform_mode}"
        )

        gate_normal_camera = self._vec3(out.get("gate_normal_camera"), default=None)
        if gate_normal_camera is not None:
            normal_body_frd = self.camera_to_body @ gate_normal_camera
            normal_world_ned = rot_ned_body @ normal_body_frd
            out["gate_normal_body"] = normal_body_frd.copy()
            out["gate_normal_body_frd"] = normal_body_frd.copy()
            out["gate_normal_world_ned"] = normal_world_ned.copy()
            out["gate_normal_world"] = local_ned_to_neu(normal_world_ned)

        return out

    def _perception_rpy(self, drone_rpy_rad: np.ndarray) -> np.ndarray:
        rpy = np.asarray(drone_rpy_rad, dtype=float).reshape(3).copy()
        rpy[2] += float(self.perception_yaw_correction_rad)
        return rpy

    def _normalize_detection(self, detection: dict[str, Any], index: int) -> dict[str, Any]:
        detection_id = self._detection_id(detection, index)

        keypoints_px, keypoint_conf = self._keypoints_from_detection(detection)
        gate_camera = self._vec3(detection.get("gate_center_camera"), default=np.full(3, np.nan))
        gate_body = self._vec3(
            detection.get("gate_center_body"),
            default=self.camera_to_body @ gate_camera,
        )
        gate_world = detection.get("gate_center_world")
        gate_world = None if gate_world is None else self._vec3(gate_world, default=None)
        gate_world_ned = detection.get("gate_center_world_ned")
        gate_world_ned = None if gate_world_ned is None else self._vec3(gate_world_ned, default=None)
        gate_normal_camera = self._vec3(detection.get("gate_normal_camera"), default=None)
        gate_normal_body = self._vec3(detection.get("gate_normal_body"), default=None)
        if gate_normal_body is None and gate_normal_camera is not None:
            gate_normal_body = self.camera_to_body @ gate_normal_camera
        gate_normal_body_frd = self._vec3(
            detection.get("gate_normal_body_frd"),
            default=gate_normal_body,
        )
        gate_normal_world = detection.get("gate_normal_world")
        gate_normal_world = (
            None if gate_normal_world is None else self._vec3(gate_normal_world, default=None)
        )
        gate_normal_world_ned = detection.get("gate_normal_world_ned")
        gate_normal_world_ned = (
            None
            if gate_normal_world_ned is None
            else self._vec3(gate_normal_world_ned, default=None)
        )
        area_confidence = self._float_or_default(detection.get("confidence"), 0.0)
        yolo_confidence = self._float_or_default(
            detection.get("yolo_confidence"),
            area_confidence,
        )
        memory_confidence = self._float_or_default(
            detection.get("memory_confidence"),
            yolo_confidence,
        )

        return {
            "detection_id": detection_id,
            "associated_gate_id": None,
            "keypoints_px": keypoints_px,
            "keypoint_conf": keypoint_conf,
            "object_points_m": self.object_points_m.copy(),
            "rvec": self._vec3(detection.get("rvec"), default=np.full(3, np.nan)),
            "tvec": self._vec3(detection.get("tvec"), default=gate_camera),
            "gate_center_camera": gate_camera,
            "gate_center_body": gate_body,
            "gate_center_body_frd": self._vec3(
                detection.get("gate_center_body_frd"),
                default=gate_body,
            ),
            "gate_center_world": gate_world,
            "gate_center_world_ned": gate_world_ned,
            "gate_normal_camera": gate_normal_camera,
            "gate_normal_body": gate_normal_body,
            "gate_normal_body_frd": gate_normal_body_frd,
            "gate_normal_world": gate_normal_world,
            "gate_normal_world_ned": gate_normal_world_ned,
            "confidence": area_confidence,
            "memory_confidence": memory_confidence,
            "yolo_confidence": yolo_confidence,
            "quad_area_confidence": self._float_or_default(
                detection.get("quad_area_confidence"),
                area_confidence,
            ),
            "old_area_confidence": self._float_or_default(
                detection.get("old_area_confidence"),
                area_confidence,
            ),
            "quad_area_px2": self._float_or_default(
                detection.get("quad_area_px2"),
                np.nan,
                allow_nan=True,
            ),
            "reprojection_error": float(detection.get("reprojection_error", np.nan)),
            "pnp_selected_order": str(detection.get("pnp_selected_order", "")),
            "pnp_selected_solver": str(detection.get("pnp_selected_solver", "")),
            "pnp_selected_score": self._float_or_default(
                detection.get("pnp_selected_score"),
                np.nan,
                allow_nan=True,
            ),
            "pnp_selected_reprojection_error": self._float_or_default(
                detection.get("pnp_selected_reprojection_error"),
                np.nan,
                allow_nan=True,
            ),
            "pnp_selected_reason": str(detection.get("pnp_selected_reason", "")),
            "pnp_debug_best_order": str(detection.get("pnp_debug_best_order", "")),
            "pnp_live_vs_debug_best_order_mismatch": bool(
                detection.get("pnp_live_vs_debug_best_order_mismatch", False)
            ),
            "allow_pnp_corner_reordering": bool(
                detection.get("allow_pnp_corner_reordering", False)
            ),
            "keypoint_polygon_winding": str(detection.get("keypoint_polygon_winding", "")),
            "drone_pos_ned": self._vec3(detection.get("drone_pos_ned"), default=None),
            "drone_pos_neu": self._vec3(detection.get("drone_pos_neu"), default=None),
            "drone_rpy_rad_used": self._vec3(
                detection.get("drone_rpy_rad_used", detection.get("drone_rpy_rad_mavlink")),
                default=None,
            ),
            "perception_yaw_correction_rad": self._float_or_default(
                detection.get("perception_yaw_correction_rad"),
                self.perception_yaw_correction_rad,
                allow_nan=False,
            ),
            "perception_yaw_correction_deg": self._float_or_default(
                detection.get("perception_yaw_correction_deg"),
                self.perception_yaw_correction_deg,
                allow_nan=False,
            ),
            "camera_translation_body_used": self._vec3(
                detection.get("camera_translation_body_used"),
                default=None,
            ),
            "gazebo_camera_pos_neu": self._vec3(
                detection.get("gazebo_camera_pos_neu"),
                default=None,
            ),
            "gazebo_camera_pos_world": self._vec3(
                detection.get("gazebo_camera_pos_world"),
                default=None,
            ),
            "gazebo_camera_quat_world": self._quat4(
                detection.get("gazebo_camera_quat_world")
            ),
            "gazebo_pose_wall_time": self._float_or_none(
                detection.get("gazebo_pose_wall_time")
            ),
            "gazebo_pose_selection_method": str(
                detection.get("gazebo_pose_selection_method", "")
            ),
            "camera_to_body_matrix_used": self.camera_to_body.copy(),
            "body_to_world_method_used": detection.get("body_to_world_method_used", ""),
        }

    @staticmethod
    def _detection_id(detection: dict[str, Any], index: int) -> int:
        for key in ("processed_detection_index", "detection_index"):
            value = detection.get(key)
            if value is None:
                continue
            try:
                detection_id = int(value)
            except (TypeError, ValueError):
                continue
            if detection_id >= 0:
                return detection_id
        return int(index)

    def _keypoints_from_detection(self, detection: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        expected_count = int(max(4, self.object_points_m.shape[0]))
        yolo_keypoints = detection.get("yolo_keypoints")
        if yolo_keypoints is not None:
            keypoints = np.asarray(yolo_keypoints, dtype=float)
            if (
                keypoints.ndim == 2
                and keypoints.shape[0] >= expected_count
                and keypoints.shape[1] >= 2
            ):
                keypoints = keypoints[:expected_count]
                keypoints_px = np.asarray(keypoints[:, :2], dtype=float).reshape(
                    expected_count,
                    2,
                )
                if keypoints.shape[1] >= 3:
                    keypoint_conf = np.asarray(keypoints[:, 2], dtype=float).reshape(
                        expected_count
                    )
                else:
                    keypoint_conf = np.ones(expected_count, dtype=float)
                return keypoints_px, keypoint_conf

        for key in ("ordered_corners", "raw_corners"):
            corners = detection.get(key)
            if corners is None:
                continue
            arr = np.asarray(corners, dtype=float)
            if arr.ndim >= 2 and arr.shape[0] >= expected_count and arr.shape[1] >= 2:
                return (
                    arr[:expected_count, :2].reshape(expected_count, 2).copy(),
                    np.ones(expected_count, dtype=float),
                )

        return (
            np.full((expected_count, 2), np.nan, dtype=float),
            np.zeros(expected_count, dtype=float),
        )

    @staticmethod
    def _vec3(value, default=None):
        if value is None:
            return default
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size < 3:
            return default
        out = arr[:3].astype(float).copy()
        if not np.all(np.isfinite(out)):
            return default
        return out

    @staticmethod
    def _float_or_default(value, default: float, allow_nan: bool = False) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        if math.isfinite(out) or (allow_nan and math.isnan(out)):
            return out
        return float(default)

    @staticmethod
    def _float_or_none(value) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    @staticmethod
    def _int_or_none(value) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _wall_time(cls, source) -> Optional[float]:
        if not isinstance(source, dict):
            return None
        return cls._float_or_none(source.get("wall_time"))

    def _gazebo_camera_pose_ok(self, gazebo_pose) -> tuple[bool, str]:
        if not isinstance(gazebo_pose, dict):
            return False, "missing_pose"
        if self._vec3(gazebo_pose.get("gazebo_camera_pos_world"), default=None) is None:
            return False, "missing_camera_position"
        if self._quat4(gazebo_pose.get("gazebo_camera_quat_world")) is None:
            return False, "missing_camera_quaternion"
        age_s = self._gazebo_pose_age_s(gazebo_pose)
        max_age_s = float(self.gazebo_camera_sim_max_pose_age_s)
        if max_age_s > 0.0 and (not math.isfinite(age_s) or age_s > max_age_s):
            return False, f"stale_pose_age_s={age_s:.3f}"
        return True, ""

    def _maybe_warn_gazebo_camera_sim(self, message: str) -> None:
        now_s = time.monotonic()
        if now_s - self._last_gazebo_camera_sim_warning_s < 1.0:
            return
        self._last_gazebo_camera_sim_warning_s = now_s
        print(f"[perception_wrapper] {message}", flush=True)

    def _world_from_camera_gazebo(
        self,
        gate_camera: np.ndarray,
        gazebo_pose: dict[str, Any],
    ) -> Optional[np.ndarray]:
        camera_pos = self._vec3(gazebo_pose.get("gazebo_camera_pos_world"), default=None)
        camera_quat = self._quat4(gazebo_pose.get("gazebo_camera_quat_world"))
        if camera_pos is None or camera_quat is None:
            return None
        point_body = self._camera_optical_to_gazebo_body(gate_camera)
        if not np.all(np.isfinite(point_body)):
            return None
        r_wc = self._quat_to_rotmat(camera_quat)
        if self.gazebo_rotation_mode == "transpose":
            rel_world = r_wc @ point_body
        else:
            rel_world = r_wc.T @ point_body
        return self._gazebo_world_to_neu(camera_pos + rel_world)

    def _vector_from_camera_gazebo(
        self,
        vector_camera: np.ndarray,
        gazebo_pose: dict[str, Any],
    ) -> Optional[np.ndarray]:
        camera_quat = self._quat4(gazebo_pose.get("gazebo_camera_quat_world"))
        if camera_quat is None:
            return None
        vector_body = self._camera_optical_to_gazebo_body(vector_camera)
        if not np.all(np.isfinite(vector_body)):
            return None
        r_wc = self._quat_to_rotmat(camera_quat)
        if self.gazebo_rotation_mode == "transpose":
            vector_world = r_wc @ vector_body
        else:
            vector_world = r_wc.T @ vector_body
        return self._gazebo_world_to_neu(vector_world)

    def _camera_optical_to_gazebo_body(self, point_camera: np.ndarray) -> np.ndarray:
        camera = np.asarray(point_camera, dtype=float).reshape(3)
        if self.gazebo_optical_mode in ("current", "flip_y"):
            if self.gazebo_optical_mode == "flip_y":
                camera = camera.copy()
                camera[1] *= -1.0
            r_body_camera = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=float,
            )
            return r_body_camera @ camera
        if self.gazebo_optical_mode == "physical":
            return np.array([camera[2], camera[0], -camera[1]], dtype=float)
        if self.gazebo_optical_mode == "physical_minus_y":
            return np.array([camera[2], -camera[0], -camera[1]], dtype=float)
        return np.full(3, np.nan, dtype=float)

    @staticmethod
    def _gazebo_world_to_neu(point_gazebo: np.ndarray) -> np.ndarray:
        gazebo = np.asarray(point_gazebo, dtype=float).reshape(3)
        return np.array([gazebo[1], gazebo[0], gazebo[2]], dtype=float)

    @classmethod
    def _quat4(cls, value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            quat = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if quat.size < 4:
            return None
        quat = quat[:4].astype(float).copy()
        norm = float(np.linalg.norm(quat))
        if not math.isfinite(norm) or norm <= 1e-12:
            return None
        quat /= norm
        if not np.all(np.isfinite(quat)):
            return None
        return quat

    @staticmethod
    def _quat_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = np.asarray(quat_xyzw, dtype=float).reshape(4)
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=float,
        )

    @staticmethod
    def _gazebo_pose_age_s(gazebo_pose) -> float:
        if not isinstance(gazebo_pose, dict):
            return math.nan
        try:
            stamp = float(gazebo_pose.get("gazebo_pose_wall_time"))
        except (TypeError, ValueError):
            return math.nan
        if not math.isfinite(stamp):
            return math.nan
        return time.time() - stamp

    @staticmethod
    def _fmt_array(value, precision: int = 3) -> str:
        arr = np.asarray(value, dtype=float)
        return np.array2string(
            arr,
            precision=int(precision),
            suppress_small=True,
            separator=",",
            max_line_width=1000,
        )

    def _position_neu(self, source) -> Optional[np.ndarray]:
        if not isinstance(source, dict):
            return None

        pos_neu = self._vec3(source.get("pos_neu"), default=None)
        if pos_neu is not None:
            return pos_neu

        pos_ned = self._vec3(source.get("pos_ned"), default=None)
        if pos_ned is None and all(key in source for key in ("x", "y", "z")):
            pos_ned = self._vec3(
                (source.get("x"), source.get("y"), source.get("z")),
                default=None,
            )
        if pos_ned is None:
            return None

        return np.array([pos_ned[0], pos_ned[1], -pos_ned[2]], dtype=float)

    def _position_ned(self, source) -> Optional[np.ndarray]:
        if not isinstance(source, dict):
            return None

        pos_ned = self._vec3(source.get("pos_ned"), default=None)
        if pos_ned is not None:
            return pos_ned

        if all(key in source for key in ("x", "y", "z")):
            pos_ned = self._vec3(
                (source.get("x"), source.get("y"), source.get("z")),
                default=None,
            )
            if pos_ned is not None:
                return pos_ned

        pos_neu = self._vec3(source.get("pos_neu"), default=None)
        if pos_neu is not None:
            return local_neu_to_ned(pos_neu)

        return None

    @staticmethod
    def _attitude_rpy(attitude) -> Optional[np.ndarray]:
        if not isinstance(attitude, dict):
            return None
        try:
            rpy = np.array(
                [
                    float(attitude["roll"]),
                    float(attitude["pitch"]),
                    float(attitude["yaw"]),
                ],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if np.all(np.isfinite(rpy)):
            return rpy
        return None

    def _odometry_rpy(self, odometry) -> Optional[np.ndarray]:
        if not isinstance(odometry, dict):
            return None
        q_wxyz = odometry.get("q_wxyz")
        if q_wxyz is None and "q" in odometry:
            q_wxyz = odometry.get("q")
        return self._rpy_from_q_wxyz(q_wxyz)

    @staticmethod
    def _rpy_from_q_wxyz(q_wxyz) -> Optional[np.ndarray]:
        if q_wxyz is None:
            return None
        try:
            w, x, y, z = np.asarray(q_wxyz, dtype=float).reshape(4)
        except (TypeError, ValueError):
            return None

        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if not math.isfinite(norm) or norm <= 1e-12:
            return None
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        rpy = np.array([roll, pitch, yaw], dtype=float)
        if np.all(np.isfinite(rpy)):
            return rpy
        return None
