import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque

from autonomy_core.core.competition_config import VADR_TS_002, planar_square_object_points_m

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without YOLO.
    YOLO = None


def default_live_pnp_corner_reordering(corners_are_semantic: bool) -> bool:
    return not bool(corners_are_semantic)


class GatePerception:

    def __init__(self,
                 gate_size=VADR_TS_002.gate_inner_square_m,
                 smoothing_window=5,
                 max_failures=10,
                 yolo_model_path=None,
                 yolo_conf=0.1,
                 yolo_imgsz=640,
                 yolo_device=None,
                 preprocess_mode="distinctive"):

        self.gate_size = gate_size
        self.pose_history = deque(maxlen=smoothing_window)
        self.max_failures = max_failures
        self.failure_counter = 0
        self.last_debug = {}
        self.last_yolo_detection_count = 0
        self.last_yolo_detection_confidences = []
        self.last_yolo_detection_bboxes = []
        self.last_yolo_detection_keypoints = []
        self.last_yolo_candidate_debug = []
        self.live_solver_name = "IPPE_SQUARE_CANDIDATE_SWEEP"

        # YOLO pose settings
        if yolo_model_path is None:
            raise ValueError(
                "GatePerception now uses YOLO. Pass yolo_model_path='runs/pose/.../weights/best.pt'"
            )
        self.yolo_model_path = str(yolo_model_path)
        if YOLO is None:
            raise ModuleNotFoundError(
                "GatePerception requires the 'ultralytics' package for YOLO inference"
            )
        self.yolo_model = YOLO(self.yolo_model_path)
        self.yolo_conf = float(yolo_conf)
        self.yolo_imgsz = int(yolo_imgsz)
        self.yolo_device = yolo_device
        self.preprocess_mode = preprocess_mode
        self.corners_are_semantic = True  # YOLO pose outputs TL, TR, BR, BL directly.
        self.allow_pnp_corner_reordering = default_live_pnp_corner_reordering(
            self.corners_are_semantic
        )

        print(f"[YOLO PERCEPTION] model={self.yolo_model_path}")
        print(f"[YOLO PERCEPTION] preprocess_mode={self.preprocess_mode}, conf={self.yolo_conf}, imgsz={self.yolo_imgsz}")
        print("[LIVE PNP] using IPPE_SQUARE candidate sweep")
        print(f"[LIVE PNP] allow_pnp_corner_reordering={self.allow_pnp_corner_reordering}")

        # TII keypoint labels are inner opening corners, so use the inner opening size.
        # Competition/spec gate: inner opening = 1.5 m x 1.5 m.
        # If you switch back to HSV outer-frame contour detection, use gate_size=2.7 instead.
        self.model_points = np.array(
            planar_square_object_points_m(gate_size),
            dtype=np.float32,
        )

        print("Model points:")
        print(self.model_points)

    # -------------------------------------------------
    # Main API
    # -------------------------------------------------
    def process(self, frame, camera_matrix, dist_coeffs):

        # Show grayscale image
        # cv2.imshow("Gray", frame)
        # cv2.waitKey(1)
        corners = self.detect_gate(frame)

        if corners is None:
            return self.handle_failure()

        self.failure_counter = 0

        ordered = self.order_corners(corners)

        result = self._estimate_detection_result(corners, ordered, camera_matrix, dist_coeffs)
        if result is None:
            return None
        self.last_debug = result["debug"].copy()

        # -------- DEBUG DRAW --------

        # If the image is already BGR (3 channels), just copy it.
        # If it's grayscale (2 dims), convert it to BGR so we can draw green lines on it.
        if len(frame.shape) == 3:
            debug = frame.copy()
        else:
            debug = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for pt in ordered:
            cv2.circle(
                debug,
                tuple(pt.astype(int)),
                6,
                (0,255,0),
                -1
            )

        cv2.polylines(
            debug,
            [ordered.astype(int)],
            True,
            (0,255,0),
            2
        )

        # cv2.imshow("Detection", debug)
        # cv2.waitKey(1)

        # Print pose

        print("\n------ Pose ------")

        print("t (meters):")
        print(result["t"].flatten())

        print("confidence:", result["confidence"])
        print("reprojection_error:", result["debug"]["reprojection_error"])

        return result

    def process_all(self, frame, camera_matrix, dist_coeffs):
        corners_list = self.detect_gate_candidates(frame)
        if len(corners_list) == 0:
            self.handle_failure()
            return []

        self.failure_counter = 0
        detections = []
        candidate_debug = getattr(self, "last_yolo_candidate_debug", [])
        for processed_index, corners in enumerate(corners_list):
            ordered = self.order_corners(corners)
            result = self._estimate_detection_result(corners, ordered, camera_matrix, dist_coeffs)
            if result is not None:
                meta = candidate_debug[processed_index] if processed_index < len(candidate_debug) else {}
                yolo_confidence = float(
                    meta.get("box_confidence", result.get("confidence", np.nan))
                )
                old_area_confidence = float(result.get("confidence", np.nan))
                quad_area_px2 = float(abs(cv2.contourArea(np.asarray(ordered, dtype=np.float32))))
                result["yolo_confidence"] = yolo_confidence
                result["quad_area_px2"] = quad_area_px2
                result["quad_area_confidence"] = old_area_confidence
                result["old_area_confidence"] = old_area_confidence
                # Gate existence comes from YOLO; apparent image size is diagnostic only.
                result["memory_confidence"] = yolo_confidence
                result["debug"]["detection_index"] = int(meta.get("detection_index", processed_index))
                result["debug"]["processed_detection_index"] = int(processed_index)
                result["debug"]["yolo_bbox"] = np.asarray(
                    meta.get("bbox", np.full(4, np.nan)),
                    dtype=float,
                ).reshape(4).copy()
                result["debug"]["yolo_box_confidence"] = float(
                    meta.get("box_confidence", result.get("confidence", np.nan))
                )
                result["debug"]["yolo_keypoints"] = np.asarray(
                    meta.get("keypoints", np.full((4, 3), np.nan)),
                    dtype=float,
                ).reshape(-1, 3).copy()
                result["debug"]["quad_area_px2"] = quad_area_px2
                result["debug"]["quad_area_confidence"] = old_area_confidence
                result["debug"]["old_area_confidence"] = old_area_confidence
                result["debug"]["memory_confidence"] = yolo_confidence
                detections.append(result)
            else:
                if processed_index < len(candidate_debug):
                    candidate_debug[processed_index]["rejection_reason"] = "no_pnp_solution"

        detections.sort(
            key=lambda det: (
                -float(det.get("confidence", 0.0)),
                float(det.get("debug", {}).get("reprojection_error", np.inf)),
            )
        )
        if detections:
            self.last_debug = detections[0]["debug"].copy()
        return detections

    def _estimate_detection_result(self, corners, ordered, camera_matrix, dist_coeffs):
        pose = self.estimate_pose(
            ordered,
            camera_matrix,
            dist_coeffs
        )

        if pose is None:
            return None

        R_mat, tvec, pnp_debug = pose
        selected_ordered = np.asarray(
            pnp_debug.get("ordered_points", ordered),
            dtype=np.float32,
        ).reshape(4, 2)
        confidence = self.compute_confidence(ordered)

        # Do not smooth camera-frame pose while the camera is moving. Averaging
        # tvec across frames from different drone poses creates a biased world
        # landmark after the current drone pose is applied.
        R_out, t_out = R_mat, tvec
        reprojection_error = self.compute_reprojection_error(
            R_out,
            t_out,
            camera_matrix,
            dist_coeffs,
            selected_ordered,
        )
        reprojected_corners = self.project_model_points(
            R_out,
            t_out,
            camera_matrix,
            dist_coeffs,
        )
        gate_normal_camera = R_out[:, 2].astype(float)
        gate_normal_camera /= np.linalg.norm(gate_normal_camera) + 1e-12
        debug = {
            "raw_corners": np.asarray(corners, dtype=float).reshape(-1, 2).copy(),
            "ordered_corners": selected_ordered.copy(),
            "pnp_debug_best_ordered_corners": np.asarray(
                pnp_debug.get("debug_best_ordered_points", selected_ordered),
                dtype=float,
            ).reshape(4, 2).copy(),
            "reprojected_corners": reprojected_corners.copy(),
            "rvec": pnp_debug["rvec"].copy(),
            "tvec": np.asarray(t_out, dtype=float).reshape(3).copy(),
            "reprojection_error": float(reprojection_error),
            "corner_reprojection_error_px": float(reprojection_error),
            "gate_normal_camera": gate_normal_camera.copy(),
            "pnp_candidates": pnp_debug["candidates"],
            "chosen_candidate": int(pnp_debug["chosen_candidate"]),
            "live_solver_name": pnp_debug.get("live_solver_name", ""),
            "pnp_fallback_reason": pnp_debug.get("fallback_reason", ""),
            "pnp_selected_order": pnp_debug.get("selected_order", ""),
            "pnp_selected_solver": pnp_debug.get("selected_solver", ""),
            "pnp_selected_score": float(pnp_debug.get("selected_score", np.nan)),
            "pnp_selected_reprojection_error": float(pnp_debug.get("selected_reprojection_error", np.nan)),
            "pnp_selected_gate_center_camera": np.asarray(t_out, dtype=float).reshape(3).copy(),
            "pnp_selected_reason": pnp_debug.get("selected_reason", ""),
            "pnp_candidate_summary": pnp_debug.get("candidate_summary", ""),
            "allow_pnp_corner_reordering": bool(pnp_debug.get("allow_pnp_corner_reordering", False)),
            "pnp_live_candidate_orders_allowed": pnp_debug.get("live_candidate_orders_allowed", ""),
            "pnp_debug_best_order": pnp_debug.get("debug_best_order", ""),
            "pnp_live_vs_debug_best_order_mismatch": bool(
                pnp_debug.get("live_vs_debug_best_order_mismatch", False)
            ),
            "pnp_lateral_angle": float(pnp_debug.get("pnp_lateral_angle", np.nan)),
            "image_center_offset_normalized": float(
                pnp_debug.get("image_center_offset_normalized", np.nan)
            ),
            "keypoint_polygon_signed_area": float(
                pnp_debug.get("keypoint_polygon_signed_area", np.nan)
            ),
            "keypoint_polygon_winding": pnp_debug.get("keypoint_polygon_winding", ""),
            "keypoint_edge_top": float(pnp_debug.get("keypoint_edge_top", np.nan)),
            "keypoint_edge_right": float(pnp_debug.get("keypoint_edge_right", np.nan)),
            "keypoint_edge_bottom": float(pnp_debug.get("keypoint_edge_bottom", np.nan)),
            "keypoint_edge_left": float(pnp_debug.get("keypoint_edge_left", np.nan)),
            "keypoint_bbox_center": np.asarray(
                pnp_debug.get("keypoint_bbox_center", np.full(2, np.nan)),
                dtype=float,
            ).reshape(2).copy(),
            "keypoint_polygon_center": np.asarray(
                pnp_debug.get("keypoint_polygon_center", np.full(2, np.nan)),
                dtype=float,
            ).reshape(2).copy(),
            "keypoint_bbox_polygon_delta": np.asarray(
                pnp_debug.get("keypoint_bbox_polygon_delta", np.full(2, np.nan)),
                dtype=float,
            ).reshape(2).copy(),
            "raw_keypoint_polygon_signed_area": float(
                pnp_debug.get("raw_keypoint_polygon_signed_area", np.nan)
            ),
            "raw_keypoint_polygon_winding": pnp_debug.get("raw_keypoint_polygon_winding", ""),
            "gate_size_sweep": self.solve_pnp_gate_size_sweep(
                selected_ordered,
                camera_matrix,
                dist_coeffs,
                sizes=(1.40, 1.50, 1.60),
            ),
            "pnp_formulation_debug": self.solve_pnp_formulation_debug(
                selected_ordered,
                camera_matrix,
                dist_coeffs,
            ),
        }
        return {
            "R": R_out,
            "t": t_out,
            "confidence": confidence,
            "debug": debug,
        }


    # -------------------------------------------------
    # YOLO preprocessing helpers
    # -------------------------------------------------
    @staticmethod
    def normalize_uint8(x):
        x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
        return x.astype(np.uint8)

    @staticmethod
    def make_clahe_gray(img_bgr, clip_limit=2.0):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def make_color_saliency(self, img_bgr, blur_ksize=41):
        if blur_ksize % 2 == 0:
            blur_ksize += 1

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        _, A, B = cv2.split(lab)

        A_blur = cv2.GaussianBlur(A, (blur_ksize, blur_ksize), 0)
        B_blur = cv2.GaussianBlur(B, (blur_ksize, blur_ksize), 0)

        diff_a = cv2.absdiff(A, A_blur)
        diff_b = cv2.absdiff(B, B_blur)

        color_diff = cv2.addWeighted(diff_a, 0.5, diff_b, 0.5, 0)
        return self.normalize_uint8(color_diff)

    def make_local_contrast(self, img_bgr, blur_ksize=41):
        if blur_ksize % 2 == 0:
            blur_ksize += 1

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        contrast = cv2.absdiff(gray, blur)
        return self.normalize_uint8(contrast)

    def make_pseudo_image(self, img_bgr, mode):
        """
        Must match the preprocessing used during training.

        gray_clahe:
            ch1/ch2/ch3 = CLAHE grayscale
        no_edge:
            ch1 = CLAHE grayscale, ch2 = original grayscale, ch3 = LAB color saliency
        distinctive:
            ch1 = CLAHE grayscale, ch2 = local contrast, ch3 = LAB color saliency
        none/raw:
            original BGR frame
        """
        if mode in [None, "none", "raw"]:
            return img_bgr

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = self.make_clahe_gray(img_bgr, clip_limit=2.0)

        if mode == "gray_clahe":
            return cv2.merge([gray_eq, gray_eq, gray_eq])

        if mode == "no_edge":
            color_saliency = self.make_color_saliency(img_bgr, blur_ksize=31)
            return cv2.merge([gray_eq, gray, color_saliency])

        if mode == "distinctive":
            local_contrast = self.make_local_contrast(img_bgr, blur_ksize=41)
            color_saliency = self.make_color_saliency(img_bgr, blur_ksize=41)
            return cv2.merge([gray_eq, local_contrast, color_saliency])

        raise ValueError(f"Unknown preprocess_mode: {mode}")

    # -------------------------------------------------
    # Look for holes algorithm
    # -------------------------------------------------
    def detect_gate(self, frame):
        candidates = self.detect_gate_candidates(frame)
        if len(candidates) == 0:
            return None
        return candidates[0]

    def detect_gate_candidates(self, frame):
        """
        Returns a list of 4-point arrays in TL, TR, BR, BL order.
        These are YOLO-predicted inner opening corners, not HSV outer-frame corners.
        """
        if frame is None:
            return []

        if len(frame.shape) == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame

        yolo_input = self.make_pseudo_image(frame_bgr, self.preprocess_mode)

        # Ultralytics accepts numpy images. Keep verbose=False so the flight loop does not spam.
        kwargs = dict(
            source=yolo_input,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_conf,
            verbose=False,
        )
        if self.yolo_device is not None:
            kwargs["device"] = self.yolo_device

        try:
            result = self.yolo_model.predict(**kwargs)[0]
        except Exception as exc:
            print(f"[YOLO PERCEPTION] prediction failed: {exc}")
            self.last_yolo_detection_count = 0
            self.last_yolo_detection_confidences = []
            self.last_yolo_detection_bboxes = []
            self.last_yolo_detection_keypoints = []
            self.last_yolo_candidate_debug = []
            return []

        if result.boxes is None or len(result.boxes) == 0:
            self.last_yolo_detection_count = 0
            self.last_yolo_detection_confidences = []
            self.last_yolo_detection_bboxes = []
            self.last_yolo_detection_keypoints = []
            self.last_yolo_candidate_debug = []
            return []
        if result.keypoints is None or result.keypoints.data is None:
            self.last_yolo_detection_count = 0
            self.last_yolo_detection_confidences = []
            self.last_yolo_detection_bboxes = []
            self.last_yolo_detection_keypoints = []
            self.last_yolo_candidate_debug = []
            return []

        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        boxes_conf = result.boxes.conf.detach().cpu().numpy()
        keypoints = result.keypoints.data.detach().cpu().numpy()  # shape: N x K x 3
        keep_mask = boxes_conf >= self.yolo_conf
        self.last_yolo_detection_count = int(np.sum(keep_mask))
        self.last_yolo_detection_confidences = [
            float(c) for c in boxes_conf[keep_mask]
        ]
        self.last_yolo_detection_bboxes = [
            np.asarray(b, dtype=float).reshape(4).tolist()
            for b in boxes_xyxy[keep_mask]
        ]
        self.last_yolo_detection_keypoints = [
            np.asarray(k, dtype=float).tolist()
            for k in keypoints[keep_mask]
        ]

        candidates = []
        h, w = frame_bgr.shape[:2]

        for i in range(len(boxes_conf)):
            if float(boxes_conf[i]) < self.yolo_conf:
                continue
            meta = {
                "detection_index": int(i),
                "box_confidence": float(boxes_conf[i]),
                "bbox": np.asarray(boxes_xyxy[i], dtype=float).reshape(4).copy(),
                "keypoints": np.asarray(keypoints[i], dtype=float).copy(),
                "rejection_reason": "",
            }
            if keypoints.shape[1] < 4:
                meta["rejection_reason"] = "fewer_than_4_keypoints"
                candidates.append((-np.inf, float(boxes_conf[i]), 0.0, None, meta))
                continue

            pts = keypoints[i, :4, :2].astype(np.float32)
            kconf = keypoints[i, :4, 2].astype(float) if keypoints.shape[2] >= 3 else np.ones(4)

            # Reject broken keypoint sets.
            if not np.isfinite(pts).all():
                meta["rejection_reason"] = "non_finite_keypoints"
                candidates.append((-np.inf, float(boxes_conf[i]), 0.0, None, meta))
                continue
            if np.any(pts[:, 0] < -5) or np.any(pts[:, 0] > w + 5):
                meta["rejection_reason"] = "keypoints_outside_image_x"
                candidates.append((-np.inf, float(boxes_conf[i]), 0.0, None, meta))
                continue
            if np.any(pts[:, 1] < -5) or np.any(pts[:, 1] > h + 5):
                meta["rejection_reason"] = "keypoints_outside_image_y"
                candidates.append((-np.inf, float(boxes_conf[i]), 0.0, None, meta))
                continue

            # Very low keypoint confidence means the box may be okay but corners are not.
            mean_kconf = float(np.mean(kconf))
            if mean_kconf < 0.10:
                meta["rejection_reason"] = "keypoint_confidence_low"
                candidates.append((-np.inf, float(boxes_conf[i]), mean_kconf, None, meta))
                continue

            area = abs(cv2.contourArea(pts))
            if area < 50:
                meta["rejection_reason"] = "keypoint_area_low"
                candidates.append((-np.inf, float(boxes_conf[i]), mean_kconf, None, meta))
                continue

            # Score by bbox confidence, keypoint confidence, and keypoint quad area.
            score = float(boxes_conf[i]) + 0.25 * mean_kconf + min(area / 40000.0, 1.0) * 0.10

            candidates.append((score, float(boxes_conf[i]), mean_kconf, pts, meta))

        candidates.sort(key=lambda item: item[0], reverse=True)

        # De-duplicate similar detections.
        deduped = []
        rejected_meta = []
        for score, box_conf, mean_kconf, pts, meta in candidates:
            if pts is None:
                rejected_meta.append(meta)
                continue
            center = np.mean(pts, axis=0)
            size = np.ptp(pts, axis=0)
            duplicate = False
            for kept_score, kept_box_conf, kept_mean_kconf, kept_pts, kept_meta in deduped:
                kept_center = np.mean(kept_pts, axis=0)
                kept_size = np.ptp(kept_pts, axis=0)
                center_close = np.linalg.norm(center - kept_center) < 12.0
                size_close = np.linalg.norm(size - kept_size) < 20.0
                if center_close and size_close:
                    duplicate = True
                    break
            if not duplicate:
                deduped.append((score, box_conf, mean_kconf, pts, meta))
            else:
                meta["rejection_reason"] = "duplicate_yolo_detection"
                rejected_meta.append(meta)

        self.last_yolo_candidate_debug = [
            meta for _, _, _, _, meta in deduped
        ] + rejected_meta
        return [pts for _, _, _, pts, _ in deduped]

    # -------------------------------------------------
    # Correct Corner Ordering
    # -------------------------------------------------
    def order_corners(self, corners):
        pts = np.asarray(corners, dtype=np.float32)
        if pts.shape[0] < 4:
            return None

        # YOLO pose labels already provide semantic order: TL, TR, BR, BL.
        # Do not reorder using sum/diff, because that can scramble keypoint identity.
        if getattr(self, "corners_are_semantic", False) and pts.shape[0] == 4:
            return pts.astype(np.float32)

        # If more than 4 points, take convex hull then approximate to 4
        if pts.shape[0] > 4:
            hull = cv2.convexHull(pts)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            pts = approx.reshape(-1, 2).astype(np.float32)

        if pts.shape[0] != 4:
            return None

        # Classic TL/TR/BR/BL ordering using sum and diff
        s = pts.sum(axis=1)  # x+y
        d = (pts[:, 0] - pts[:, 1])  # x-y

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmax(d)]
        bl = pts[np.argmin(d)]

        ordered = np.array([tl, tr, br, bl], dtype=np.float32)
        return ordered


    # -------------------------------------------------
    # Pose Estimation
    # -------------------------------------------------
    @staticmethod
    def pnp_order_permutations():
        return [
            ("tl_tr_br_bl", [0, 1, 2, 3]),
            ("tr_br_bl_tl", [1, 2, 3, 0]),
            ("br_bl_tl_tr", [2, 3, 0, 1]),
            ("bl_tl_tr_br", [3, 0, 1, 2]),
            ("tl_bl_br_tr", [0, 3, 2, 1]),
            ("tr_tl_bl_br", [1, 0, 3, 2]),
        ]

    @staticmethod
    def _pnp_err_scalar(err):
        if err is None:
            return float("nan")
        return float(np.asarray(err).reshape(-1)[0])

    def estimate_size_depth(self, image_points, camera_matrix):
        pts = np.asarray(image_points, dtype=float).reshape(4, 2)
        k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        fx = float(k[0, 0])
        fy = float(k[1, 1])
        width_px = 0.5 * (
            np.linalg.norm(pts[1] - pts[0]) +
            np.linalg.norm(pts[2] - pts[3])
        )
        height_px = 0.5 * (
            np.linalg.norm(pts[3] - pts[0]) +
            np.linalg.norm(pts[2] - pts[1])
        )
        depths = []
        if width_px > 1.0 and np.isfinite(fx):
            depths.append(fx * float(self.gate_size) / width_px)
        if height_px > 1.0 and np.isfinite(fy):
            depths.append(fy * float(self.gate_size) / height_px)
        if not depths:
            return float("nan")
        return float(np.mean(depths))

    def image_center_offset_normalized(self, image_points, camera_matrix):
        pts = np.asarray(image_points, dtype=float).reshape(4, 2)
        k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        fx = float(k[0, 0])
        cx = float(k[0, 2])
        if not np.isfinite(fx) or abs(fx) < 1e-9:
            return float("nan")
        return float((np.mean(pts[:, 0]) - cx) / fx)

    @staticmethod
    def keypoint_geometry(image_points):
        pts = np.asarray(image_points, dtype=float).reshape(4, 2)
        signed_area = 0.5 * float(
            np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) -
            np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
        )
        edge_lengths = [
            float(np.linalg.norm(pts[1] - pts[0])),
            float(np.linalg.norm(pts[2] - pts[1])),
            float(np.linalg.norm(pts[3] - pts[2])),
            float(np.linalg.norm(pts[0] - pts[3])),
        ]
        bbox_center = 0.5 * (np.min(pts, axis=0) + np.max(pts, axis=0))
        polygon_center = np.mean(pts, axis=0)
        return {
            "signed_area": signed_area,
            "winding": "ccw" if signed_area > 0.0 else "cw" if signed_area < 0.0 else "degenerate",
            "edge_top": edge_lengths[0],
            "edge_right": edge_lengths[1],
            "edge_bottom": edge_lengths[2],
            "edge_left": edge_lengths[3],
            "bbox_center": bbox_center,
            "polygon_center": polygon_center,
            "bbox_polygon_delta": bbox_center - polygon_center,
        }

    def score_pnp_candidate(self, error, tvec, normal, size_depth, image_center_offset):
        tvec = np.asarray(tvec, dtype=float).reshape(3)
        normal = np.asarray(normal, dtype=float).reshape(3)
        depth = float(tvec[2])
        lateral_angle = float(tvec[0] / depth) if abs(depth) > 1e-9 else float("nan")
        normal_score = abs(float(normal[2]))
        depth_disagreement = (
            abs(depth - float(size_depth))
            if np.isfinite(depth) and np.isfinite(size_depth)
            else float("nan")
        )
        lateral_offset_disagreement = (
            abs(lateral_angle - float(image_center_offset))
            if np.isfinite(lateral_angle) and np.isfinite(image_center_offset)
            else float("nan")
        )
        error_value = float(error) if np.isfinite(error) else 1e6
        score = -error_value
        reason = "lowest_reprojection_geometry_pass"

        if not np.isfinite(error):
            score -= 1e6
            reason = "non_finite_reprojection"
        if depth <= 0.0:
            score -= 1e6
            reason = "non_positive_depth"
        elif depth < 0.5 or depth > 30.0:
            score -= 50.0
            reason = "depth_outside_soft_range"
        if normal_score < 0.15:
            score -= 25.0
            reason = "normal_grazing_camera"
        if np.isfinite(depth_disagreement) and depth_disagreement > 1.0:
            score -= 10.0 * depth_disagreement
            reason = "size_depth_disagreement"
        if np.isfinite(lateral_offset_disagreement) and lateral_offset_disagreement > 0.25:
            score -= 25.0 * lateral_offset_disagreement
            reason = "lateral_angle_inconsistent_with_image_center"

        # Mild preference for less grazing planar solutions after reprojection.
        score += 0.25 * normal_score
        return (
            float(score),
            reason,
            float(depth),
            float(normal_score),
            float(size_depth) if np.isfinite(size_depth) else float("nan"),
            float(depth_disagreement) if np.isfinite(depth_disagreement) else float("nan"),
            float(lateral_angle) if np.isfinite(lateral_angle) else float("nan"),
            float(image_center_offset) if np.isfinite(image_center_offset) else float("nan"),
            float(lateral_offset_disagreement) if np.isfinite(lateral_offset_disagreement) else float("nan"),
        )

    def estimate_pose(self, image_points, camera_matrix, dist_coeffs):
        ordered_points = np.asarray(image_points, dtype=np.float32).reshape(4, 2)

        print("\n------ Perception ------")
        candidates = []

        for order_name, idx in self.pnp_order_permutations():
            pts = ordered_points[idx].astype(np.float32)
            image_points_ordered = pts.reshape(-1, 1, 2)
            size_depth = self.estimate_size_depth(pts, camera_matrix)
            image_center_offset = self.image_center_offset_normalized(pts, camera_matrix)
            try:
                ok, rvecs, tvecs, reprojErrs = cv2.solvePnPGeneric(
                    self.model_points,
                    image_points_ordered,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            except Exception:
                continue

            if not ok or rvecs is None or len(rvecs) == 0:
                continue

            for candidate_index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                R_mat, _ = cv2.Rodrigues(rvec)
                normal = R_mat[:, 2].astype(float)
                normal /= np.linalg.norm(normal) + 1e-12
                err = (
                    self._pnp_err_scalar(reprojErrs[candidate_index])
                    if reprojErrs is not None and len(reprojErrs) > candidate_index
                    else self.compute_reprojection_error(
                        R_mat,
                        tvec,
                        camera_matrix,
                        dist_coeffs,
                        image_points_ordered,
                    )
                )
                (
                    score,
                    reason,
                    depth,
                    normal_score,
                    size_depth,
                    depth_disagreement,
                    lateral_angle,
                    image_center_offset,
                    lateral_offset_disagreement,
                ) = self.score_pnp_candidate(err, tvec, normal, size_depth, image_center_offset)
                candidates.append({
                    "index": len(candidates),
                    "solver": "IPPE_SQUARE",
                    "order": order_name,
                    "order_indices": list(idx),
                    "solver_candidate_index": int(candidate_index),
                    "rvec": np.asarray(rvec, dtype=float).reshape(3),
                    "tvec": np.asarray(tvec, dtype=float).reshape(3),
                    "normal": normal.copy(),
                    "error": float(err),
                    "score": float(score),
                    "selected_reason": reason,
                    "depth": float(depth),
                    "normal_score": float(normal_score),
                    "size_depth": float(size_depth),
                    "depth_disagreement": float(depth_disagreement),
                    "lateral_angle": float(lateral_angle),
                    "image_center_offset_normalized": float(image_center_offset),
                    "lateral_offset_disagreement": float(lateral_offset_disagreement),
                    "ordered_points": pts.copy(),
                    "projected_corners": self.project_model_points(
                        R_mat,
                        tvec,
                        camera_matrix,
                        dist_coeffs,
                    ),
                })

        image_points_default = ordered_points.reshape(-1, 1, 2)
        size_depth_default = self.estimate_size_depth(ordered_points, camera_matrix)
        image_center_offset_default = self.image_center_offset_normalized(ordered_points, camera_matrix)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points_default,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            ok = False
        if ok:
            R_mat, _ = cv2.Rodrigues(rvec)
            normal = R_mat[:, 2].astype(float)
            normal /= np.linalg.norm(normal) + 1e-12
            err = self.compute_reprojection_error(
                R_mat,
                tvec,
                camera_matrix,
                dist_coeffs,
                image_points_default,
            )
            (
                score,
                reason,
                depth,
                normal_score,
                size_depth,
                depth_disagreement,
                lateral_angle,
                image_center_offset,
                lateral_offset_disagreement,
            ) = self.score_pnp_candidate(err, tvec, normal, size_depth_default, image_center_offset_default)
            candidates.append({
                "index": len(candidates),
                "solver": "ITERATIVE",
                "order": "tl_tr_br_bl",
                "order_indices": [0, 1, 2, 3],
                "solver_candidate_index": 0,
                "rvec": np.asarray(rvec, dtype=float).reshape(3),
                "tvec": np.asarray(tvec, dtype=float).reshape(3),
                "normal": normal.copy(),
                "error": float(err),
                "score": float(score),
                "selected_reason": reason,
                "depth": float(depth),
                "normal_score": float(normal_score),
                "size_depth": float(size_depth),
                "depth_disagreement": float(depth_disagreement),
                "lateral_angle": float(lateral_angle),
                "image_center_offset_normalized": float(image_center_offset),
                "lateral_offset_disagreement": float(lateral_offset_disagreement),
                "ordered_points": ordered_points.copy(),
                "projected_corners": self.project_model_points(
                    R_mat,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                ),
            })

        if len(candidates) == 0:
            print("[LIVE PNP] candidate sweep returned no solutions")
            return None

        live_allowed_orders = (
            {order for order, _ in self.pnp_order_permutations()}
            if self.allow_pnp_corner_reordering
            else {"tl_tr_br_bl"}
        )
        live_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("order", "") in live_allowed_orders
        ]
        if len(live_candidates) == 0:
            print("[LIVE PNP] semantic-order candidate set returned no solutions")
            return None

        debug_best = max(candidates, key=lambda item: item["score"])
        best = max(live_candidates, key=lambda item: item["score"])
        live_vs_debug_best_order_mismatch = best["order"] != debug_best["order"]
        selected_geometry = self.keypoint_geometry(best["ordered_points"])
        raw_geometry = self.keypoint_geometry(ordered_points)
        R_mat, _ = cv2.Rodrigues(best["rvec"])
        tvec = best["tvec"].reshape(3, 1)
        candidate_summary = ";".join(
            f"{c['solver']}/{c['order']}/{c['solver_candidate_index']}:"
            f"err={c['error']:.2f},z={c['depth']:.2f},size_z={c['size_depth']:.2f},"
            f"dz={c['depth_disagreement']:.2f},lat={c['lateral_angle']:.2f},"
            f"img={c['image_center_offset_normalized']:.2f},"
            f"lat_d={c['lateral_offset_disagreement']:.2f},"
            f"n={c['normal_score']:.2f},score={c['score']:.2f}"
            for c in candidates[:16]
        )
        print(
            "[LIVE PNP] selected "
            f"solver={best['solver']} order={best['order']} "
            f"candidate={best['solver_candidate_index']} error={best['error']:.3f} "
            f"depth={best['depth']:.3f} score={best['score']:.3f}"
        )
        if live_vs_debug_best_order_mismatch:
            print(
                "[LIVE PNP] debug best order differs from live semantic order: "
                f"debug={debug_best['order']} live={best['order']}"
            )
        return R_mat, tvec, {
            "rvec": best["rvec"].copy(),
            "tvec": best["tvec"].copy(),
            "chosen_candidate": int(best["index"]),
            "candidates": candidates,
            "live_solver_name": f"{best['solver']}_{best['order']}",
            "fallback_reason": "",
            "selected_order": best["order"],
            "selected_solver": best["solver"],
            "selected_score": float(best["score"]),
            "selected_reprojection_error": float(best["error"]),
            "selected_reason": best["selected_reason"],
            "ordered_points": best["ordered_points"].copy(),
            "debug_best_ordered_points": debug_best["ordered_points"].copy(),
            "candidate_summary": candidate_summary,
            "allow_pnp_corner_reordering": bool(self.allow_pnp_corner_reordering),
            "live_candidate_orders_allowed": ",".join(sorted(live_allowed_orders)),
            "debug_best_order": debug_best["order"],
            "live_vs_debug_best_order_mismatch": bool(live_vs_debug_best_order_mismatch),
            "pnp_lateral_angle": float(best["lateral_angle"]),
            "image_center_offset_normalized": float(best["image_center_offset_normalized"]),
            "keypoint_polygon_signed_area": float(selected_geometry["signed_area"]),
            "keypoint_polygon_winding": selected_geometry["winding"],
            "keypoint_edge_top": float(selected_geometry["edge_top"]),
            "keypoint_edge_right": float(selected_geometry["edge_right"]),
            "keypoint_edge_bottom": float(selected_geometry["edge_bottom"]),
            "keypoint_edge_left": float(selected_geometry["edge_left"]),
            "keypoint_bbox_center": selected_geometry["bbox_center"].copy(),
            "keypoint_polygon_center": selected_geometry["polygon_center"].copy(),
            "keypoint_bbox_polygon_delta": selected_geometry["bbox_polygon_delta"].copy(),
            "raw_keypoint_polygon_signed_area": float(raw_geometry["signed_area"]),
            "raw_keypoint_polygon_winding": raw_geometry["winding"],
        }

    @staticmethod
    def model_points_for_size(gate_size: float):
        return np.array(planar_square_object_points_m(gate_size), dtype=np.float32)

    def solve_pnp_gate_size_sweep(self, image_points, camera_matrix, dist_coeffs, sizes=(1.90, 2.00, 2.10)):
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
        out = {}

        def err_scalar(e):
            if e is None:
                return float("nan")
            return float(np.asarray(e).reshape(-1)[0])

        for size in sizes:
            key = f"{int(round(float(size) * 100)):03d}"
            model_points = self.model_points_for_size(size)
            ok, rvecs, tvecs, reprojErrs = cv2.solvePnPGeneric(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok or rvecs is None or len(rvecs) == 0:
                out[key] = {
                    "rvec": np.full(3, np.nan, dtype=float),
                    "tvec": np.full(3, np.nan, dtype=float),
                    "reprojection_error": float("nan"),
                    "chosen_candidate": None,
                    "candidate_count": 0,
                }
                continue

            best_i = 0
            best_score = -1e18
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                R_mat, _ = cv2.Rodrigues(rvec)
                n = R_mat[:, 2].astype(float)
                n /= np.linalg.norm(n) + 1e-12
                e = err_scalar(reprojErrs[i]) if reprojErrs is not None and len(reprojErrs) > i else 0.0
                tz = float(np.asarray(tvec).reshape(-1)[2])
                score = (10.0 * n[2]) - e
                if tz <= 0:
                    score -= 1e6
                if score > best_score:
                    best_score = score
                    best_i = i

            out[key] = {
                "rvec": np.asarray(rvecs[best_i], dtype=float).reshape(3),
                "tvec": np.asarray(tvecs[best_i], dtype=float).reshape(3),
                "reprojection_error": err_scalar(reprojErrs[best_i]) if reprojErrs is not None and len(reprojErrs) > best_i else float("nan"),
                "chosen_candidate": int(best_i),
                "candidate_count": int(len(rvecs)),
            }

        return out

    def solve_pnp_formulation_debug(self, ordered_points, camera_matrix, dist_coeffs):
        ordered_points = np.asarray(ordered_points, dtype=np.float32).reshape(4, 2)
        out = []

        solvers = [
            ("IPPE_SQUARE", cv2.SOLVEPNP_IPPE_SQUARE),
            ("IPPE", cv2.SOLVEPNP_IPPE),
            ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
        ]
        if hasattr(cv2, "SOLVEPNP_SQPNP"):
            solvers.append(("SQPNP", cv2.SOLVEPNP_SQPNP))

        for solver_name, flag in solvers:
            result = self.solve_pnp_debug_variant(
                ordered_points,
                camera_matrix,
                dist_coeffs,
                flag=flag,
                solver_name=solver_name,
                order_name="tl_tr_br_bl",
            )
            if result is not None:
                out.append(result)

        permutations = [
            ("tl_tr_br_bl", [0, 1, 2, 3]),
            ("tr_br_bl_tl", [1, 2, 3, 0]),
            ("br_bl_tl_tr", [2, 3, 0, 1]),
            ("bl_tl_tr_br", [3, 0, 1, 2]),
            ("tl_bl_br_tr", [0, 3, 2, 1]),
            ("tr_tl_bl_br", [1, 0, 3, 2]),
        ]
        for order_name, idx in permutations:
            pts = ordered_points[idx]
            result = self.solve_pnp_debug_variant(
                pts,
                camera_matrix,
                dist_coeffs,
                flag=cv2.SOLVEPNP_IPPE_SQUARE,
                solver_name="IPPE_SQUARE",
                order_name=order_name,
            )
            if result is not None:
                out.append(result)

        return out

    def solve_pnp_debug_variant(self, image_points, camera_matrix, dist_coeffs, flag, solver_name, order_name):
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
        try:
            ok, rvecs, tvecs, reprojErrs = cv2.solvePnPGeneric(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=flag,
            )
        except Exception:
            return None

        if not ok or rvecs is None or len(rvecs) == 0:
            return None

        def err_scalar(e):
            if e is None:
                return float("nan")
            return float(np.asarray(e).reshape(-1)[0])

        candidates = []
        best_i = 0
        best_score = -1e18
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R_mat, _ = cv2.Rodrigues(rvec)
            normal = R_mat[:, 2].astype(float)
            normal /= np.linalg.norm(normal) + 1e-12
            err = err_scalar(reprojErrs[i]) if reprojErrs is not None and len(reprojErrs) > i else self.compute_reprojection_error(
                R_mat,
                tvec,
                camera_matrix,
                dist_coeffs,
                image_points,
            )
            tz = float(np.asarray(tvec).reshape(-1)[2])
            score = (10.0 * normal[2]) - err
            if tz <= 0:
                score -= 1e6
            if score > best_score:
                best_score = score
                best_i = i
            candidates.append({
                "index": int(i),
                "rvec": np.asarray(rvec, dtype=float).reshape(3),
                "tvec": np.asarray(tvec, dtype=float).reshape(3),
                "normal": normal.copy(),
                "error": float(err),
                "projected_corners": self.project_model_points(
                    R_mat,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                ),
            })

        chosen = candidates[best_i]
        return {
            "solver": solver_name,
            "order": order_name,
            "chosen_candidate": int(best_i),
            "candidate_count": int(len(candidates)),
            "rvec": chosen["rvec"].copy(),
            "tvec": chosen["tvec"].copy(),
            "normal": chosen["normal"].copy(),
            "reprojection_error": float(chosen["error"]),
            "candidates": candidates,
        }

    def compute_reprojection_error(self, R_mat, tvec, camera_matrix, dist_coeffs, image_points):
        projected = self.project_model_points(R_mat, tvec, camera_matrix, dist_coeffs)
        image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))

    def project_model_points(self, R_mat, tvec, camera_matrix, dist_coeffs):
        rvec, _ = cv2.Rodrigues(R_mat)
        projected, _ = cv2.projectPoints(
            self.model_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        return projected.reshape(-1, 2).astype(float)


    # -------------------------------------------------
    # Confidence
    # -------------------------------------------------
    def compute_confidence(self, pts):

        area = cv2.contourArea(pts)

        return min(area/40000.0,1.0)


    # -------------------------------------------------
    # Pose Smoothing
    # -------------------------------------------------
    def smooth_pose(self, R_mat, tvec):

        pose_vec = np.hstack(

            (
                R.from_matrix(R_mat).as_euler('xyz'),
                tvec.flatten()
            )

        )

        self.pose_history.append(pose_vec)

        avg = np.mean(self.pose_history, axis=0)

        R_smooth = R.from_euler(
            'xyz',
            avg[:3]
        ).as_matrix()

        t_smooth = avg[3:].reshape(3,1)

        return R_smooth,t_smooth


    # -------------------------------------------------
    # Failure Handling
    # -------------------------------------------------
    def handle_failure(self):

        self.failure_counter += 1

        if self.failure_counter > self.max_failures:

            self.pose_history.clear()

            return None

        return "TEMP_LOST"
