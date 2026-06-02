import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
from ultralytics import YOLO


class GatePerception:

    def __init__(self,
                 gate_size=1.5,
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
        self.live_solver_name = "SOLVEPNP_ITERATIVE"

        # YOLO pose settings
        if yolo_model_path is None:
            raise ValueError(
                "GatePerception now uses YOLO. Pass yolo_model_path='runs/pose/.../weights/best.pt'"
            )
        self.yolo_model_path = str(yolo_model_path)
        self.yolo_model = YOLO(self.yolo_model_path)
        self.yolo_conf = float(yolo_conf)
        self.yolo_imgsz = int(yolo_imgsz)
        self.yolo_device = yolo_device
        self.preprocess_mode = preprocess_mode
        self.corners_are_semantic = True  # YOLO pose outputs TL, TR, BR, BL directly.

        print(f"[YOLO PERCEPTION] model={self.yolo_model_path}")
        print(f"[YOLO PERCEPTION] preprocess_mode={self.preprocess_mode}, conf={self.yolo_conf}, imgsz={self.yolo_imgsz}")
        print("[LIVE PNP] using SOLVEPNP_ITERATIVE")

        # TII keypoint labels are inner opening corners, so use the inner opening size.
        # Competition/spec gate: inner opening = 1.5 m x 1.5 m.
        # If you switch back to HSV outer-frame contour detection, use gate_size=2.7 instead.
        s = gate_size / 2.0
        self.model_points = np.array([
            [-s,  s, 0],  # TL inner corner
            [ s,  s, 0],  # TR inner corner
            [ s, -s, 0],  # BR inner corner
            [-s, -s, 0],  # BL inner corner
        ], dtype=np.float32)

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
        for corners in corners_list:
            ordered = self.order_corners(corners)
            result = self._estimate_detection_result(corners, ordered, camera_matrix, dist_coeffs)
            if result is not None:
                detections.append(result)

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
            ordered,
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
            "ordered_corners": ordered.copy(),
            "reprojected_corners": reprojected_corners.copy(),
            "rvec": pnp_debug["rvec"].copy(),
            "tvec": np.asarray(t_out, dtype=float).reshape(3).copy(),
            "reprojection_error": float(reprojection_error),
            "corner_reprojection_error_px": float(reprojection_error),
            "gate_normal_camera": gate_normal_camera.copy(),
            "pnp_candidates": pnp_debug["candidates"],
            "chosen_candidate": int(pnp_debug["chosen_candidate"]),
            "live_solver_name": pnp_debug.get("live_solver_name", "SOLVEPNP_ITERATIVE"),
            "pnp_fallback_reason": pnp_debug.get("fallback_reason", ""),
            "gate_size_sweep": self.solve_pnp_gate_size_sweep(
                ordered,
                camera_matrix,
                dist_coeffs,
                sizes=(1.40, 1.50, 1.60),
            ),
            "pnp_formulation_debug": self.solve_pnp_formulation_debug(
                ordered,
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
            return []

        if result.boxes is None or len(result.boxes) == 0:
            return []
        if result.keypoints is None or result.keypoints.data is None:
            return []

        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        boxes_conf = result.boxes.conf.detach().cpu().numpy()
        keypoints = result.keypoints.data.detach().cpu().numpy()  # shape: N x K x 3

        candidates = []
        h, w = frame_bgr.shape[:2]

        for i in range(len(boxes_conf)):
            if keypoints.shape[1] < 4:
                continue

            pts = keypoints[i, :4, :2].astype(np.float32)
            kconf = keypoints[i, :4, 2].astype(float) if keypoints.shape[2] >= 3 else np.ones(4)

            # Reject broken keypoint sets.
            if not np.isfinite(pts).all():
                continue
            if np.any(pts[:, 0] < -5) or np.any(pts[:, 0] > w + 5):
                continue
            if np.any(pts[:, 1] < -5) or np.any(pts[:, 1] > h + 5):
                continue

            # Very low keypoint confidence means the box may be okay but corners are not.
            mean_kconf = float(np.mean(kconf))
            if mean_kconf < 0.10:
                continue

            area = abs(cv2.contourArea(pts))
            if area < 50:
                continue

            # Score by bbox confidence, keypoint confidence, and keypoint quad area.
            score = float(boxes_conf[i]) + 0.25 * mean_kconf + min(area / 40000.0, 1.0) * 0.10

            candidates.append((score, float(boxes_conf[i]), mean_kconf, pts))

        candidates.sort(key=lambda item: item[0], reverse=True)

        # De-duplicate similar detections.
        deduped = []
        for score, box_conf, mean_kconf, pts in candidates:
            center = np.mean(pts, axis=0)
            size = np.ptp(pts, axis=0)
            duplicate = False
            for kept_score, kept_box_conf, kept_mean_kconf, kept_pts in deduped:
                kept_center = np.mean(kept_pts, axis=0)
                kept_size = np.ptp(kept_pts, axis=0)
                center_close = np.linalg.norm(center - kept_center) < 12.0
                size_close = np.linalg.norm(size - kept_size) < 20.0
                if center_close and size_close:
                    duplicate = True
                    break
            if not duplicate:
                deduped.append((score, box_conf, mean_kconf, pts))

        return [pts for _, _, _, pts in deduped]

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
    def estimate_pose(self, image_points, camera_matrix, dist_coeffs):
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)

        print("\n------ Perception ------")
        fallback_reason = ""
        try:
            ok, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception as exc:
            ok = False
            fallback_reason = f"iterative_exception:{exc}"

        if ok:
            R_mat, _ = cv2.Rodrigues(rvec)
            reprojection_error = self.compute_reprojection_error(
                R_mat,
                tvec,
                camera_matrix,
                dist_coeffs,
                image_points,
            )
            normal = R_mat[:, 2].astype(float)
            normal /= np.linalg.norm(normal) + 1e-12
            print("[LIVE PNP] solver=SOLVEPNP_ITERATIVE")
            print("  iterative t:", np.asarray(tvec, dtype=float).reshape(3))
            print("  iterative reprojection_error:", reprojection_error)
            candidate = {
                "index": 0,
                "rvec": np.asarray(rvec, dtype=float).reshape(3),
                "tvec": np.asarray(tvec, dtype=float).reshape(3),
                "normal": normal,
                "error": float(reprojection_error),
            }
            return R_mat, tvec, {
                "rvec": np.asarray(rvec, dtype=float).reshape(3),
                "tvec": np.asarray(tvec, dtype=float).reshape(3),
                "chosen_candidate": 0,
                "candidates": [candidate],
                "live_solver_name": "SOLVEPNP_ITERATIVE",
                "fallback_reason": "",
            }

        if not fallback_reason:
            fallback_reason = "iterative_failed"
        print(f"[LIVE PNP] ITERATIVE failed; falling back to SOLVEPNP_IPPE_SQUARE reason={fallback_reason}")

        ok, rvecs, tvecs, reprojErrs = cv2.solvePnPGeneric(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok or rvecs is None or len(rvecs) == 0:
            print("fallback solvePnPGeneric returned no solutions")
            return None

        print(f"fallback solvePnPGeneric returned {len(rvecs)} solutions")

        def err_scalar(e):
            if e is None:
                return 0.0
            return float(np.asarray(e).reshape(-1)[0])

        best_i = 0
        best_score = -1e18

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R_mat, _ = cv2.Rodrigues(rvec)
            n = R_mat[:, 2].astype(float)
            n /= (np.linalg.norm(n) + 1e-12)

            z = n[2]
            e = err_scalar(reprojErrs[i]) if reprojErrs is not None and len(reprojErrs) > i else 0.0
            tz = float(np.asarray(tvec).reshape(-1)[2])

            score = (10.0 * z) - e
            if tz <= 0:
                score -= 1e6

            print(f"  cand {i}: n={n}, n.z={z:.3f}, t.z={tz:.3f}, err={e:.6f}, score={score:.6f}")

            if score > best_score:
                best_score = score
                best_i = i

        print("Chosen candidate:", best_i)

        rvec = rvecs[best_i]
        tvec = tvecs[best_i]
        R_mat, _ = cv2.Rodrigues(rvec)
        candidates = []
        for i, (cand_rvec, cand_tvec) in enumerate(zip(rvecs, tvecs)):
            cand_R, _ = cv2.Rodrigues(cand_rvec)
            cand_normal = cand_R[:, 2].astype(float)
            cand_normal /= np.linalg.norm(cand_normal) + 1e-12
            candidates.append({
                "index": int(i),
                "rvec": np.asarray(cand_rvec, dtype=float).reshape(3),
                "tvec": np.asarray(cand_tvec, dtype=float).reshape(3),
                "normal": cand_normal,
                "error": err_scalar(reprojErrs[i]) if reprojErrs is not None and len(reprojErrs) > i else float("nan"),
            })
        return R_mat, tvec, {
            "rvec": np.asarray(rvec, dtype=float).reshape(3),
            "tvec": np.asarray(tvec, dtype=float).reshape(3),
            "chosen_candidate": best_i,
            "candidates": candidates,
            "live_solver_name": "SOLVEPNP_IPPE_SQUARE_FALLBACK",
            "fallback_reason": fallback_reason,
        }

    @staticmethod
    def model_points_for_size(gate_size: float):
        s = float(gate_size) / 2.0
        return np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0],
        ], dtype=np.float32)

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
