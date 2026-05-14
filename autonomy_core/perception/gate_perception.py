import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque


class GatePerception:

    def __init__(self,
                 gate_size=2.0,
                 smoothing_window=5,
                 max_failures=10):

        self.gate_size = gate_size
        self.pose_history = deque(maxlen=smoothing_window)
        self.max_failures = max_failures
        self.failure_counter = 0
        self.last_debug = {}
        self.live_solver_name = "SOLVEPNP_ITERATIVE"
        print("[LIVE PNP] using SOLVEPNP_ITERATIVE")

        # Exact 1m square model
        s = gate_size / 2.0
        self.model_points = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0]
        ], dtype=np.float32)
        # ADD THESE LINES HERE
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
        self.last_debug = {
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
                sizes=(1.90, 2.00, 2.10),
            ),
            "pnp_formulation_debug": self.solve_pnp_formulation_debug(
                ordered,
                camera_matrix,
                dist_coeffs,
            ),
        }

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
        print(t_out.flatten())

        print("confidence:", confidence)
        print("reprojection_error:", reprojection_error)

        return {
            "R": R_out,
            "t": t_out,
            "confidence": confidence,
            "debug": self.last_debug.copy(),
        }

    # -------------------------------------------------
    # Look for holes algorithm
    # -------------------------------------------------
    def detect_gate(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- Orange mask (tune these) ---
        lower_orange = np.array([5, 80, 80])
        upper_orange = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Clean up mask: fill small gaps / remove noise
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, k, iterations=1)

        # cv2.imshow("Full_Orange_Mask", mask_opened)

        # Find contours on the COLOR mask (not grayscale edges)
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_box = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            rect = cv2.minAreaRect(cnt)
            (w, h) = rect[1]
            if w <= 1 or h <= 1:
                continue

            aspect = max(w, h) / (min(w, h) + 1e-6)

            # Gate should be roughly square-ish (tune range as needed)
            if aspect > 1.8:
                continue

            if area > best_area:
                best_area = area
                box = cv2.boxPoints(rect)
                best_box = box.astype(np.float32)

        return best_box
        #     # --- GEOMETRY ---
        #     peri = cv2.arcLength(cnt, True)
        #     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        #
        #     # Accept a wider range of "rectangles" (3 to 10 points) just to get a hit
        #     if 3 <= len(approx) <= 10:
        #         child_idx = hierarchy[i][2]
        #         if child_idx != -1:
        #             child_cnt = contours[child_idx]
        #             child_peri = cv2.arcLength(child_cnt, True)
        #             child_approx = cv2.approxPolyDP(child_cnt, 0.02 * child_peri, True)
        #             if len(child_approx) >= 4:
        #                 return child_approx.reshape(-1, 2)
        #
        #         if area > max_area:
        #             max_area = area
        #             best_candidate = approx.reshape(-1, 2)
        #
        # cv2.waitKey(1)
        # return best_candidate
    # -------------------------------------------------
    # Correct Corner Ordering
    # -------------------------------------------------
    def order_corners(self, corners):
        pts = np.asarray(corners, dtype=np.float32)
        if pts.shape[0] < 4:
            return None

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
