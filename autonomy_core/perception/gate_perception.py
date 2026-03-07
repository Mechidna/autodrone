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
        cv2.imshow("Gray", frame)
        cv2.waitKey(1)
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

        R_mat, tvec = pose

        confidence = self.compute_confidence(ordered)

        R_smooth, t_smooth = self.smooth_pose(
            R_mat,
            tvec
        )

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

        cv2.imshow("Detection", debug)
        cv2.waitKey(1)

        # Print pose

        print("\n------ Pose ------")

        print("t (meters):")
        print(t_smooth.flatten())

        print("confidence:", confidence)

        return {
            "R": R_smooth,
            "t": t_smooth,
            "confidence": confidence
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

        cv2.imshow("Full_Orange_Mask", mask_opened)

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

        ok, rvecs, tvecs, reprojErrs = cv2.solvePnPGeneric(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        print("\n------ Perception ------")
        if not ok or rvecs is None or len(rvecs) == 0:
            print("solvePnPGeneric returned no solutions")
            return None

        print(f"solvePnPGeneric returned {len(rvecs)} solutions")

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
        return R_mat, tvec


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