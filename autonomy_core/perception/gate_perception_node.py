import numpy as np


class GatePerceptionNode:
    """
    Wraps an existing GatePerception module and converts detections
    into world-frame gate center measurements.
    """

    def __init__(self, gate_perception, camera_to_body_rotmat=None):
        self.gate_perception = gate_perception
        if camera_to_body_rotmat is None:
            # OpenCV optical frame: x right, y down, z forward.
            # Body/world convention used here: x forward, y left, z up.
            camera_to_body_rotmat = np.array([
                [-1.0,  0.0, 0.0],
                [0.0, 0.0, 1.0],
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

        gate_world = np.asarray(drone_pos, dtype=float).reshape(3) + R_wb @ gate_body

        return {
            "confidence": float(conf),
            "gate_center_camera": gate_camera,
            "gate_center_body": gate_body,
            "gate_center_cam": gate_body,
            "gate_center_world": gate_world,
            "raw": perception,
        }
