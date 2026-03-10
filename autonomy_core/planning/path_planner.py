import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Reuse the Planner Logic ---
class GateTrajectoryPlanner:
    def __init__(self):
        self.coeffs = {"x": np.zeros(4), "y": np.zeros(4), "z": np.zeros(4)}
        self.duration = 1.0

    def solve_cubic(self, p0, v0, p1, v1, T):
        # Solves: p(t) = at^3 + bt^2 + ct + d
        M = np.array([
            [0, 0, 0, 1],
            [T**3, T**2, T, 1],
            [0, 0, 1, 0],
            [3*T**2, 2*T, 1, 0]
        ])
        b = np.array([p0, p1, v0, v1])
        return np.linalg.solve(M, b)

    def update(self, p0, v0, p1, v1, duration):
        self.duration = duration
        for i, axis in enumerate(["x", "y", "z"]):
            # We solve for each axis independently
            self.coeffs[axis] = self.solve_cubic(p0[i], v0[i], p1[i], v1[i], duration)

    def sample(self, t):
        t = np.clip(t, 0, self.duration)
        pos, vel, accel = [], [], []
        for axis in ["x", "y", "z"]:
            c = self.coeffs[axis]
            pos.append(np.polyval(c, t))
            vel.append(np.polyval(np.polyder(c), t))
            accel.append(np.polyval(np.polyder(c, 2), t))
        return np.array(pos), np.array(vel), np.array(accel)