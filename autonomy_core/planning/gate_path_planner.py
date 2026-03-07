import numpy as np


class GatePathPlanner:
    def __init__(self,
                 approach_distance=2.0,
                 exit_distance=2.0,
                 smoothing=0.3,
                 normal_axis=2,          # 0 -> R[:,0], 1 -> R[:,1], 2 -> R[:,2]
                 enforce_drone_side=True # auto-flip normal so approach is on drone side
                 ):

        self.approach_distance = float(approach_distance)
        self.exit_distance = float(exit_distance)
        self.smoothing = float(smoothing)

        self.normal_axis = int(normal_axis)
        self.enforce_drone_side = bool(enforce_drone_side)

        self.last_target = None
        self.last_plan = None


    def plan(self, perception_output):

        if perception_output is None:
            return None

        if perception_output == "TEMP_LOST":
            # Return the last full plan if available, otherwise just last target
            return self.last_plan if self.last_plan is not None else self.last_target


        R = np.asarray(perception_output["R"], dtype=float)
        t = np.asarray(perception_output["t"], dtype=float)

        gate_center = t.reshape(-1)  # ensures shape (3,)

        if gate_center.shape[0] != 3:
            raise ValueError(f"Expected t to have 3 elements, got shape {t.shape}")

        if R.shape != (3, 3):
            raise ValueError(f"Expected R shape (3,3), got {R.shape}")


        # ---------------------------------------
        # Gate normal direction (choose axis)
        # ---------------------------------------
        gate_normal = R[:, self.normal_axis].astype(float)

        nrm = np.linalg.norm(gate_normal)
        if nrm < 1e-9:
            return None

        gate_normal = gate_normal / nrm


        # ---------------------------------------
        # IMPORTANT: Flip normal so APPROACH is on drone side
        # Assumption: drone/camera at origin, gate_center points from drone -> gate
        # We want gate_normal to generally point in the same half-space as gate_center,
        # so that approach_point = gate_center - gate_normal*d moves toward the drone.
        # ---------------------------------------
        if self.enforce_drone_side:
            # If normal points opposite the gate direction, flip it
            if np.dot(gate_normal, gate_center) < 0.0:
                gate_normal = -gate_normal


        # ---------------------------------------
        # Waypoints
        # ---------------------------------------
        approach_point = gate_center - gate_normal * self.approach_distance
        pass_point = gate_center
        exit_point = gate_center + gate_normal * self.exit_distance


        # Choose stage based on distance to gate center
        dist_to_gate = np.linalg.norm(gate_center)

        if dist_to_gate > self.approach_distance:
            target = approach_point
            stage = "APPROACH"
        elif dist_to_gate > 0.5:
            target = pass_point
            stage = "PASS"
        else:
            target = exit_point
            stage = "EXIT"


        # ---------------------------------------
        # Smooth target motion
        # ---------------------------------------
        if self.last_target is None:
            smoothed = target
        else:
            smoothed = (
                self.last_target * (1.0 - self.smoothing)
                + target * self.smoothing
            )

        self.last_target = smoothed


        plan = {
            "target_position": smoothed,
            "stage": stage,
            "gate_center": gate_center,
            "gate_normal": gate_normal,
            "approach_point": approach_point,
            "pass_point": pass_point,
            "exit_point": exit_point,
            "normal_axis": self.normal_axis,
        }

        self.last_plan = plan


        # Debug prints
        print("\n------ Path Plan ------")
        print("Stage:", stage)
        print("Gate center (m):", gate_center)
        print("Gate normal:", gate_normal)
        print("Approach point:", approach_point)
        print("Target position (m):", smoothed)

        return plan