import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from autonomy_core.planning.gate_path_planner import GatePathPlanner


class PathVisualizer:

    def __init__(self):

        self.planner = GatePathPlanner()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        plt.show(block=False)

    def visualize(self, plan):

        if plan is None:
            return

        gate_center = plan["gate_center"]
        gate_normal = plan["gate_normal"]
        target = plan["target_position"]

        drone = np.array([0, 0, 0])

        approach = gate_center - gate_normal * self.planner.approach_distance
        exitp = gate_center + gate_normal * self.planner.exit_distance

        # Clear entire figure instead of axes
        self.fig.clear()

        self.ax = self.fig.add_subplot(111, projection='3d')

        # Points
        self.ax.scatter([drone[0]], [drone[1]], [drone[2]], s=80)

        self.ax.scatter(
            [float(gate_center[0])],
            [float(gate_center[1])],
            [float(gate_center[2])],
            s=80
        )

        self.ax.scatter(
            [float(approach[0])],
            [float(approach[1])],
            [float(approach[2])],
            s=80
        )

        self.ax.scatter(
            [float(exitp[0])],
            [float(exitp[1])],
            [float(exitp[2])],
            s=80
        )

        self.ax.scatter(
            [float(target[0])],
            [float(target[1])],
            [float(target[2])],
            s=120
        )

        # Gate normal arrow
        self.ax.quiver(
            gate_center[0],
            gate_center[1],
            gate_center[2],
            gate_normal[0],
            gate_normal[1],
            gate_normal[2],
            length=1.0
        )

        # Path line
        path = np.vstack([drone, target])

        self.ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2]
        )

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-5, 5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.show()