import numpy as np
from autonomy_core.planning.test import GateTrajectoryPlanner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def planner_visual(planner, p0, p1, T):
    t_steps = np.linspace(0, T, 100)
    positions, velocities, accelerations = [], [], []

    for t in t_steps:
        p, v, a = planner.sample(t)
        positions.append(p)
        velocities.append(v)
        accelerations.append(a)

    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    print (positions)

    fig = plt.figure(figsize=(12, 8))

    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='blue', label='Flight Path')
    ax3d.scatter(*p0, color='green', label='Start')
    ax3d.scatter(*p1, color='red', label='Gate')
    ax3d.set_title("3D Trajectory")
    ax3d.legend()

    ax_vel = fig.add_subplot(2, 2, 3)
    ax_vel.plot(t_steps, velocities[:, 0], label='Vx')
    ax_vel.plot(t_steps, velocities[:, 1], label='Vy')
    ax_vel.plot(t_steps, velocities[:, 2], label='Vz')
    ax_vel.set_title("Velocity")
    ax_vel.set_ylabel("m/s")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.legend()

    ax_acc = fig.add_subplot(2, 2, 4)
    ax_acc.plot(t_steps, accelerations[:, 0], label='Ax')
    ax_acc.plot(t_steps, accelerations[:, 1], label='Ay')
    ax_acc.plot(t_steps, accelerations[:, 2], label='Az')
    ax_acc.set_title("Acceleration")
    ax_acc.set_ylabel("m/s²")
    ax_acc.set_xlabel("Time (s)")
    ax_acc.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    planner = GateTrajectoryPlanner()

    p0 = np.array([10, 0, 10])
    v0 = np.array([1, 0, 0])
    p1 = np.array([10, 5, 5])
    v1 = np.array([5, 0, 0])
    T = 5.0


    planner.update(p0, v0, p1, v1, T)
    planner_visual(planner, p0, p1, T)