import random
import time
import numpy as np
from autonomy_core.planning.test import GateTrajectoryPlanner
import matplotlib.pyplot as plt

# Interactive mode lets figures update without blocking
plt.ion()

# Persistent figure/axes references
fig_traj = None
ax_traj = None

fig_vel = None
ax_vel = None

fig_acc = None
ax_acc = None


import numpy as np
import matplotlib.pyplot as plt

plt.ion()

fig_traj = None
ax_traj = None

fig_vel = None
ax_vel = None

fig_acc = None
ax_acc = None


def planner_visual(planner, current_pos, target_pos, T, time_elapsed=None):
    global fig_traj, ax_traj, fig_vel, ax_vel, fig_acc, ax_acc

    if fig_traj is None or not plt.fignum_exists(fig_traj.number):
        fig_traj = plt.figure("3D Trajectory", figsize=(8, 6))
        ax_traj = fig_traj.add_subplot(111, projection="3d")

    if fig_vel is None or not plt.fignum_exists(fig_vel.number):
        fig_vel = plt.figure("Velocity", figsize=(8, 5))
        ax_vel = fig_vel.add_subplot(111)

    if fig_acc is None or not plt.fignum_exists(fig_acc.number):
        fig_acc = plt.figure("Acceleration", figsize=(8, 5))
        ax_acc = fig_acc.add_subplot(111)

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

    ax_traj.cla()
    ax_traj.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Planned Path")
    ax_traj.scatter(*current_pos, label="Real Drone Position")
    ax_traj.scatter(*target_pos, label="Target")

    if time_elapsed is not None:
        ref_p, _, _ = planner.sample(time_elapsed)
        ax_traj.scatter(*ref_p, label="Reference Point")

    ax_traj.set_title("3D Trajectory")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    ax_traj.set_zlabel("Z")
    ax_traj.legend()

    ax_traj.set_xlim(-12, 12)
    ax_traj.set_ylim(-12, 12)
    ax_traj.set_zlim(0, 12)
    ax_traj.set_xticks(np.arange(-12, 13, 2))
    ax_traj.set_yticks(np.arange(-12, 13, 2))
    ax_traj.set_zticks(np.arange(0, 13, 2))
    ax_traj.set_autoscale_on(False)

    fig_traj.canvas.draw()
    fig_traj.canvas.flush_events()

    ax_vel.cla()
    ax_vel.plot(t_steps, velocities[:, 0], label="Vx")
    ax_vel.plot(t_steps, velocities[:, 1], label="Vy")
    ax_vel.plot(t_steps, velocities[:, 2], label="Vz")
    ax_vel.set_title("Planned Velocity")
    ax_vel.set_ylabel("m/s")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.legend()
    ax_vel.grid(True)
    fig_vel.canvas.draw()
    fig_vel.canvas.flush_events()

    ax_acc.cla()
    ax_acc.plot(t_steps, accelerations[:, 0], label="Ax")
    ax_acc.plot(t_steps, accelerations[:, 1], label="Ay")
    ax_acc.plot(t_steps, accelerations[:, 2], label="Az")
    ax_acc.set_title("Planned Acceleration")
    ax_acc.set_ylabel("m/s²")
    ax_acc.set_xlabel("Time (s)")
    ax_acc.legend()
    ax_acc.grid(True)
    fig_acc.canvas.draw()
    fig_acc.canvas.flush_events()

    plt.pause(0.01)


if __name__ == "__main__":
    planner = GateTrajectoryPlanner()

    # Initial state
    p0 = np.array([10.0, 0.0, 10.0], dtype=float)
    v0 = np.array([1.0, 0.0, 0.0], dtype=float)
    p1 = np.array([10.0, 5.0, 5.0], dtype=float)
    v1 = np.array([5.0, 0.0, 0.0], dtype=float)
    T = 5.0

    # Build first trajectory
    planner.update(p0, v0, p1, v1, T)
    segment_start_time = time.time()

    while True:
        if fig_traj is not None and not plt.fignum_exists(fig_traj.number):
            break

        # Time along current segment
        t_now = time.time() - segment_start_time
        t_sample = min(t_now, T)
        print(t_sample)

        # Sample current trajectory
        p_sample, v_sample, a_sample = planner.sample(t_sample)
        print(p_sample, v_sample, a_sample)
        # Drone follows the path perfectly
        p0 = np.array(p_sample, dtype=float)
        v0 = np.array(v_sample, dtype=float)

        # Draw current state
        planner_visual(planner, p0, p1, T)

        # When current segment ends, generate a new one
        if t_now >= T:
            # Start next segment from current endpoint
            p0 = np.array(p_sample, dtype=float)
            v0 = np.array(v_sample, dtype=float)

            # Generate a new random target and end velocity
            p1 = np.random.uniform(0.0, 20.0, size=3)
            v1 = np.random.uniform(-3.0, 3.0, size=3)
            T = random.uniform(3.0, 8.0)

            # Rebuild planner for the new segment
            planner.update(p0, v0, p1, v1, T)

            # Reset timer for the new segment
            segment_start_time = time.time()

    print("Trajectory loop stopped.")

    plt.ioff()
    plt.show()