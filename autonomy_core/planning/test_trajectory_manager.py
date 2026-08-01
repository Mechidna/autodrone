import numpy as np
import pytest

from autonomy_core.planning.trajectory_manager import allocate_segment_times


def test_allocate_segment_times_preserves_legacy_stop_assumption_by_default():
    waypoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]
    )

    times = allocate_segment_times(
        waypoints,
        current_vel=np.zeros(3),
        vmax=10.0,
        amax=5.0,
        T_min=0.1,
    )

    np.testing.assert_allclose(times, [2.0 * np.sqrt(2.0)] * 2)


def test_allocate_segment_times_uses_passthrough_entry_and_exit_velocities():
    waypoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]
    )
    waypoint_velocities = np.array(
        [
            [np.nan, np.nan, np.nan],
            [5.0, 0.0, 0.0],
            [np.nan, np.nan, np.nan],
        ]
    )

    times = allocate_segment_times(
        waypoints,
        current_vel=np.zeros(3),
        vmax=10.0,
        amax=5.0,
        T_min=0.1,
        waypoint_velocities=waypoint_velocities,
        terminal_vel=np.zeros(3),
    )

    peak_speed = np.sqrt(5.0 * 10.0 + 0.5 * 5.0**2)
    expected = (peak_speed + (peak_speed - 5.0)) / 5.0
    np.testing.assert_allclose(times, [expected, expected])
    assert float(np.sum(times)) < 4.5


def test_allocate_segment_times_projects_turning_gate_velocity_onto_each_segment():
    waypoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
        ]
    )
    tangent = np.array([5.0, 5.0, 0.0]) / np.sqrt(2.0)
    waypoint_velocities = np.array(
        [
            [np.nan, np.nan, np.nan],
            tangent,
            [np.nan, np.nan, np.nan],
        ]
    )

    times = allocate_segment_times(
        waypoints,
        current_vel=np.zeros(3),
        vmax=10.0,
        amax=5.0,
        T_min=0.1,
        waypoint_velocities=waypoint_velocities,
        terminal_vel=np.zeros(3),
    )

    assert times[0] == pytest.approx(times[1])
    assert times[0] < 2.0 * np.sqrt(2.0)


def test_allocate_segment_times_rejects_mismatched_velocity_shape():
    waypoints = np.zeros((3, 3), dtype=float)

    with pytest.raises(ValueError, match="waypoint_velocities"):
        allocate_segment_times(
            waypoints,
            current_vel=np.zeros(3),
            waypoint_velocities=np.zeros((2, 3)),
        )
