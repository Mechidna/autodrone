import unittest

import numpy as np

from autonomy_core.planning.minimum_snap_planner_multi_time_optimized import (
    MultiSegmentMinimumSnapPlanner,
)


def _minimum_segment_forward_speed(planner, waypoints, samples_per_segment=401):
    minimum = float("inf")
    for segment_idx, duration in enumerate(planner.times):
        chord = waypoints[segment_idx + 1] - waypoints[segment_idx]
        direction = chord / np.linalg.norm(chord)
        start = planner.segment_starts[segment_idx]
        for tau in np.linspace(start, start + duration, samples_per_segment):
            _, velocity, _ = planner.sample(float(tau))
            minimum = min(minimum, float(np.dot(direction, velocity)))
    return minimum


class ForwardProgressMinimumSnapTests(unittest.TestCase):
    def test_uniform_retime_preserves_curve_and_scales_derivatives(self):
        waypoints = np.array(
            [
                [0.0, 0.0, 1.0],
                [7.0, 1.0, 2.0],
                [14.0, -2.0, 0.5],
                [20.0, 0.0, 1.5],
            ],
            dtype=float,
        )
        planner = MultiSegmentMinimumSnapPlanner()
        planner.update(
            waypoints,
            times=[2.5, 3.0, 2.25],
            v_start=[1.2, 0.1, -0.2],
            v_end=[0.4, 0.0, 0.1],
            forward_progress_enabled=True,
        )

        scale = 1.7
        retimed = planner.retimed(scale)

        np.testing.assert_allclose(retimed.times, planner.times * scale)
        self.assertAlmostEqual(retimed.total_time, planner.total_time * scale)
        self.assertIsNot(retimed, planner)
        for old_time in np.linspace(0.0, planner.total_time, 41):
            old_state = planner.sample_full(float(old_time))
            new_state = retimed.sample_full(float(old_time * scale))
            for derivative, (old_value, new_value) in enumerate(
                zip(old_state, new_state)
            ):
                np.testing.assert_allclose(
                    new_value,
                    old_value / (scale ** derivative),
                    atol=2e-7,
                    rtol=2e-7,
                )

    def test_constraint_removes_recorded_gate_five_reversal(self):
        # Exact installed plan from run 20260731_124358.  The legacy fully
        # constrained interpolation moved from x=-137.58 back to x=-134.83
        # after Gate 5 even though Gate 6 was at x=-160.
        waypoints = np.array(
            [
                [-0.815, -0.034, 0.252],
                [-25.940, -0.540, 1.140],
                [-47.280, -2.250, -4.240],
                [-73.840, 0.990, -12.710],
                [-111.370, -5.250, -23.500],
                [-135.270, -0.880, -24.000],
                [-160.000, -4.500, -24.750],
                [-161.484, -4.717, -24.795],
            ],
            dtype=float,
        )
        times = np.array([7.155, 6.551, 7.814, 10.235, 7.021, 7.169, 1.549])
        waypoint_velocities = np.array(
            [
                [np.nan, np.nan, np.nan],
                [-3.926, -0.195, -0.416],
                [-3.800, 0.075, -1.084],
                [-3.782, -0.084, -1.147],
                [-3.908, 0.045, -0.593],
                [-3.951, 0.070, -0.101],
                [-3.909, -0.572, -0.119],
                [np.nan, np.nan, np.nan],
            ],
            dtype=float,
        )

        legacy = MultiSegmentMinimumSnapPlanner()
        legacy.update(
            waypoints,
            times,
            v_start=[-0.274, -0.011, 0.0],
            v_end=[0.0, 0.0, 0.0],
            waypoint_velocities=waypoint_velocities,
        )
        self.assertLess(_minimum_segment_forward_speed(legacy, waypoints), -0.5)

        constrained = MultiSegmentMinimumSnapPlanner()
        constrained.update(
            waypoints,
            times,
            v_start=[-0.274, -0.011, 0.0],
            v_end=[0.0, 0.0, 0.0],
            waypoint_velocities=waypoint_velocities,
            forward_progress_enabled=True,
        )

        self.assertTrue(
            constrained.forward_progress_solver_status.startswith("bernstein:")
        )
        self.assertGreaterEqual(
            constrained.forward_progress_min_bernstein_speed_m_s,
            -1e-7,
        )
        self.assertGreaterEqual(
            constrained.forward_progress_min_speed_m_s_solved,
            -1e-7,
        )
        self.assertGreaterEqual(
            _minimum_segment_forward_speed(constrained, waypoints),
            -1e-7,
        )

        for segment_idx in range(len(times) - 1):
            left = constrained.get_segment_endpoint_state(segment_idx, at_end=True)
            right = constrained.get_segment_endpoint_state(
                segment_idx + 1,
                at_end=False,
            )
            for derivative_idx in range(3):
                np.testing.assert_allclose(
                    left[derivative_idx],
                    right[derivative_idx],
                    atol=2e-8,
                )

    def test_constraint_uses_each_local_chord_not_a_global_axis(self):
        waypoints = np.array(
            [
                [0.0, 0.0, 0.0],
                [4.0, 1.0, 0.5],
                [2.0, 5.0, 1.0],
                [-3.0, 7.0, 0.0],
            ],
            dtype=float,
        )
        planner = MultiSegmentMinimumSnapPlanner()
        planner.update(
            waypoints,
            times=[3.0, 3.0, 3.0],
            v_start=[0.5, 0.1, 0.0],
            v_end=[0.0, 0.0, 0.0],
            forward_progress_enabled=True,
        )

        self.assertGreaterEqual(
            _minimum_segment_forward_speed(planner, waypoints),
            -1e-7,
        )
        # The course itself changes from positive to negative global X; that is
        # allowed because progress is measured in each segment's local direction.
        midpoint_a = planner.sample(1.5)[0]
        midpoint_c = planner.sample(7.5)[0]
        self.assertGreater(midpoint_a[0], 0.0)
        self.assertLess(midpoint_c[0], waypoints[1, 0])


if __name__ == "__main__":
    unittest.main()
