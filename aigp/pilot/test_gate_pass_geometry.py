import numpy as np

from gate_pass_geometry import check_gate_plane_pass, unit_vector_from_to


def test_inside_radius_before_plane_does_not_pass():
    start = np.array([0.55, 0.57, 0.37])
    center = np.array([-1.62, 23.97, 5.09])
    previous = np.array([-1.46, 22.12, 4.70])
    current = np.array([-1.52, 22.78, 4.82])
    normal = unit_vector_from_to(start, center)

    result = check_gate_plane_pass(
        previous_position=previous,
        position=current,
        center=center,
        normal=normal,
        lateral_radius_m=0.75,
        plane_tolerance_m=0.05,
    )

    assert not result.passed
    assert result.reason == "not_past_gate_plane"
    assert result.signed_progress_m < -1.0
    assert result.lateral_error_m < 0.1


def test_crossing_plane_inside_opening_passes():
    center = np.array([0.0, 10.0, 1.5])
    normal = np.array([0.0, 1.0, 0.0])

    result = check_gate_plane_pass(
        previous_position=np.array([0.0, 9.8, 1.5]),
        position=np.array([0.0, 10.2, 1.5]),
        center=center,
        normal=normal,
        lateral_radius_m=0.75,
        plane_tolerance_m=0.05,
    )

    assert result.passed
    assert result.reason == "crossed_gate_plane"
    assert result.crossed_plane
    np.testing.assert_allclose(result.crossing_point, center)


def test_crossing_plane_outside_opening_is_rejected():
    center = np.array([0.0, 10.0, 1.5])
    normal = np.array([0.0, 1.0, 0.0])

    result = check_gate_plane_pass(
        previous_position=np.array([1.0, 9.8, 1.5]),
        position=np.array([1.0, 10.2, 1.5]),
        center=center,
        normal=normal,
        lateral_radius_m=0.75,
        plane_tolerance_m=0.05,
    )

    assert not result.passed
    assert result.reason == "lateral_error_too_large:1.00"
    assert result.crossed_plane
    assert result.lateral_error_m > 0.75
