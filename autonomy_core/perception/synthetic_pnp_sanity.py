import itertools

import cv2
import numpy as np

from autonomy_core.perception.gate_perception import GatePerception


def project_gate(perception, rvec, tvec, camera_matrix, dist_coeffs):
    image_points, _ = cv2.projectPoints(
        perception.model_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    return image_points.reshape(-1, 2).astype(np.float32)


def rvec_from_euler_xyz(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry_mat = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    rot = rz_mat @ ry_mat @ rx_mat
    rvec, _ = cv2.Rodrigues(rot)
    return rvec.reshape(3)


def assert_pose(perception, image_points, expected_tvec, camera_matrix, dist_coeffs, name):
    ordered = perception.order_corners(image_points)
    pose = perception.estimate_pose(ordered, camera_matrix, dist_coeffs)
    if pose is None:
        raise AssertionError(f"{name}: pose failed")

    _, tvec, _ = pose
    tvec = np.asarray(tvec, dtype=float).reshape(3)
    err = float(np.linalg.norm(tvec - expected_tvec))
    if err > 0.05:
        raise AssertionError(f"{name}: tvec error {err:.3f}, got {tvec}, expected {expected_tvec}")


def main():
    perception = GatePerception(gate_size=2.0, smoothing_window=1)
    camera_matrix = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)

    cases = [
        ("frontal_centered", rvec_from_euler_xyz(np.pi, 0.0, 0.0), np.array([0.0, 0.0, 8.0])),
        ("frontal_offset", rvec_from_euler_xyz(np.pi, 0.0, 0.0), np.array([0.6, -0.3, 8.0])),
        ("yawed_view", rvec_from_euler_xyz(np.pi, np.deg2rad(18.0), 0.0), np.array([0.4, 0.2, 7.5])),
    ]

    for name, rvec, tvec in cases:
        image_points = project_gate(perception, rvec, tvec, camera_matrix, dist_coeffs)
        assert_pose(perception, image_points, tvec, camera_matrix, dist_coeffs, name)

        for perm in itertools.permutations(range(4)):
            permuted = image_points[list(perm)]
            assert_pose(
                perception,
                permuted,
                tvec,
                camera_matrix,
                dist_coeffs,
                f"{name}_perm_{perm}",
            )

    print("synthetic PnP sanity checks passed")


if __name__ == "__main__":
    main()
