import numpy as np

from autonomy_core.tools import autolabel_gazebo_yolo_pose as autolabel


def _visible_gazebo_metadata():
    return {
        "image_filename": "frame_000001.jpg",
        "timestamp": 0.0,
        "drone_pos": np.zeros(3, dtype=float),
        "drone_rpy_rad": np.zeros(3, dtype=float),
        "camera_matrix": np.array(
            [
                [320.0, 0.0, 320.0],
                [0.0, 320.0, 180.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "dist_coeffs": np.zeros((5, 1), dtype=float),
        "image_width": 640,
        "image_height": 360,
        "gazebo_camera_pos_world": np.array([0.0, 0.0, 1.35], dtype=float),
        "gazebo_camera_quat_world": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    }


def _box_occluder(center_world, half_extents, name="test_box"):
    return {
        "name": name,
        "center_world": np.asarray(center_world, dtype=float),
        "rotation_world_from_box": np.eye(3, dtype=float),
        "half_extents": np.asarray(half_extents, dtype=float),
    }


def _single_gate_geometry(center_world, yaw_rad, gate_occluders):
    return {
        "source": "test",
        "gazebo_centers": (np.asarray(center_world, dtype=float),),
        "gazebo_yaws": (float(yaw_rad),),
        "gazebo_occluders": (tuple(gate_occluders),),
        "gazebo_frame": "test_gazebo",
        "mavsdk_centers": (np.asarray(center_world, dtype=float),),
        "mavsdk_yaws": (float(yaw_rad),),
        "mavsdk_occluders": ((),),
        "mavsdk_frame": "test_mavsdk",
    }


def test_write_yaml_supports_inner4_outer4(tmp_path):
    default_yaml_path = autolabel.write_yaml(tmp_path)
    default_text = default_yaml_path.read_text(encoding="utf-8")
    assert "kpt_shape: [4, 3]" in default_text
    assert "flip_idx: [1, 0, 3, 2]" in default_text

    yaml_path = autolabel.write_yaml(
        tmp_path,
        keypoint_layout=autolabel.KEYPOINT_LAYOUT_INNER4_OUTER4,
    )

    text = yaml_path.read_text(encoding="utf-8")
    assert "kpt_shape: [8, 3]" in text
    assert "flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]" in text


def test_gate_corners_use_exit_face_inner_and_visible_entry_face_outer():
    center = np.array([8.0, 0.0, 1.35], dtype=float)
    gate_yaw_rad = np.pi / 2.0
    expected_exit_x = center[0] + autolabel.GATE_EXIT_FACE_OFFSET_M
    expected_entry_x = center[0] - autolabel.GATE_EXIT_FACE_OFFSET_M

    inner = autolabel.gate_inner_corners_world(center, gate_yaw_rad)
    outer = autolabel.gate_outer_corners_world(center, gate_yaw_rad)

    np.testing.assert_allclose(inner[:, 0], np.full(4, expected_exit_x), atol=1e-9)
    np.testing.assert_allclose(outer[:, 0], np.full(4, expected_entry_x), atol=1e-9)
    assert np.isclose(autolabel.GATE_EXIT_FACE_OFFSET_M, 0.130)


def test_same_gate_self_occlusion_ignores_endpoint_only_contact():
    gate_occluders = ((
        _box_occluder(
            center_world=[0.0, 0.0, 0.0],
            half_extents=[0.5, 0.5, 0.5],
            name="gate_self_box",
        ),
    ),)
    camera_world = np.array([-2.0, 0.0, 0.0], dtype=float)
    keypoints_world = np.array(
        [
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ],
        dtype=float,
    )

    occluded, occluders = autolabel.keypoint_same_gate_occlusion(
        camera_world,
        keypoints_world,
        target_gate_idx=0,
        gate_occluders=gate_occluders,
    )

    np.testing.assert_array_equal(occluded, np.array([True, False]))
    assert occluders[0] == "self:gate_self_box"
    assert occluders[1] == ""


def test_build_labels_marks_same_gate_self_occlusion_visibility_one():
    metadata = _visible_gazebo_metadata()
    center = np.array([8.0, 0.0, 1.35], dtype=float)
    yaw_rad = np.pi / 2.0
    first_inner_corner = autolabel.gate_inner_corners_world(center, yaw_rad)[0]
    camera_world = metadata["gazebo_camera_pos_world"]
    occluder_center = camera_world + 0.5 * (first_inner_corner - camera_world)
    gate_geometry = _single_gate_geometry(
        center,
        yaw_rad,
        [
            _box_occluder(
                center_world=occluder_center,
                half_extents=[0.08, 0.08, 0.08],
                name="same_gate_test_bar",
            ),
        ],
    )

    labels, _candidates = autolabel.build_labels(
        metadata,
        gate_geometry=gate_geometry,
        order_image_corners=False,
        gazebo_rotation_mode="direct",
        gazebo_optical_mode="current",
    )

    assert labels
    label = labels[0]
    assert label["keypoint_visibility"][0] == 1
    assert label["keypoint_occluded"][0]
    assert label["keypoint_occluders"][0] == "self:same_gate_test_bar"


def test_build_labels_inner4_outer4_emits_eight_keypoints():
    labels, candidates = autolabel.build_labels(
        _visible_gazebo_metadata(),
        keypoint_layout=autolabel.KEYPOINT_LAYOUT_INNER4_OUTER4,
        gazebo_rotation_mode="direct",
        gazebo_optical_mode="current",
        enable_gate_occlusion=False,
    )

    assert candidates
    assert labels
    label = labels[0]
    assert label["keypoint_layout"] == autolabel.KEYPOINT_LAYOUT_INNER4_OUTER4
    assert label["keypoints_image"].shape == (8, 2)
    assert label["projected_keypoints_clipped"].shape == (8, 2)
    np.testing.assert_array_equal(label["keypoint_visibility"], np.full(8, 2))

    line = autolabel.yolo_pose_line(
        label["bbox_points_image"],
        label["keypoints_image"],
        label["keypoint_visibility"],
        640,
        360,
    )
    assert len(line.split()) == 5 + 8 * 3
