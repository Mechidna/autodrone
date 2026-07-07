from types import SimpleNamespace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from autonomy_core.core.competition_config import VADR_TS_002
from perception_wrapper import PerceptionWrapper
from autonomy_core.perception.gate_perception_yolo import (
    KEYPOINT_LAYOUT_INNER4_OUTER4,
    object_points_for_keypoint_layout,
)


def _wrapper() -> PerceptionWrapper:
    return PerceptionWrapper(
        gate_perception=SimpleNamespace(
            model_points=np.asarray(
                VADR_TS_002.gate_inner_object_points_m,
                dtype=float,
            )
        )
    )


def test_normalize_detection_preserves_gate_normals():
    wrapper = _wrapper()
    detection = {
        "gate_center_camera": np.array([0.0, 0.0, 10.0], dtype=float),
        "gate_center_world": np.array([1.0, 2.0, 3.0], dtype=float),
        "gate_normal_camera": np.array([0.0, 0.0, 1.0], dtype=float),
        "gate_normal_world": np.array([0.0, 1.0, 0.0], dtype=float),
        "gate_normal_world_ned": np.array([0.0, 1.0, -0.0], dtype=float),
        "confidence": 0.9,
    }

    normalized = wrapper._normalize_detection(detection, 0)

    np.testing.assert_allclose(normalized["gate_normal_camera"], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        normalized["gate_normal_body"],
        wrapper.camera_to_body @ np.array([0.0, 0.0, 1.0], dtype=float),
    )
    np.testing.assert_allclose(normalized["gate_normal_body_frd"], normalized["gate_normal_body"])
    np.testing.assert_allclose(normalized["gate_normal_world"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(normalized["gate_normal_world_ned"], [0.0, 1.0, 0.0])


def test_normalize_detection_preserves_inner4_outer4_keypoints():
    object_points = object_points_for_keypoint_layout(KEYPOINT_LAYOUT_INNER4_OUTER4)
    wrapper = PerceptionWrapper(
        gate_perception=SimpleNamespace(model_points=object_points)
    )
    yolo_keypoints = np.column_stack(
        [
            np.arange(8, dtype=float),
            np.arange(10, 18, dtype=float),
            np.linspace(0.5, 0.9, 8, dtype=float),
        ]
    )
    detection = {
        "gate_center_camera": np.array([0.0, 0.0, 10.0], dtype=float),
        "yolo_keypoints": yolo_keypoints,
        "confidence": 0.9,
    }

    normalized = wrapper._normalize_detection(detection, 0)

    assert normalized["keypoints_px"].shape == (8, 2)
    assert normalized["keypoint_conf"].shape == (8,)
    assert normalized["object_points_m"].shape == (8, 3)
    np.testing.assert_allclose(normalized["keypoints_px"], yolo_keypoints[:, :2])
    np.testing.assert_allclose(normalized["keypoint_conf"], yolo_keypoints[:, 2])
