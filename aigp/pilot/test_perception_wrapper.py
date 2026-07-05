from types import SimpleNamespace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from autonomy_core.core.competition_config import VADR_TS_002
from perception_wrapper import PerceptionWrapper


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
