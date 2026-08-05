import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_openvins_gate_alignment import (  # noqa: E402
    AlignmentOptions,
    OfflineGateMapAligner,
    _match_camera_rows,
    _ned_to_neu,
)


def _options(**overrides):
    values = {
        "min_confidence": 0.7,
        "max_reprojection_error": 1.5,
        "min_depth_m": 2.0,
        "max_depth_m": 45.0,
        "initial_association_radius_m": 6.0,
        "association_radius_m": 3.0,
        "temporal_window_s": 1.0,
        "min_consistent_frames": 3,
        "consistency_radius_m": 0.2,
        "max_initial_offset_m": 6.0,
        "max_update_innovation_m": 1.5,
        "correction_alpha": 1.0,
        "max_step_m": 2.0,
        "stale_after_s": 2.0,
    }
    values.update(overrides)
    return AlignmentOptions(**values)


def _detection(relative_neu, *, index=0, confidence=0.95, reprojection=0.2):
    return {
        "detection_index": index,
        "confidence": confidence,
        "reprojection_error": reprojection,
        "gate_center_camera": np.asarray((0.0, 0.0, 10.0)),
        "relative_gate_neu": np.asarray(relative_neu, dtype=float),
    }


class OfflineGateMapAlignerTests(unittest.TestCase):
    def test_initializes_from_consistent_metric_gate_observations(self):
        aligner = OfflineGateMapAligner(((10.0, 0.0, 0.0),), _options())
        raw_position = np.asarray((1.0, 1.0, 0.0))
        # The actual vehicle is at (2, 0, 0), so map minus raw is (1, -1, 0).
        relative_gate = np.asarray((8.0, 0.0, 0.0))

        decisions = []
        for frame in range(3):
            decisions.append(
                aligner.update(
                    0.1 * frame,
                    raw_position,
                    [_detection(relative_gate)],
                )
            )

        self.assertFalse(decisions[1]["initialized"])
        self.assertTrue(decisions[2]["initialized"])
        self.assertTrue(decisions[2]["accepted"])
        np.testing.assert_allclose(aligner.offset_neu, (1.0, -1.0, 0.0))
        np.testing.assert_allclose(
            decisions[2]["aligned_position_neu"], (2.0, 0.0, 0.0)
        )

    def test_rejects_low_confidence_and_large_update_innovation(self):
        aligner = OfflineGateMapAligner(
            ((10.0, 0.0, 0.0),),
            _options(min_consistent_frames=1, max_update_innovation_m=0.5),
        )
        initialized = aligner.update(
            0.0,
            np.zeros(3),
            [_detection((9.0, 0.0, 0.0))],
        )
        self.assertTrue(initialized["initialized"])
        np.testing.assert_allclose(aligner.offset_neu, (1.0, 0.0, 0.0))

        low_confidence = aligner.update(
            0.1,
            np.zeros(3),
            [_detection((9.0, 0.0, 0.0), confidence=0.2)],
        )
        self.assertFalse(low_confidence["accepted"])
        self.assertEqual(low_confidence["reason"], "no_geometry_measurements")

        bad_innovation = aligner.update(
            0.2,
            np.zeros(3),
            [_detection((7.0, 0.0, 0.0))],
        )
        self.assertFalse(bad_innovation["accepted"])
        self.assertEqual(bad_innovation["reason"], "no_gated_measurements")
        np.testing.assert_allclose(aligner.offset_neu, (1.0, 0.0, 0.0))

    def test_ned_to_neu_changes_only_vertical_sign(self):
        values = np.asarray(((1.0, 2.0, 3.0), (-4.0, 5.0, -6.0)))
        np.testing.assert_allclose(
            _ned_to_neu(values), ((1.0, 2.0, -3.0), (-4.0, 5.0, 6.0))
        )

    def test_camera_matching_requires_exact_timestamp_correspondence(self):
        rows = [
            {"timestamp": 1.0, "frame_id": 10},
            {"timestamp": 2.0, "frame_id": 11},
            {"timestamp": 3.0, "frame_id": 12},
        ]
        matched = _match_camera_rows(rows, np.asarray((1.0, 2.0, 3.0)))
        self.assertEqual([row["frame_id"] for row in matched], [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
