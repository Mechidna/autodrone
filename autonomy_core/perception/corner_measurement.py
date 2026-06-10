from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CornerMeasurement:
    """Passive four-corner image measurement for offline estimation work."""

    image_stamp: Tuple[int, int]
    camera_pose_world: np.ndarray
    keypoints_px: np.ndarray
    keypoint_conf: np.ndarray
    visibility: np.ndarray
    associated_track_id: Optional[int]
    race_index_candidate: Optional[int]

    def __post_init__(self):
        object.__setattr__(
            self,
            "camera_pose_world",
            np.asarray(self.camera_pose_world, dtype=float).reshape(4, 4).copy(),
        )
        object.__setattr__(
            self,
            "keypoints_px",
            np.asarray(self.keypoints_px, dtype=float).reshape(4, 2).copy(),
        )
        object.__setattr__(
            self,
            "keypoint_conf",
            np.asarray(self.keypoint_conf, dtype=float).reshape(4).copy(),
        )
        object.__setattr__(
            self,
            "visibility",
            np.asarray(self.visibility, dtype=bool).reshape(4).copy(),
        )
