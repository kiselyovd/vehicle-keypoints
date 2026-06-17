"""Fixed rotation between our canonical model frame and Apollo's CAD frame.

For the same physical car: R_pred ~= R_gt @ MODEL_TO_APOLLO. The value is
calibrated empirically (scripts/calibrate_pose3d_frame.py) and validated by
the qualitative overlay before being trusted. It starts as identity.
"""

from __future__ import annotations

import numpy as np

# Calibrated constant: data-driven SO(3) rotation averaging over 57 matched
# (predicted, GT) car pairs on ApolloCar3D val (scripts/calibrate_pose3d_frame.py).
# Drops the median rotation error from 109.4 deg (A = I) to 15.5 deg.
MODEL_TO_APOLLO: np.ndarray = np.array(
    [
        [0.04839175014657592, -0.9987186608435892, -0.014807937079130445],
        [0.2812307456776302, -0.0006018116992685564, 0.9596399874475239],
        [-0.9584192747453256, -0.05060310568993528, 0.2808412709898206],
    ]
)


def aligned_gt_rotation(r_gt: np.ndarray) -> np.ndarray:
    """Map a GT rotation into our model frame for comparison with R_pred."""
    return np.asarray(r_gt) @ MODEL_TO_APOLLO
