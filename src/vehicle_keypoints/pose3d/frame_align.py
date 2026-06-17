"""Fixed rotation between our canonical model frame and Apollo's CAD frame.

For the same physical car: R_pred ~= R_gt @ MODEL_TO_APOLLO. The value is
calibrated empirically (scripts/calibrate_pose3d_frame.py) and validated by
the qualitative overlay before being trusted. It starts as identity.
"""

from __future__ import annotations

import numpy as np

# Calibrated constant (Task 8). Identity until calibration sets it.
MODEL_TO_APOLLO: np.ndarray = np.eye(3)


def aligned_gt_rotation(r_gt: np.ndarray) -> np.ndarray:
    """Map a GT rotation into our model frame for comparison with R_pred."""
    return np.asarray(r_gt) @ MODEL_TO_APOLLO
