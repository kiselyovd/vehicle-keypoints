"""6DoF pose error metrics (no scipy; pure numpy)."""

from __future__ import annotations

import numpy as np


def geodesic_rotation_deg(r_pred: np.ndarray, r_gt: np.ndarray) -> float:
    """Shortest geodesic angle (degrees) between two rotation matrices."""
    rel = r_pred.T @ r_gt
    cos = (np.trace(rel) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def translation_error_m(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Euclidean translation error in metres."""
    return float(np.linalg.norm(np.asarray(t_pred).ravel() - np.asarray(t_gt).ravel()))
