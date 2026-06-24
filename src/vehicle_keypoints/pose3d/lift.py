"""2D->3D pose lift via PnP (OpenCV)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_MIN_POINTS = 4


@dataclass(frozen=True)
class PoseResult:
    """Recovered 6DoF pose: X_cam = r @ X_model + t."""

    r: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)
    inliers: int
    reproj_err_px: float


def solve_pose(
    keypoints_2d: np.ndarray,
    visibility: np.ndarray,
    k: np.ndarray,
    model_3d: np.ndarray,
) -> PoseResult | None:
    """Solve model->camera pose from visible 2D<->3D correspondences.

    Returns None when fewer than 4 visible correspondences or PnP fails.
    """
    vis = np.asarray(visibility).ravel() > 0
    if int(vis.sum()) < _MIN_POINTS:
        return None
    obj = np.ascontiguousarray(model_3d[vis], dtype=np.float64)
    img = np.ascontiguousarray(np.asarray(keypoints_2d)[vis], dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, k, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=8.0, iterationsCount=200
    )
    if not ok or inliers is None or len(inliers) < _MIN_POINTS:
        return None
    inl = inliers.ravel()
    # Refine on inliers with the iterative solver.
    _ok, rvec, tvec = cv2.solvePnP(
        obj[inl],
        img[inl],
        k,
        None,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, k, None)
    reproj = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img[inl], axis=1)))
    r = cv2.Rodrigues(rvec)[0]
    return PoseResult(r=r, t=tvec.ravel(), inliers=len(inl), reproj_err_px=reproj)
