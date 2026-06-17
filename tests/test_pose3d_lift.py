"""Round-trip tests for the PnP lift."""

from __future__ import annotations

import cv2
import numpy as np

from vehicle_keypoints.pose3d.lift import solve_pose
from vehicle_keypoints.pose3d.metrics import geodesic_rotation_deg, translation_error_m
from vehicle_keypoints.pose3d.model import CanonicalCarModel


def _intrinsics() -> np.ndarray:
    return np.array([[2304.5, 0.0, 1686.2], [0.0, 2305.9, 1355.0], [0.0, 0.0, 1.0]])


def test_round_trip_recovers_pose() -> None:
    model = CanonicalCarModel.load_default()
    k = _intrinsics()
    rvec_true = np.array([0.1, -0.3, 1.2])
    r_true = cv2.Rodrigues(rvec_true)[0]
    t_true = np.array([1.5, 0.4, 18.0])
    img_pts, _ = cv2.projectPoints(model.points, rvec_true, t_true, k, None)
    img_pts = img_pts.reshape(-1, 2)
    vis = np.full(14, 2)  # all visible
    res = solve_pose(img_pts, vis, k, model.points)
    assert res is not None
    assert geodesic_rotation_deg(res.r, r_true) < 1.0
    assert translation_error_m(res.t, t_true) < 0.1


def test_round_trip_robust_to_noise_and_outliers() -> None:
    model = CanonicalCarModel.load_default()
    k = _intrinsics()
    rng = np.random.default_rng(0)
    rvec_true = np.array([0.0, 0.5, 0.0])
    r_true = cv2.Rodrigues(rvec_true)[0]
    t_true = np.array([0.2, 0.1, 22.0])
    img_pts, _ = cv2.projectPoints(model.points, rvec_true, t_true, k, None)
    img_pts = img_pts.reshape(-1, 2) + rng.normal(0, 1.5, (14, 2))
    img_pts[3] += 400.0  # gross outlier on one keypoint
    vis = np.full(14, 2)
    res = solve_pose(img_pts, vis, k, model.points)
    assert res is not None
    assert geodesic_rotation_deg(res.r, r_true) < 5.0


def test_too_few_points_returns_none() -> None:
    model = CanonicalCarModel.load_default()
    k = _intrinsics()
    img_pts = np.zeros((14, 2))
    vis = np.zeros(14, dtype=int)
    vis[:3] = 2  # only 3 visible -> not enough for PnP
    assert solve_pose(img_pts, vis, k, model.points) is None
