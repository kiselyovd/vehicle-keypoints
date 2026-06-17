"""Tests for the 3D wireframe overlay projection."""

from __future__ import annotations

import cv2
import numpy as np

from vehicle_keypoints.pose3d.model import CanonicalCarModel
from vehicle_keypoints.pose3d.overlay3d import draw_pose3d, project_model


def _k() -> np.ndarray:
    return np.array([[2304.5, 0.0, 1686.2], [0.0, 2305.9, 1355.0], [0.0, 0.0, 1.0]])


def test_project_model_returns_pixel_coords() -> None:
    model = CanonicalCarModel.load_default()
    r = cv2.Rodrigues(np.array([0.0, 0.4, 0.0]))[0]
    t = np.array([0.0, 0.0, 20.0])
    pts2d = project_model(model.points, r, t, _k())
    assert pts2d.shape == (14, 2)
    # Car at 20 m on the optical axis projects near the principal point.
    assert np.all(np.abs(pts2d[:, 0] - 1686.2) < 1500)


def test_draw_pose3d_writes_image(tmp_path) -> None:
    model = CanonicalCarModel.load_default()
    img = np.zeros((2710, 3384, 3), dtype=np.uint8)
    src = tmp_path / "frame.png"
    cv2.imwrite(str(src), img)
    out = tmp_path / "overlay.png"
    r = cv2.Rodrigues(np.array([0.0, 0.4, 0.0]))[0]
    t = np.array([0.0, 0.0, 20.0])
    draw_pose3d(src, model, r, t, _k(), out)
    assert out.exists()
    assert cv2.imread(str(out)) is not None
