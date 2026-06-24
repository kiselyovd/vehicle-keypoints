"""Render a posed 3D wireframe and 3D bounding box onto an image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .model import CanonicalCarModel

_WIRE_COLOR = (0, 255, 0)  # green (BGR)
_BOX_COLOR = (0, 200, 255)  # amber


def project_model(points_3d: np.ndarray, r: np.ndarray, t: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Project model-frame 3D points to pixels via X_cam = r @ X + t."""
    rvec = cv2.Rodrigues(np.ascontiguousarray(r, dtype=np.float64))[0]
    proj, _ = cv2.projectPoints(
        np.ascontiguousarray(points_3d, dtype=np.float64),
        rvec,
        np.ascontiguousarray(t, dtype=np.float64).reshape(3, 1),
        k,
        None,
    )
    return proj.reshape(-1, 2)


def _bbox_corners(points_3d: np.ndarray) -> np.ndarray:
    lo = points_3d.min(axis=0)
    hi = points_3d.max(axis=0)
    return np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
    )


_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def draw_pose3d(
    image_path: str | Path,
    model: CanonicalCarModel,
    r: np.ndarray,
    t: np.ndarray,
    k: np.ndarray,
    out_path: str | Path,
) -> None:
    """Draw the posed wireframe + 3D bbox on the image and save."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    pts = project_model(model.points, r, t, k).astype(int)
    for a, b in model.skeleton:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), _WIRE_COLOR, 2)
    box = project_model(_bbox_corners(model.points), r, t, k).astype(int)
    for a, b in _BOX_EDGES:
        cv2.line(img, tuple(box[a]), tuple(box[b]), _BOX_COLOR, 2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
