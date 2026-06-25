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


def _crop_around(img: np.ndarray, xywh: tuple[float, float, float, float], pad: float) -> np.ndarray:
    """Crop the image to a padded box around a 2D detection (x, y, w, h)."""
    h_img, w_img = img.shape[:2]
    x, y, w, h = xywh
    m = pad * max(w, h)
    x0 = max(0, int(x - m))
    y0 = max(0, int(y - m))
    x1 = min(w_img, int(x + w + m))
    y1 = min(h_img, int(y + h + m))
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]


def draw_pose3d(
    image_path: str | Path,
    model: CanonicalCarModel,
    r: np.ndarray,
    t: np.ndarray,
    k: np.ndarray,
    out_path: str | Path,
    *,
    draw_box: bool = True,
    crop_xywh: tuple[float, float, float, float] | None = None,
    crop_pad: float = 0.35,
) -> None:
    """Draw the posed wireframe (and optionally the 3D bbox) on the image and save.

    Set ``draw_box=False`` for a clean wireframe-only overlay, and pass
    ``crop_xywh`` (the 2D detection box) to crop tightly around a single car.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    pts = project_model(model.points, r, t, k).astype(int)
    for a, b in model.skeleton:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), _WIRE_COLOR, 2, cv2.LINE_AA)
    if draw_box:
        box = project_model(_bbox_corners(model.points), r, t, k).astype(int)
        for a, b in _BOX_EDGES:
            cv2.line(img, tuple(box[a]), tuple(box[b]), _BOX_COLOR, 2, cv2.LINE_AA)
    if crop_xywh is not None:
        img = _crop_around(img, crop_xywh, crop_pad)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
