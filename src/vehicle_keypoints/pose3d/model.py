"""Canonical rigid 3D car model (14 CarFusion keypoints), metres.

Frame: actor-local, x=forward, y=right, z=up, origin at car centre on ground.
Source: vendored from the UE5 City Sample vehicle13 24-keypoint config; the
first 14 keys are the CarFusion canonical keypoints in canonical index order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The 14 config keys whose order matches CarFusion canonical indices 0..13
# (same order as inference.overlay.CARFUSION_KEYPOINT_NAMES).
CARFUSION_14_CONFIG_KEYS: tuple[str, ...] = (
    "Right_Front_wheel",
    "Left_Front_wheel",
    "Right_Back_wheel",
    "Left_Back_wheel",
    "Right_Front_HeadLight",
    "Left_Front_HeadLight",
    "Right_Back_HeadLight",
    "Left_Back_HeadLight",
    "Exhaust",
    "Right_Front_Top",
    "Left_Front_Top",
    "Right_Back_Top",
    "Left_Back_Top",
    "Center",
)

# CarFusion canonical 18-edge skeleton (indices 0..13).
CARFUSION_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 2), (1, 3), (0, 1), (2, 3),
    (9, 11), (10, 12), (9, 10), (11, 12),
    (4, 0), (4, 9), (4, 5),
    (5, 1), (5, 10),
    (6, 2), (6, 11),
    (7, 3), (7, 12),
    (6, 7),
)

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "pose3d"
    / "citysample_vehCar_vehicle13.json"
)


@dataclass(frozen=True)
class CanonicalCarModel:
    """Rigid 14-point car wireframe in metres, canonical index order."""

    points: np.ndarray  # (14, 3) float64
    names: tuple[str, ...]
    skeleton: tuple[tuple[int, int], ...]

    @classmethod
    def load_default(cls) -> CanonicalCarModel:
        """Load the vendored vehicle13 canonical model."""
        return cls.from_config(_DEFAULT_CONFIG)

    @classmethod
    def from_config(cls, path: str | Path) -> CanonicalCarModel:
        """Load 14 canonical keypoints (metres) from a 24-point vehicle config."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kpts = data["keypoints"]
        pts = np.array(
            [kpts[name] for name in CARFUSION_14_CONFIG_KEYS], dtype=np.float64
        )
        pts /= 100.0  # centimetres -> metres
        return cls(points=pts, names=CARFUSION_14_CONFIG_KEYS, skeleton=CARFUSION_SKELETON)
