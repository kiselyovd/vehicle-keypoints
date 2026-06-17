"""Tests for 6DoF pose error metrics."""

from __future__ import annotations

import cv2
import numpy as np

from vehicle_keypoints.pose3d.metrics import (
    geodesic_rotation_deg,
    translation_error_m,
)


def _rot_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    return cv2.Rodrigues(np.array([0.0, 0.0, rad]))[0]


def test_geodesic_identity_is_zero() -> None:
    r = _rot_z(0.0)
    assert geodesic_rotation_deg(r, r) < 1e-6


def test_geodesic_90_degrees() -> None:
    assert abs(geodesic_rotation_deg(_rot_z(0.0), _rot_z(90.0)) - 90.0) < 1e-4


def test_geodesic_symmetric_and_wraps() -> None:
    a, b = _rot_z(10.0), _rot_z(350.0)
    # shortest angle between 10deg and 350deg about z is 20deg
    assert abs(geodesic_rotation_deg(a, b) - 20.0) < 1e-4


def test_translation_error() -> None:
    t_pred = np.array([1.0, 2.0, 3.0])
    t_gt = np.array([1.0, 2.0, 7.0])
    assert abs(translation_error_m(t_pred, t_gt) - 4.0) < 1e-9
