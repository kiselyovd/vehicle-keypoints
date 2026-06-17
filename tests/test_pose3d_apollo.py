"""Tests for the ApolloCar3D loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vehicle_keypoints.pose3d.apollo import (
    load_frame_cars,
    load_intrinsics,
    pose6_to_rt,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "apollo"


def test_load_intrinsics() -> None:
    k = load_intrinsics(FIX / "camera" / "5.cam")
    assert k.shape == (3, 3)
    assert np.isclose(k[0, 0], 2304.54786556982)
    assert np.isclose(k[1, 2], 1354.98486439791)
    assert k[2, 2] == 1.0


def test_pose6_to_rt_identity() -> None:
    r, t = pose6_to_rt([0.0, 0.0, 0.0, 1.0, 2.0, 20.0])
    assert np.allclose(r, np.eye(3), atol=1e-9)
    assert np.allclose(t, [1.0, 2.0, 20.0])


def test_pose6_to_rt_is_rotation() -> None:
    r, _ = pose6_to_rt([0.1, 0.5, -3.0, 3.0, 1.0, 15.0])
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-6)


def test_load_frame_cars() -> None:
    cars = load_frame_cars(FIX / "car_poses" / "frame_Camera_5.json")
    assert len(cars) == 2
    assert cars[0].car_id == 0
    assert cars[0].r.shape == (3, 3)
    assert cars[0].t.shape == (3,)
