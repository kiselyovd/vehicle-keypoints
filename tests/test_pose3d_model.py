"""Tests for the canonical rigid car model."""

from __future__ import annotations

import numpy as np

from vehicle_keypoints.pose3d.model import CARFUSION_14_CONFIG_KEYS, CanonicalCarModel


def test_model_has_14_points_in_canonical_order() -> None:
    m = CanonicalCarModel.load_default()
    assert m.points.shape == (14, 3)
    assert m.names == CARFUSION_14_CONFIG_KEYS
    assert len(m.names) == 14


def test_model_units_are_metres() -> None:
    m = CanonicalCarModel.load_default()
    # The car is roughly 4-5 m long; front wheel x ~ 1.52 m, rear wheel x ~ -1.45 m.
    span_x = m.points[:, 0].max() - m.points[:, 0].min()
    assert 3.0 < span_x < 6.0  # metres, not centimetres
    assert np.isclose(m.points[0, 0], 1.52, atol=0.01)  # Right_Front_wheel x


def test_skeleton_indices_in_range() -> None:
    m = CanonicalCarModel.load_default()
    for a, b in m.skeleton:
        assert 0 <= a < 14 and 0 <= b < 14
