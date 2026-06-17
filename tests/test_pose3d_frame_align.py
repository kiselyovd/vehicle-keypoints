"""Tests for the model-frame alignment helpers."""

from __future__ import annotations

import numpy as np

from vehicle_keypoints.pose3d.frame_align import MODEL_TO_APOLLO, aligned_gt_rotation


def test_alignment_is_a_valid_rotation() -> None:
    assert MODEL_TO_APOLLO.shape == (3, 3)
    assert np.allclose(MODEL_TO_APOLLO @ MODEL_TO_APOLLO.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(MODEL_TO_APOLLO), 1.0, atol=1e-9)


def test_aligned_gt_rotation_applies_a_on_the_right() -> None:
    r_gt = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    out = aligned_gt_rotation(r_gt)
    assert np.allclose(out, r_gt @ MODEL_TO_APOLLO)
