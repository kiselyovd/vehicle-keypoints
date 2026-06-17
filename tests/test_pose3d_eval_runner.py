"""Tests for evaluation matching + aggregation (no model, no I/O)."""

from __future__ import annotations

import numpy as np

from vehicle_keypoints.pose3d.apollo import ApolloCar
from vehicle_keypoints.pose3d.eval_runner import (
    PredPose,
    aggregate_metrics,
    match_by_center,
)


def test_match_by_center_pairs_nearest() -> None:
    gts = [
        ApolloCar(car_id=0, r=np.eye(3), t=np.array([0.0, 0.0, 10.0])),
        ApolloCar(car_id=1, r=np.eye(3), t=np.array([0.0, 0.0, 30.0])),
    ]
    preds = [
        PredPose(center_px=(100.0, 100.0), r=np.eye(3), t=np.array([0.0, 0.0, 31.0])),
    ]
    gt_centers = [(100.0, 100.0), (500.0, 100.0)]
    pairs = match_by_center(preds, gts, gt_centers, max_px=50.0)
    assert pairs == [(0, 0)]  # pred 0 matched gt index 0


def test_aggregate_metrics_means() -> None:
    rows = [
        {"rot_deg": 10.0, "trans_m": 1.0, "rel_trans": 0.05},
        {"rot_deg": 20.0, "trans_m": 3.0, "rel_trans": 0.15},
    ]
    out = aggregate_metrics(rows, n_gt=4, n_detected=3)
    assert np.isclose(out["rot_deg_mean"], 15.0)
    assert np.isclose(out["rot_deg_median"], 15.0)
    assert np.isclose(out["trans_m_mean"], 2.0)
    assert np.isclose(out["detector_hit_rate"], 3 / 4)
    assert out["n_matched"] == 2
