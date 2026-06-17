"""Evaluation matching + metric aggregation for the Apollo 3D baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .apollo import ApolloCar
from .frame_align import aligned_gt_rotation
from .metrics import geodesic_rotation_deg, translation_error_m


@dataclass(frozen=True)
class PredPose:
    """One predicted car: 2D detection centre + recovered pose."""

    center_px: tuple[float, float]
    r: np.ndarray
    t: np.ndarray


def match_by_center(
    preds: list[PredPose],
    gts: list[ApolloCar],
    gt_centers_px: list[tuple[float, float]],
    max_px: float,
) -> list[tuple[int, int]]:
    """Greedy nearest-centre matching pred->gt within max_px. Returns (pred_i, gt_i)."""
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for pi, p in enumerate(preds):
        best_j, best_d = -1, max_px
        for gj, c in enumerate(gt_centers_px):
            if gj in used:
                continue
            d = float(np.hypot(p.center_px[0] - c[0], p.center_px[1] - c[1]))
            if d < best_d:
                best_j, best_d = gj, d
        if best_j >= 0:
            used.add(best_j)
            pairs.append((pi, best_j))
    return pairs


def pose_error_row(pred: PredPose, gt: ApolloCar) -> dict[str, float]:
    """Per-match error row using frame-aligned GT rotation."""
    rot = geodesic_rotation_deg(pred.r, aligned_gt_rotation(gt.r))
    trans = translation_error_m(pred.t, gt.t)
    depth = float(np.linalg.norm(gt.t)) or 1.0
    return {"rot_deg": rot, "trans_m": trans, "rel_trans": trans / depth}


def aggregate_metrics(
    rows: list[dict[str, float]], n_gt: int, n_detected: int
) -> dict[str, float]:
    """Aggregate per-match rows into summary metrics."""
    rot = np.array([r["rot_deg"] for r in rows]) if rows else np.array([0.0])
    trans = np.array([r["trans_m"] for r in rows]) if rows else np.array([0.0])
    return {
        "n_matched": float(len(rows)),
        "rot_deg_mean": float(rot.mean()),
        "rot_deg_median": float(np.median(rot)),
        "trans_m_mean": float(trans.mean()),
        "trans_m_median": float(np.median(trans)),
        "acc_rot_10deg": float((rot < 10.0).mean()) if rows else 0.0,
        "acc_trans_2m": float((trans < 2.0).mean()) if rows else 0.0,
        "detector_hit_rate": float(n_detected / n_gt) if n_gt else 0.0,
    }
