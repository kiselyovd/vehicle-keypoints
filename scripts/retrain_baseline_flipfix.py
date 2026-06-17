"""Retrain the v1 CarFusion baseline with the CORRECTED left/right flip_idx.

v1 was trained with ultralytics' default fliplr=0.5 but an identity flip_idx in
data/processed/data.yaml, mirroring images without swapping L/R keypoints. The
yaml is now fixed; this re-runs the same recipe (yolo26n-pose, 30 ep, imgsz 480)
so the updated baseline can replace the model on the Hub if it beats v1.

batch=16 (not 32) keeps VRAM low enough to run alongside another training job.

  uv run python scripts/retrain_baseline_flipfix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ultralytics
from ultralytics import YOLO

REPO = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO / "scripts"))
from phase0_train import log, run_eval  # noqa: E402

RUN_DIR = REPO / "artifacts" / "baseline_flipfix"


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ultralytics.settings.update({"runs_dir": str(RUN_DIR).replace("\\", "/")})
    log("=== baseline retrain (corrected flip_idx) ===")
    model = YOLO(str(REPO / "yolo26n-pose.pt"))
    model.train(
        data="data/processed/data.yaml",
        epochs=30,
        imgsz=480,
        batch=16,
        workers=0,
        patience=8,
        project=str(RUN_DIR).replace("\\", "/"),
        name="sota_flipfix",
        cache="ram",  # RAM cache (64GB box) - no disk .npy, no lock race with the parallel v5
        verbose=True,
        plots=True,
    )
    best = RUN_DIR / "sota_flipfix" / "weights" / "best.pt"
    log(f"trained -> {best}")
    m = run_eval(best, REPO / "reports" / "baseline_flipfix_metrics.json")
    v1 = json.loads((REPO / "reports" / "metrics.json").read_text(encoding="utf-8"))
    delta = (m["oks_map"] - v1["oks_map"]) * 100
    log(
        f"BASELINE-FLIPFIX OKS-mAP {m['oks_map']:.4f} (v1 {v1['oks_map']:.4f}, "
        f"delta {delta:+.2f}pp) | PCK {m.get('pck_0.05', 0):.4f} (v1 {v1.get('pck_0.05', 0):.4f})"
    )


if __name__ == "__main__":
    main()
