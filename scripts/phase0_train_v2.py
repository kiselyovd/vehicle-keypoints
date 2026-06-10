"""Phase 0 kill-switch experiment, CORRECTED design (v2).

Run #1 initialized stage 1 from base yolo26n-pose.pt, making the comparison
"964 frames from scratch vs v1 trained on the full CarFusion train set" -
unwinnable and uninformative (final OKS-mAP 0.0075).

The plan's actual design starts FROM THE V1 CHECKPOINT and asks whether synth
data lifts it by >= +2pp after an identical small real recalibration:

  arm A (treatment): v1 ckpt -> synth pretrain (864 UE5 frames) -> 100 real ft
  arm B (control):   v1 ckpt -------------------------------------> 100 real ft

Identical budgets except the synth stage; eval both on the full CarFusion test
set with the same evaluation module that produced the v1 baseline metrics.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase0_train import (  # noqa: E402
    _yolo_train,
    log,
    run_eval,
    stage2_real_finetune,
)

V1_CKPT = REPO_ROOT / "artifacts" / "sota" / "weights" / "best.pt"
RUN_DIR = REPO_ROOT / "artifacts" / "phase0_v2_runs"
REPORTS = REPO_ROOT / "reports"


def _report(v1: dict, arm_a: dict, arm_b: dict, synth_only: dict | None, out_path: Path) -> str:
    v1_oks = v1.get("oks_map", 0.0)
    a_oks = arm_a.get("oks_map", 0.0)
    b_oks = arm_b.get("oks_map", 0.0)
    threshold = v1_oks + 0.02

    if a_oks >= threshold:
        verdict = f"PASS (arm A {a_oks:.4f} >= v1+2pp {threshold:.4f})"
    elif a_oks >= v1_oks:
        verdict = f"MARGINAL (arm A {a_oks:.4f} in [v1, v1+2pp))"
    else:
        verdict = f"FAIL (arm A {a_oks:.4f} < v1 {v1_oks:.4f})"
    synth_delta = (a_oks - b_oks) * 100

    def row(label: str, m: dict) -> str:
        return (
            f"| {label} | {m.get('oks_map', 0.0):.4f} | {m.get('oks_map_50', 0.0):.4f} "
            f"| {m.get('pck_0.05', 0.0):.4f} | {(m.get('oks_map', 0.0) - v1_oks) * 100:+.2f}pp |"
        )

    rows = [row("v1 baseline (full real train)", v1)]
    if synth_only is not None:
        rows.append(row("arm A stage 1 only (v1 + synth, no real ft)", synth_only))
    rows.append(row("arm B control (v1 + 100 real, no synth)", arm_b))
    rows.append(row("arm A treatment (v1 + synth + 100 real)", arm_a))

    report = f"""# Phase 0 Kill-Switch Report (v2, corrected design)

**Date:** {date.today().isoformat()}
**Design:** both arms start from the v1 checkpoint; identical 100-real-frame
fine-tune; arm A additionally pretrains on 864 UE5 synth frames first.
**Kill switch:** arm A OKS-mAP >= v1 + 2pp ({threshold:.4f}).

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs v1 |
|---|---|---|---|---|
{chr(10).join(rows)}

**Synth contribution (arm A - arm B): {synth_delta:+.2f}pp OKS-mAP**

## Verdict

**{verdict}**
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    log(f"v2 report written to {out_path}")
    log(f"VERDICT v2: {verdict}")
    return verdict


def main() -> None:
    t0 = time.time()
    work = REPO_ROOT / "artifacts" / "phase0_work"
    synth_yaml = Path(
        os.environ.get("VK_SYNTH_YOLO_YAML", str(work / "synth_yolo" / "synth_data.yaml"))
    )
    real_yaml = Path(
        os.environ.get("VK_REAL100_YAML", str(work / "real_subset" / "real_subset_data.yaml"))
    )
    for p, name in ((V1_CKPT, "v1 checkpoint"), (synth_yaml, "synth yaml"), (real_yaml, "real-100 yaml")):
        if not p.exists():
            log(f"FATAL: {name} not found at {p}")
            raise SystemExit(2)

    v1_metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))

    # arm A stage 1: adapt v1 to synth (gentler LR than from-scratch, no freeze)
    log("=== v2 arm A stage 1: v1 ckpt + synth pretrain ===")
    a1_best = _yolo_train(
        init_model=str(V1_CKPT),
        data_yaml=synth_yaml,
        run_dir=RUN_DIR,
        name="v2_synth_on_v1",
        epochs=30,
        imgsz=480,
        batch=16,
        lr0=5e-4,
        patience=9,
    )
    # arm A stage 2: real recalibration
    a2_best = stage2_real_finetune(
        init_ckpt=a1_best, real_yaml=real_yaml, run_dir=RUN_DIR, name="v2_real_finetune"
    )
    # arm B control: v1 + same real recalibration, no synth
    b_best = stage2_real_finetune(
        init_ckpt=V1_CKPT, real_yaml=real_yaml, run_dir=RUN_DIR, name="v2_control_v1_plus_real"
    )

    log("=== v2 evals (full CarFusion test, methodology parity with v1) ===")
    synth_only_metrics = run_eval(a1_best, REPORTS / "phase0_v2_stage1_metrics.json")
    arm_a_metrics = run_eval(a2_best, REPORTS / "phase0_v2_armA_metrics.json")
    arm_b_metrics = run_eval(b_best, REPORTS / "phase0_v2_armB_metrics.json")

    _report(
        v1_metrics,
        arm_a_metrics,
        arm_b_metrics,
        synth_only_metrics,
        REPO_ROOT / "docs" / "phase0" / "kill_switch_report_v2.md",
    )
    log(f"Phase 0 v2 complete in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
