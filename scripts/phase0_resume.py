"""Resume Phase 0 after stage-2 interruption (epoch 16/20).

The original phase0_train.py run was killed mid-validation of stage-2 epoch 17.
This script resumes stage 2 from artifacts/phase0_runs/real_finetune/weights/last.pt
and then completes the remaining pipeline steps:

  1. Resume stage 2 (ultralytics resume=True, finishes epochs 17-20)
  2. Copy best.pt -> artifacts/phase0/main.pt and run the test-set eval
  3. Control run: base yolo26n-pose.pt -> 100 real frames (no synth)
  4. Control eval
  5. Kill-switch report (docs/phase0/kill_switch_report.md)

Usage:
    .venv/Scripts/python.exe -u scripts/phase0_resume.py >> logs/phase0.log 2>&1
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase0_train import (  # noqa: E402
    control_real_finetune,
    log,
    run_eval,
    write_kill_switch_report,
)


def resume_stage2(last_ckpt: Path) -> Path:
    """Resume the interrupted stage-2 run; returns best.pt path."""
    import ultralytics
    from ultralytics import YOLO

    run_dir = REPO_ROOT / "artifacts" / "phase0_runs"
    ultralytics.settings.update({"runs_dir": str(run_dir).replace("\\", "/")})

    log(f"=== Resuming stage 2 from {last_ckpt} (epochs 17-20) ===")
    model = YOLO(str(last_ckpt))
    results = model.train(resume=True)
    best = Path(str(results.save_dir)) / "weights" / "best.pt"
    log(f"  best.pt = {best}")
    return best


def main() -> None:
    t0 = time.time()
    artifacts = REPO_ROOT / "artifacts"
    run_dir = artifacts / "phase0_runs"

    last_ckpt = run_dir / "real_finetune" / "weights" / "last.pt"
    if not last_ckpt.is_file():
        raise SystemExit(f"missing {last_ckpt}; nothing to resume")

    # --- 5 (cont). Resume stage 2 ---
    final_ckpt = resume_stage2(last_ckpt)
    final_ckpt_copy = artifacts / "phase0" / "main.pt"
    shutil.copy2(final_ckpt, final_ckpt_copy)
    log(f"Phase 0 final checkpoint saved to {final_ckpt_copy}")

    # --- 6. Eval final checkpoint (synth + real) ---
    log("=== Eval Phase 0 final checkpoint ===")
    phase0_eval_out = REPO_ROOT / "reports" / "phase0_main_metrics.json"
    phase0_metrics = run_eval(final_ckpt_copy, phase0_eval_out)

    # --- 7. Control: base -> 100 real (no synth) ---
    real_yaml = artifacts / "phase0_work" / "real_subset" / "real_subset_data.yaml"
    if not real_yaml.is_file():
        # fall back: locate the yaml written by build_real_subset_yaml
        candidates = list((artifacts / "phase0_work" / "real_subset").glob("*.yaml"))
        if not candidates:
            raise SystemExit("real subset yaml not found")
        real_yaml = candidates[0]
    control_ckpt = control_real_finetune(real_yaml, run_dir, epochs=20, imgsz=480, batch=16)
    control_ckpt_copy = artifacts / "phase0" / "control.pt"
    shutil.copy2(control_ckpt, control_ckpt_copy)
    log(f"Control checkpoint saved to {control_ckpt_copy}")

    log("=== Eval control checkpoint ===")
    control_eval_out = REPO_ROOT / "reports" / "phase0_control_metrics.json"
    try:
        control_metrics = run_eval(control_ckpt_copy, control_eval_out)
    except Exception as exc:
        log(f"WARNING: control eval failed: {exc}")
        control_metrics = None

    # --- 8. Load v1 baseline + stage-1 synth metrics ---
    v1_metrics = json.loads((REPO_ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    synth_json = REPO_ROOT / "reports" / "phase0_synth_metrics.json"
    synth_metrics = (
        json.loads(synth_json.read_text(encoding="utf-8")) if synth_json.is_file() else None
    )

    # --- 9. Kill-switch report ---
    report_path = REPO_ROOT / "docs" / "phase0" / "kill_switch_report.md"
    write_kill_switch_report(
        v1_metrics, synth_metrics, phase0_metrics, control_metrics, report_path
    )

    log(f"Phase 0 complete in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
