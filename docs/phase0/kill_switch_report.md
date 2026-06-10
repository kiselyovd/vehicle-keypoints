# Phase 0 Kill-Switch Report

**Date:** 2026-06-10
**Experiment:** Synth-pretrain (864 UE5 frames, 14-pt) then real fine-tune (100 CarFusion frames)
**Kill-switch threshold:** OKS-mAP >= 0.2399 (v1 baseline + 2pp)

## Data

- Synth pretrain: 864 frames (UE5 City Sample, 14 canonical CarFusion keypoints)
- Real fine-tune / control: 100 CarFusion train frames (random seed 42 subset)
- Eval: 12,761 CarFusion test images (identical to v1 baseline run)

## Metrics

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | Delta vs v1 |
|---|---|---|---|---|
| v1 baseline (YOLO26n-pose, full train) | 0.2199 (22.0%) | 0.3499 (35.0%) | 0.4958 (49.6%) | - |
| Phase 0 stage 1 (synth-only, no real ft) | 0.0004 (0.0%) | 0.0008 (0.1%) | 0.0435 (4.4%) | -21.95pp |
| Control (base -> 100 real, no synth) | 0.0000 (0.0%) | 0.0000 (0.0%) | 0.0800 (8.0%) | -21.99pp |
| Phase 0 final (synth pretrain -> 100 real ft) | 0.0075 (0.7%) | 0.0191 (1.9%) | 0.1487 (14.9%) | -21.24pp |

## Delta summary (Phase 0 final vs v1)

- OKS-mAP: -21.24pp
- OKS-mAP50: -33.08pp
- PCK@0.05: -34.71pp

## Verdict

**FAIL (regression: -21.24pp vs v1)**
