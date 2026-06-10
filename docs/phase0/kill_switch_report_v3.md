# Phase 0 Kill-Switch Report (v3: mixed training, multi-vehicle synth v2)

**Date:** 2026-06-11
**Design:** v1 checkpoint fine-tuned on ONE mixed dataset (816 synth v2 frames
with 9056 multi-vehicle instances + the same 100 real frames, seed 42), val on
real CarFusion val. Control (arm B) reused from v2: v1 + 100 real, no synth.
**Kill switch:** arm A OKS-mAP >= v1 + 2pp (0.2399).

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs v1 |
|---|---|---|---|---|
| v1 baseline (full real train) | 0.2199 | 0.3499 | 0.4958 | +0.00pp |
| arm B control (v1 + 100 real, no synth) | 0.2204 | 0.3502 | 0.4962 | +0.05pp |
| arm A v3 (v1 + mixed synth_v2+100real) | 0.0533 | 0.1657 | 0.2860 | -16.66pp |

**Synth contribution (arm A - arm B): -16.71pp OKS-mAP**

## Verdict

**FAIL (arm A 0.0533 < v1 0.2199)**
