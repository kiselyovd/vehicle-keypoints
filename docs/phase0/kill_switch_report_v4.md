# Phase 0 Kill-Switch Report (v4: mixed training, wide multi-venue MRQ synth v4, real x8)

**Date:** 2026-06-17
**Design:** v1 checkpoint fine-tuned on ONE mixed dataset (synth v4 wide multi-venue MRQ
frames + the same 100 real frames oversampled x8, seed 42), val on real CarFusion val.
Control (arm B) reused from v2: v1 + 100 real, no synth.
**Kill switch:** arm A OKS-mAP >= v1 + 2pp (0.2399).

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs v1 |
|---|---|---|---|---|
| v1 baseline (full real train) | 0.2199 | 0.3499 | 0.4958 | +0.00pp |
| arm B control (v1 + 100 real, no synth) | 0.2204 | 0.3502 | 0.4962 | +0.05pp |
| arm A v4 (v1 + mixed synth_v4+100real x8) | 0.1494 | 0.2762 | 0.4130 | -7.05pp |

**Synth contribution (arm A - arm B): -7.10pp OKS-mAP**

## Verdict

**FAIL (arm A 0.1494 < v1 0.2199)**
