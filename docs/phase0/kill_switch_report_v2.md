# Phase 0 Kill-Switch Report (v2, corrected design)

**Date:** 2026-06-10
**Design:** both arms start from the v1 checkpoint; identical 100-real-frame
fine-tune; arm A additionally pretrains on 864 UE5 synth frames first.
**Kill switch:** arm A OKS-mAP >= v1 + 2pp (0.2399).

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs v1 |
|---|---|---|---|---|
| v1 baseline (full real train) | 0.2199 | 0.3499 | 0.4958 | +0.00pp |
| arm A stage 1 only (v1 + synth, no real ft) | 0.0020 | 0.0036 | 0.0698 | -21.79pp |
| arm B control (v1 + 100 real, no synth) | 0.2204 | 0.3502 | 0.4962 | +0.05pp |
| arm A treatment (v1 + synth + 100 real) | 0.0840 | 0.1916 | 0.3314 | -13.59pp |

**Synth contribution (arm A - arm B): -13.64pp OKS-mAP**

## Verdict

**FAIL (arm A 0.0840 < v1 0.2199)**
