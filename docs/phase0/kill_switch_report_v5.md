# Phase 0 Kill-Switch Report (v5: corrected flip-aug, fresh two-arm ablation)

**Date:** 2026-06-17
**Fix:** horizontal-flip augmentation now uses the correct L/R `flip_idx`
([1, 0, 3, 2, 5, 4, 7, 6, 8, 10, 9, 12, 11, 13]); earlier runs mirrored images with an identity flip_idx,
corrupting keypoints on ~half the augmented samples. Both arms re-run fresh from
the v1 checkpoint with identical settings; the only difference is the synthetic data.
**Kill switch:** arm A OKS-mAP >= v1 + 2pp (0.2399).

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs v1 |
|---|---|---|---|---|
| v1 baseline (full real train) | 0.2199 | 0.3499 | 0.4958 | +0.00pp |
| arm B (v1 + 100 real x8, no synth) | 0.3808 | 0.6254 | 0.6659 | +16.09pp |
| arm A (v1 + synth_v4 + 100 real x8) | 0.3575 | 0.6476 | 0.6165 | +13.76pp |

**Synth contribution (arm A - arm B): -2.33pp OKS-mAP**

## Verdict

**PASS (arm A 0.3575 >= v1+2pp 0.2399)**
