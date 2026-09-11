# Corrected Comparison + ROC Calibration Protocol Lock v2.3

**Status:** `LOCKED_FOR_N6_EXECUTION`  
**Lock JSON SHA-256:** `cfb1aa10a3cbff38f6b07ac3a3fc57e5e5d7fe2493a56828658e2d69b16e2f70`

## Scientific scope
- ImageNet-100, fixed 256-image reference set.
- PE = Learned, Sinusoidal, RoPE, ALiBi.
- Seeds = `[42,123,456,789,1011,1213]`.
- Common absolute epsilon = `{.001,.002,.005,.01,.02,.05,.1,.2}`.
- 192 attack states, all reused from canonical primary full-grid v2.1; **no new PGD**.

## Metric lock
- ADS(L4): canonical logit-domain mean(per-image KL), global operator hash unchanged.
- Attn-L2: corrected Layer-4 key-salience `mean_query`, flattened `[12,197]`.
- Diagonal Mahalanobis: same feature, clean per-coordinate variance normalization with pilot-fixed lambda rule.
- LogitKL: same-image per-image class-distribution KL in float64.

## Benign calibration lock
Identity plus nine transformed same-image conditions: JPEG q 50/30/10; Gaussian blur radius 1/2/3; pixel-space Gaussian noise sigma .05/.10/.20. Noise is deterministic and condition/image keyed. Historical normalized-space noise is retired.

## Primary AUC lock
Per PE x seed x epsilon x method: 256 attacked positives vs 2560 negatives (identity + nine benign conditions, each with 256 same-image scores); standard rank AUC with 0.5 tie handling. Report seed-level mean +/- sample SD. A benign-only AUC and paired same-image concordance are secondary diagnostics.

## High-confidence boundary
Smallest tested epsilon with AUC >= .99 in **all 6 seeds**. No post-hoc relaxation and no extrapolation.

## Explicitly retired from final numerical use
Historical ROC attack scores, historical `KL(mean||mean)` ROC ADS, 20 cross-image half-split clean scores, historical normalized-tensor Gaussian noise, threshold-grid trapezoidal AUC as primary, and all old comparison-table numerical values.
