# ADS TIFS — Canonical Adaptive n=6 Confirmatory v2.7.2 Final External Audit

**Date:** 2026-09-10  
**Protocol:** `LOCK-003 / ADS_TIFS_CANONICAL_ADAPTIVE_PROTOCOL_LOCK_v2_7_20260910`  
**Execution:** `ADS_TIFS_CANONICAL_ADAPTIVE_N6_CONFIRMATORY_v2_7_2_20260910`  
**Decision:** `PASS / COMPLETE / CANONICAL`

## Source and provenance

RESULTS ZIP SHA-256: `3da48d5c97c9851ecafc0e43ada060139294e281beac920dbb21b4a6aacd32f0`  
Sidecar match: **PASS**  
Merged result JSON SHA-256: `fd25951e4e6fbaaa136c67802b0fef36866be476d16dd9756adb4bedd2841619`  
Internal audit SHA-256: `1c626ea0fcefdcc3e9b5be6814dbe1578f80d6cb0e72f6806cd817f6b61efab0`  
Final archive manifest SHA-256: `d77f1e29b620e93f58b5aba144eeb24c03186b6d7e05ca5a607016ac67cb97f6`

Canonical identities verified:
- operator `093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`
- LOCK-003 `1b558a44b947a0c3e74525e4acfc713c49f4757e13b5bd622512e3d11c4f251f`
- reference indices `1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509`
- holdout indices `835af1a98d05ac4f99cbb697f36d456b0a24bdd6905ada0ebac5da4eec0916a7`
- transformed reference cache `01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4`
- transformed full-validation cache `90d47ea5212f5d06b9e8d12fa80ccb20cc722a46d79d2c926ef41be2917f0f71`
- source EXP-014 RESULTS ZIP `ac40ded10f58d02d7b2b0eab7627f58b26c29569b330364a89d94aac31e0988b`
- source EXP-014 aggregate `dcd2fa22d27a9a983dfadca3d49b56f776d316add01d201a2fe5563d757203d1`
- PASS pilot final audit `1188235ae3016aa755f087f07f81f92c59dec69ab087e10be944782c2514d6f0`

## Completeness / integrity

Independent fail-closed checks:
- 672/672 locked logical rows
- 588/588 positive-lambda states
- 84/84 lambda=0 logical rows
- 48/48 unique EXP-014 source controls
- 2711/2711 final archive-manifest payloads independently rehashed
- 2544/2544 shard-manifest payloads independently rehashed
- 588/588 positive delta files
- 1272/1272 per-image canonical NPZ artifacts, all `[256,12]`, float32 storage, finite, correct operator hash
- 7350/7350 positive delta tensors finite, float32, correct topology
- Learned surface = `pos_embed`
- RoPE surface = 12 `cos_cached` + 12 `sin_cached`; `inv_freq` excluded
- 144/144 embedded EXP-014 source files byte-identical to the original canonical primary RESULTS ZIP
- max float32 projection overage `2.980232227667301004e-09` < `1e-7`
- max NPZ-mean vs cell-summary discrepancy `2.532768661467343918e-09`
- internal objective↔measurement max difference `6.661338147750939243e-16` ≤ `1e-9`

All four shard logs contain their expected PASS markers and zero `TRACEBACK`, `ERROR`, `FAIL`, or `WARNING`.

Runtime is uniform across all 12 seed executions: NVIDIA H200, PyTorch `2.8.0a0+5228986c39.nv25.06`, CUDA 12.9, deterministic algorithms enabled, TF32 disabled, math-only SDPA, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Confirmatory findings

The normalized adaptive target is always relative to the exact same-PE/seed/epsilon lambda=0 EXP-014 control.

### Prespecified descriptive damage/suppression markers

At damage ≥5 pp with target ≤50% of lambda=0 on **both** reference and holdout, at least one observed state exists for:
- Learned L4: 6/6 seeds
- Learned profile-LSE: 6/6
- RoPE L4: 6/6
- RoPE profile-LSE: 6/6

Total: **24/24 PE×objective×seed cases**.

At damage ≥10 pp under the same criterion:
- Learned L4: 6/6
- Learned profile-LSE: 6/6
- RoPE L4: 4/6
- RoPE profile-LSE: 5/6

Total: **21/24**.

These are descriptive operating-point counts, not detector thresholds.

### Reference → holdout transfer

Across all 588 positive-lambda states:
- mean signed holdout-minus-reference target-ratio difference `0.000020`
- mean absolute difference `0.001617`
- median absolute difference `0.001003`
- 95th percentile `0.004873`
- maximum `0.012291`
- descriptive Pearson r `0.999964`
- 50%-suppression marker concordance **586/588**

Within this locked reference/holdout pair, suppression is therefore not merely an evaluation artifact of the optimization-reference cohort. This is not a universal transfer claim.

### L4-targeted cross-layer displacement

Among 336 positive L4-targeted states, L4 is the most strongly suppressed layer-relative-to-lambda0 in:
- Learned **158/168**
- RoPE **168/168**
- combined **326/336**

Among material-L4-suppression states:
- Learned: 115 states; median whole-profile mean-ADS ratio `0.340`; 16/115 have at least one non-L4 layer above lambda0.
- RoPE: 115 states; median whole-profile mean-ADS ratio `0.651`; 0/115 have a non-L4 layer above lambda0.

This supports layer-selective suppression and ADS-profile redistribution / cross-layer displacement. L4 remains an operational coordinate, not a privileged universal layer.

### Full-profile adaptive branch

Among profile-LSE states satisfying target ≤50% of lambda0 and damage ≥5 pp:
- Learned: **48/49** qualifying observed states have all 12 layers below lambda0.
- RoPE: **32/32**.

Thus the direct profile-aware objective can broadly suppress the canonical ADS profile rather than only L4.

## Interpretation

**Observed result:** canonical ADS-aware reoptimization produces a clear damage–stealth tradeoff. Both targeted objectives can substantially reduce their canonical ADS target while retaining classification damage, with close suppression transfer to the locked disjoint holdout.

**Interpretation:** this is an adaptive limitation of ADS as a diagnostic/triage signal. ADS remains useful for forensic characterization, but must not be described as an adaptive-proof detector or a defense.

**Cross-layer inference:** L4-only monitoring can be selectively manipulated and may redistribute the profile. Whole-profile reporting gives richer forensic characterization, but direct profile-aware optimization can also suppress that profile.

**Scope:** Learned PE and RoPE only; one fixed reference cohort and one fixed disjoint holdout; locked objectives and observed grid only; no interpolation/extrapolation; no fixed-epsilon intrinsic cross-family robustness ranking.

## Historical status and branch closure

Historical adaptive/reference-evasion/profile-aware positive-lambda numerical results remain **RETIRED FOR FINAL CLAIMS**. EXP-021 is their canonical numerical replacement.

`ADAPTIVE BRANCH: COMPLETE / CLOSED AFTER EXP-021`

No additional adaptive optimizer/lambda/epsilon ladder is required absent a new explicit methodological question. Next phase is manuscript correction and full claim/number traceability integration from the canonical ledger.
