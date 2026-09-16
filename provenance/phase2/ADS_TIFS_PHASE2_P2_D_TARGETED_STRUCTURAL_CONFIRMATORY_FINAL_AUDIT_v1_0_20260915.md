# ADS–TIFS Phase-2 P2-D Targeted Structural Confirmatory Final Audit v1.0

**Date:** 2026-09-15  
**Submitted bundle:** `results_p2d_targeted_structural_confirmatory_v1_0.zip`  
**Bundle SHA-256:** `8a9d0d936071feb33de8172f59a926c09fb8fde00b323ec9a2609ab8213ccafc`  
**Result JSON SHA-256:** `9db0680f5a422e5a4df9515b09f375b0db26238936e93abe783ab20be046d8b1`  
**Summary SHA-256:** `51ca4fb4b8c3abd26929122fcdc412b16430e5fc334c9bda34535611a0f6e44a`  
**Execution package SHA-256:** `5531b6d8446bc4c5fd66257b92a08b2b9b872f1541736e97abe6e4c9d53f7c7e`  
**Package-manifest SHA-256:** `4feed37fba363605dde22461f3756ea1147f594846ffbd8d63bdcceb8c3e0f73`  
**Confirmatory runner SHA-256:** `3bb50d6393c5cd055ad9e9f1dc20bfb8ea122794cc1ef87b28ce4351dd0561b9`  
**Shared targeted core SHA-256:** `4ff8d7814231b473330d5cde9851cda3d4c7acb1febe2084164125b0c94b019e`  
**Merge runner SHA-256:** `edcc9a45eee5b0e9d1fee8c074c44f302de6feb182f37b40ea6d1bea58fbfacf`  
**Protocol-lock SHA-256:** `ae6285bba423fa0d63d894239ad285bac5dd08c76075e026174082f53c89f697`  
**Audit status:** `PASS — COMPLETE 432-STATE CONFIRMATORY TREE VERIFIED`  
**Scientific branch status:** `P2-D CLOSED / PARTIAL POSITIVE`

## 1. Completeness and provenance

The archive contains exactly `432` confirmatory state directories. Each state contains `cell.json`, `delta.pt`, reference/holdout target-KL NPZ, and reference/holdout canonical ADS NPZ.

All three shard results report exactly `144 / 144` completed states with the locked seed partition: shard 0 seeds `42, 789`; shard 1 seeds `123, 1011`; shard 2 seeds `456, 1213`. No `.__INPROGRESS__` directory remains.

Every one of the `432 / 432` cells independently passes the confirmatory-role, protocol, runner/package-manifest, deterministic-runtime, 20-step trajectory, gradient, projection, damage-arithmetic, helper-script, saved-delta, surface-topology, target-artifact, and canonical-ADS artifact gates. Learned surface is exactly `pos_embed [1,197,768]`; RoPE is exactly 24 per-block `cos_cached/sin_cached` tensors with `inv_freq` excluded.

The maximum difference between canonical layer means recomputed from serialized per-image ADS and cell JSON float64 means is `2.737e-09`, consistent with locked float32 per-image serialization.

## 2. Locked merge and selection

The merged result contains all `432` unique states: `2 PE x 6 seeds x 6 epsilon x 6 beta`.

Selection rule replayed exactly: retain directly observed states with full-validation damage `<=1.0 pp`; maximize reference target attainment; tie-break by lower damage, smaller epsilon, larger beta; no interpolation or extrapolation. All `12 / 12` PE x seed units are estimable.

## 3. Primary selected-state results

| PE | seed | epsilon | beta | damage pp | ref attainment | holdout attainment | ref ADS | holdout ADS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learned | 42 | 0.2 | 10 | 0.900 | 0.0881 | 0.0886 | 0.043639 | 0.043889 |
| learned | 123 | 0.2 | 10 | 0.500 | 0.0945 | 0.0937 | 0.048713 | 0.048870 |
| learned | 456 | 0.05 | 1 | 0.960 | 0.1085 | 0.1082 | 0.025582 | 0.025871 |
| learned | 789 | 0.2 | 10 | 0.480 | 0.0810 | 0.0793 | 0.034756 | 0.034976 |
| learned | 1011 | 0.05 | 1 | 0.660 | 0.0933 | 0.0921 | 0.021680 | 0.022078 |
| learned | 1213 | 0.02 | 0 | 0.860 | 0.0595 | 0.0587 | 0.009520 | 0.009613 |
| rope | 42 | 0.05 | 0.1 | 0.800 | 0.1094 | 0.1095 | 0.008407 | 0.008400 |
| rope | 123 | 0.1 | 1 | 0.720 | 0.1786 | 0.1793 | 0.023519 | 0.023690 |
| rope | 456 | 0.1 | 1 | 0.960 | 0.1856 | 0.1861 | 0.024224 | 0.024499 |
| rope | 789 | 0.1 | 1 | 0.700 | 0.1830 | 0.1834 | 0.022542 | 0.022634 |
| rope | 1011 | 0.1 | 1 | 0.900 | 0.1900 | 0.1908 | 0.025488 | 0.025635 |
| rope | 1213 | 0.2 | 10 | 0.820 | 0.1362 | 0.1369 | 0.018529 | 0.018620 |

Learned n=6: selected damage `0.727 ± 0.209 pp`; reference target attainment `0.0875 ± 0.0165`; holdout `0.0868 ± 0.0166`; reference whole-profile ADS `0.030648 ± 0.014591`; holdout `0.030883 ± 0.014591`.

RoPE n=6: selected damage `0.817 ± 0.101 pp`; reference target attainment `0.1638 ± 0.0331`; holdout `0.1643 ± 0.0332`; reference whole-profile ADS `0.020451 ± 0.006358`; holdout `0.020580 ± 0.006434`.

Absolute epsilon must not be interpreted as an intrinsic cross-family robustness coordinate.

## 4. Reference-to-holdout transfer

Learned: mean absolute target-attainment difference `0.000883`, maximum `0.001751`, mean absolute whole-ADS difference `0.000235`.

RoPE: mean absolute target-attainment difference `0.000515`, maximum `0.000715`, mean absolute whole-ADS difference `0.000131`.

This argues against a reference-cohort-only artifact for the selected states.

## 5. Prespecified target-attainment thresholds

At `<=1 pp` damage, `>=25%` target attainment on both reference and holdout is Learned `0/6`, RoPE `0/6`; `>=50%` is also `0/6` for both. P2-D therefore does **not** establish strong or near-complete structural redirection under the <=1-pp constraint. Supported wording is **modest partial target progress**.

## 6. ADS tracks an independent structural target

ADS is absent from the attack objective. Across the complete 36-state observed grid within each PE x seed, reference target attainment and canonical whole-profile ADS are strongly positively associated.

- Learned per-seed Spearman rho: mean `0.9522 ± 0.0062`, range `0.9470–0.9606`;
- RoPE per-seed Spearman rho: mean `0.9530 ± 0.0100`, range `0.9398–0.9701`.

This supports ADS as a forensic characterization / diagnostic metric. It does not establish detector specificity, defense, or adaptive-proof behavior.

## 7. Cross-layer observation

For selected <=1-pp states, reference ADS profile maxima are Learned L4 in `4/6`, L3 in `2/6`; RoPE L2 in `4/6`, L3 in `2/6`. This is P2-D-specific profile redistribution and reinforces that L4 is only an operational coordinate.

## 8. Scientific interpretation

All 12 checkpoint units have a directly observed state with `<=1 pp` full-validation damage and positive target progress. Best eligible target progress is modest: approximately 5.9–10.9% for Learned and 10.9–19.0% for RoPE, with nearly identical holdout transfer. Stronger 25% and 50% markers are not established. Despite ADS being excluded from the objective, target progress and canonical ADS have rho about 0.95 across the grid for every checkpoint.

P2-D supports the proposition that ADS is sensitive to and quantitatively characterizes an independently specified structural attention change under low task damage. It does not support large structural redirection at <=1 pp damage.

## 9. Final branch decision

`P2-D CLOSED / PARTIAL POSITIVE`

Manuscript-authorized: direct observed <=1-pp selected-state summaries; reference-to-holdout transfer; per-seed target-attainment/ADS association; negative 25%/50% markers; profile redistribution as descriptive evidence.

Not authorized: substantial/near-complete redirection wording; PE specificity against non-PE surfaces; fixed-epsilon robustness ranking; detector/defense/adaptive-proof claims.
