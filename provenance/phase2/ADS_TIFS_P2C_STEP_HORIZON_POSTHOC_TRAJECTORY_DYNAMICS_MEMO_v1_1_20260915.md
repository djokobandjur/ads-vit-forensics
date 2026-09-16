# ADS–TIFS P2-C Step-Horizon Post-Hoc Trajectory Dynamics Memo v1.0

**Date:** 2026-09-15  
**Role:** `POST-HOC / EXPLORATORY DIAGNOSTIC USING EXISTING LOCK-008 TRAJECTORY RECORDS`  
**New optimization runs:** `NONE`  
**Source result bundle SHA-256:** `b7b59d61d17aaa381f840559d1c0638c434ec0e6fb83276045389707b61f6bb1`

## Scope

This memo audits a secondary trajectory-level interpretation proposed after LOCK-008 closed. It does not change the locked result:

- DeiT: `FIXED_STEP_TRAJECTORY_NOT_CONVERGED_BY_200`
- RoPE: `FIXED_STEP_TRAJECTORY_NOT_CONVERGED_BY_200`

The analysis uses only the already-saved 200-step `reference_mean_ce_preupdate`, `grad_linf`, and `delta_linf_after_step` records.

## Recomputed trajectory statistics

| quantity | DeiT | RoPE |
|---|---:|---:|
| Mean total CE gain, step 1→200 | 8.910 | 5.599 |
| Mean fraction of total CE gain occurring after step 100 | 13.1% | 3.3% |
| Mean downward CE transitions among steps 101→200 | 47.25/99 | 47.50/99 |
| Mean late CE oscillation ratio, TV / |net drift| | 7.88 | 16.59 |
| Mean `grad_linf`, step 1 | 0.020575 | 0.182091 |
| Mean `grad_linf`, step 200 | 0.071092 | 29.988060 |
| First step at which `delta_linf` reaches epsilon | 10 in all 4 | 10 in all 4 |

After first contact, `delta_linf` remains equal to epsilon to float32 numerical tolerance (maximum absolute deviation: DeiT `4.172e-08`, RoPE `9.537e-09`).

## What the records support

The late objective dynamics are strongly oscillatory rather than monotone. For RoPE, only 3.3% of the total net CE gain occurs after step 100, while the objective decreases on an average of 47.50 of 99 late transitions and the late total variation is 16.6 times the net late drift. For DeiT the same qualitative pattern is present, but residual net drift is larger: 13.1% of total CE gain occurs after step 100.

This supports: `OSCILLATORY_FIXED_STEP_BOUNDARY_DYNAMICS / ITERATE_NONCONTRACTION`.

It is consistent with overshoot from a constant sign-PGD step that is too coarse for late-stage refinement, especially for RoPE.

## What the records do not establish

The records do **not** causally prove that the constant step size is the unique cause. No smaller-alpha or decayed-step trajectory was executed; full delta states were saved only at 20/50/100/200; a literal periodic state-space limit cycle was not demonstrated; and nonconvex/multibasin geometry cannot be uniquely separated from coarse fixed-step dynamics from these scalar traces alone.

Manuscript wording should therefore avoid `the fixed step induces a limit cycle` and instead use: `the fixed-step schedule exhibits persistent oscillatory boundary-constrained dynamics consistent with overshoot and remains noncontractive through 200 steps`.

## Objective/evaluation-population mismatch

The public reference bridge contains 256 images spanning 92 distinct ImageNet-1K class indices, whereas full-validation damage is measured over the complete 1000-class ImageNet-1K validation set. This mismatch is real but the present data do not prove that it causes the reference-objective/full-validation-damage divergence.

## Recommendation

For the manuscript, stop here. A smaller-alpha/decay experiment is unnecessary unless a stronger causal statement about step size is desired.

---

## v1.1 addendum — snapshot-state displacement geometry

**Status:** `POST-HOC / EXPLORATORY / NO NEW EXECUTION`  
**Source artifacts:** shipped step-50, step-100 and step-200 delta snapshots from all eight LOCK-008 trajectories.

| quantity | DeiT | RoPE |
|---|---:|---:|
| mean `cos(u,v)` | -0.0644 | -0.0621 |
| mean straightness | 0.6851 | 0.6863 |
| mean sign agreement, all coordinates | 0.8323 | 0.8748 |
| mean sign agreement, jointly nonzero coordinates | 0.8439 | 0.8779 |

Across all eight cells, mean successive-displacement cosine is `-0.0632`; per-cell values span `-0.1084` to `-0.0428`.

The 50→100 and 100→200 displacement vectors are near-orthogonal and slightly opposed, while most jointly nonzero coordinates retain perturbation sign from step 100 to step 200. Together with state noncontraction and saw-toothed objective traces this strengthens the description `persistent oscillatory boundary-constrained fixed-step dynamics consistent with overshoot`.

It still does **not** establish a literal periodic limit cycle, step size as unique cause, or model-independent behavior beyond the two tested public architectures.
