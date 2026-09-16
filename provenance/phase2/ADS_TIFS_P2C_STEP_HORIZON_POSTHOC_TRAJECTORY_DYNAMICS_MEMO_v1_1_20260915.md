# ADS–TIFS P2-C Step-Horizon Post-Hoc Trajectory Dynamics Memo v1.0

**Date:** 2026-09-15  
**Role:** `POST-HOC / EXPLORATORY DIAGNOSTIC USING EXISTING LOCK-008 TRAJECTORY RECORDS`  
**New optimization runs:** `NONE`  
**Source result bundle SHA-256:** `b7b59d61d17aaa381f840559d1c0638c434ec0e6fb83276045389707b61f6bb1`

## Scope

This memo audits a secondary trajectory-level interpretation proposed after LOCK-008
closed. It does not change the locked result:

- DeiT: `FIXED_STEP_TRAJECTORY_NOT_CONVERGED_BY_200`
- RoPE: `FIXED_STEP_TRAJECTORY_NOT_CONVERGED_BY_200`

The analysis uses only the already-saved 200-step `reference_mean_ce_preupdate`,
`grad_linf`, and `delta_linf_after_step` records.

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

After first contact, `delta_linf` remains equal to epsilon to float32 numerical tolerance
(maximum absolute deviation: DeiT `4.172e-08`, RoPE
`9.537e-09`).

## What the records support

The late objective dynamics are strongly oscillatory rather than monotone.

For RoPE, only 3.3% of the total net CE gain occurs
after step 100, while the objective decreases on an average of
47.50 of 99 late transitions and the late total variation is
16.6 times the net late drift.

For DeiT the same qualitative pattern is present, but the residual net drift is larger:
13.1% of the total CE gain occurs after step 100.

This supports the description:

`OSCILLATORY_FIXED_STEP_BOUNDARY_DYNAMICS / ITERATE_NONCONTRACTION`.

It is consistent with overshoot from a constant sign-PGD step that is too coarse for
late-stage refinement, especially for RoPE.

## What the records do not establish

The existing records do **not** causally prove that the constant step size is the unique
cause of the failure.

In particular:

1. no smaller-alpha or decayed-step trajectory was executed;
2. full delta states were saved only at 20/50/100/200, not at every step;
3. therefore a periodic state-space `limit cycle` was not directly demonstrated;
4. a nonconvex/multibasin landscape and coarse fixed-step dynamics are not uniquely
   separable from these scalar trajectory traces alone.

Accordingly, manuscript wording should avoid:

`the fixed step induces a limit cycle`

and instead use:

`the fixed-step schedule exhibits persistent oscillatory boundary-constrained dynamics
consistent with overshoot and remains noncontractive through 200 steps`.

## Objective/evaluation-population mismatch

The public reference bridge contains 256 images spanning 92 distinct ImageNet-1K class
indices, whereas full-validation damage is measured over the complete 1000-class
ImageNet-1K validation set.

This mismatch is real and should be recorded as a separate explanatory limitation.
However, the present data do not prove that it causes the large reference-objective /
full-validation-damage divergence. A smaller step would not remove the population
mismatch itself, but could still change its quantitative effect.

## Recommendation

For the manuscript, stop here.

A smaller-alpha/decay experiment is not needed to support the current TIFS claim. It
would only be needed if the manuscript intended to make the stronger causal statement
that the constant step size, rather than deeper attack-surface geometry, is the primary
cause.

The manuscript-safe conclusion is:

> The public-checkpoint attack trajectories are horizon-sensitive and remain
> noncontractive under the fixed `alpha/epsilon=0.1` sign-PGD schedule. Their late
> reference-objective traces are strongly oscillatory, especially for RoPE, which is
> consistent with coarse-step overshoot but does not by itself establish a literal
> limit cycle or uniquely identify step size as the cause.

---

## v1.1 addendum — snapshot-state displacement geometry

**Status:** `POST-HOC / EXPLORATORY / NO NEW EXECUTION`  
**Source artifacts:** shipped step-50, step-100 and step-200 delta snapshots from all eight LOCK-008 trajectories.

The following quantities were recomputed directly from the saved delta tensors:

- successive displacement vectors: `u = delta_100 - delta_50`, `v = delta_200 - delta_100`;
- displacement cosine: `cos(u,v)`;
- straightness: `||delta_200-delta_50||_2 / (||u||_2 + ||v||_2)`;
- sign agreement between `delta_100` and `delta_200`.

| quantity | DeiT | RoPE |
|---|---:|---:|
| mean `cos(u,v)` | -0.0644 | -0.0621 |
| mean straightness | 0.6851 | 0.6863 |
| mean sign agreement, all coordinates | 0.8323 | 0.8748 |
| mean sign agreement, jointly nonzero coordinates | 0.8439 | 0.8779 |

Across all eight cells, the mean successive-displacement cosine is `-0.0632`.
Per-cell cosine values span `-0.1084` to
`-0.0428`.

### Interpretation

The 50->100 and 100->200 displacement vectors are near-orthogonal and slightly opposed,
while most jointly nonzero coordinates retain their perturbation sign from step 100 to
step 200.

This is inconsistent with a simple picture of continued approximately collinear
progress toward a stationary point. Together with the previously recorded state
noncontraction and saw-toothed reference-objective traces, it strengthens the
description:

`persistent oscillatory boundary-constrained fixed-step dynamics consistent with overshoot`.

It still does **not** establish:

- a literal periodic limit cycle;
- that step size is the unique causal mechanism;
- model-independent behavior beyond the two tested public architectures.

The close numerical similarity between the two model families is therefore reported as
a cross-model observation within these eight trajectories, not as proof of a universal
schedule property.

### Manuscript-safe one-sentence form

> Between the shipped 50-, 100-, and 200-step states, successive displacement vectors
> are near-orthogonal and slightly opposed (mean cosine about -0.06 across all eight
> trajectories), while roughly 84% of jointly nonzero DeiT coordinates and 88% of
> jointly nonzero RoPE coordinates retain their perturbation sign, further indicating
> continued boundary-constrained motion without approximately collinear late-stage
> progress.
