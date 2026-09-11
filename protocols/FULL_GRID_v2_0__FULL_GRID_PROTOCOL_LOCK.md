# ADS–TIFS FULL-GRID PROTOCOL LOCK v2.0

**Lock ID:** `ADS_TIFS_FULL_GRID_PROTOCOL_LOCK_v2_0_20260909`  
**Date:** 2026-09-09  
**Manuscript:** *Attention Divergence Score: A Forensic Metric for Characterizing Parameter-Level Attacks in Vision Transformers*  
**Manuscript ID:** T-IFS-27152-2026  
**Status:** `LOCKED_FOR_DOWNSTREAM_CANONICAL_RERUNS`  
**Canonical JSON SHA-256:** `2b242559d0ec523683ee05a33238c819790d17ebef919745cab3b426f161021f`

## 1. Scientific role

This lock freezes the corrected **PE-only canonical fixed-absolute attack grid** and the **damage-matched cross-family estimand** before any expensive final reruns. It does not redefine fixed epsilon or relative-coordinate rho as a fair intrinsic robustness scale.

The final interpretation is:

- fixed absolute epsilon = implementation-space tampering stress axis;
- relative-coordinate rho = explanatory scale/sign sensitivity control only;
- damage-matched analysis = primary cross-family comparative interpretation.

## 2. Canonical ADS — unchanged

```text
mean(per-image KL(clean || attacked))
input: native pre-softmax attention logits
log probabilities: torch.log_softmax(logits.float64, dim=-1)
probability floor: NONE
renormalization: NONE
arithmetic: float64
operator_spec_hash:
093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362
```

Any downstream package that disagrees with this operator must stop before execution.

## 3. Fixed reference/runtime lock

```text
Dataset: ImageNet-100 validation
Fixed reference images: 256
Reference-index SHA-256:
1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509

Transformed 256-image GPU-reference SHA-256:
01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4

Full 5000-image transformed validation-cache SHA-256:
90d47ea5212f5d06b9e8d12fa80ccb20cc722a46d79d2c926ef41be2917f0f71
```

FMLE standing execution policy:

1. notebook Cell 1 is the UID/cache bootstrap **before importing torch**;
2. `num_workers=0`;
3. deterministic algorithms and math-only SDPA;
4. decode/transform the 5000-image validation set once;
5. cache that transformed tensor on GPU once;
6. derive the fixed 256-image reference view from the same cache;
7. record and verify both transformed-tensor SHA values.

## 4. Final TIFS attack generator

The primary attack generator is intentionally the same corrected generator used to localize and confirm the final operating regions. The optimizer-characterization branch remains separate evidence and does not silently replace this generator.

```text
objective          = full fixed 256-image reference mean CE ascent
model mode         = eval
steps              = 20
alpha              = 0.1 * epsilon
restarts           = 1 deterministic trajectory
delta init         = zero
selected state     = raw final iterate after step 20
budget mode        = absolute per-group L_inf
surface            = PE-only
delta convention   = per-buffer
biases             = excluded
```

No claim of global optimality or strict numerical optimizer convergence is permitted.

### PE surfaces

```text
Learned:      1 PE group
Sinusoidal:   1 PE group
RoPE:         24 groups = 12 cos_cached + 12 sin_cached
              inv_freq NOT attacked
ALiBi:        12 per-block slope groups
```

## 5. Final absolute-epsilon execution grid

### 5.1 Common grid — all four PE families

```text
epsilon = {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}
```

This is the canonical common fixed-stress grid. The clean checkpoint is measured separately as epsilon=0.

### 5.2 High-tail extension

For threshold/severe-compromise bracketing only:

```text
Learned:      + {0.5, 1.0}
Sinusoidal:   + {0.5, 1.0}
RoPE:         + {0.5, 1.0}
ALiBi:        no 0.5/1.0 extension
```

Reason: ALiBi is already severe-compromised in 6/6 seeds at epsilon=0.05 and destructive at 0.1/0.2; pushing it to 0.5/1.0 adds no required operating-region information. High-tail points must never be used to construct a common intrinsic robustness ranking.

### 5.3 Planned manuscript subsets

**Common primary fixed-stress display:**

```text
{0.005, 0.01, 0.02, 0.05, 0.1, 0.2}
```

**Near-clean anchors (supplement / threshold support):**

```text
{0.001, 0.002}
```

**ALiBi low-dose primary/inset sequence:**

```text
{0.001, 0.002, 0.005, 0.01, 0.02, 0.05}
```

ALiBi 0.1/0.2 are retained as destructive stress endpoints, not as its normal operating region.

## 6. Damage-matched cross-family design — LOCKED

### 6.1 Damage definition

For PE family `p`, seed `s`, budget `epsilon`:

```text
D_p,s(epsilon) = clean_full_val_accuracy_p,s - attacked_full_val_accuracy_p,s
```

in percentage points.

### 6.2 Prespecified targets

```text
PRIMARY:   5 pp absolute accuracy drop
SECONDARY: 10 pp, 20 pp absolute accuracy drop
SEVERE:    attacked accuracy <= 0.5 * clean accuracy
```

The 5-pp target is the primary cross-family matched-damage estimand. The severe criterion remains a distinct operating-point definition and must not be conflated with 5 pp.

### 6.3 Bracketing/interpolation

For each PE × seed and each target:

1. sort observed cells by increasing epsilon;
2. find the **first** adjacent pair that brackets the target damage;
3. exact grid hits are used directly;
4. if several crossings exist because of non-monotonic damage, use the first/smallest-epsilon crossing and record the non-monotonicity flag;
5. interpolate **linearly in damage** between the two observed endpoints;
6. use the same interpolation weight for:
   - epsilon,
   - whole-profile mean ADS,
   - L4 ADS,
   - all 12 layer ADS values,
   - the `[256,12]` per-image ADS arrays;
7. **no extrapolation is allowed**.

If the 5-pp primary target is not bracketed for any PE × seed, the analysis fails closed and a targeted refinement must be separately locked and run. Secondary targets that are not bracketed are reported as `NOT_ESTIMABLE`; they are not replaced post hoc.

A target can be described as a **common damage-matched cross-family estimand** only if all `4 PE × 6 seeds = 24` units bracket it.

## 7. Aggregation/statistics lock

- Preserve seed as the statistical replication unit (`n=6`); never treat 256 reference images as independent seeds.
- Report per-seed values plus mean ± sample SD.
- For prespecified paired PE contrasts at matched damage, pair by the common seed ID.
- Primary pairwise output: within-seed difference, mean difference, 95% t-CI, and exact two-sided sign test.
- Do not choose contrasts or targets based on whichever p-values are favorable.
- Any multiplicity correction for the final manuscript table must be frozen before testing that table family.

## 8. Required artifacts / provenance

Every executed cell must retain or reference:

```text
checkpoint path + SHA-256
reference-index path + SHA-256
transformed-reference SHA-256
full validation-cache SHA-256
protocol lock ID + lock SHA-256
canonical operator hash
attack configuration
PE surface/group topology
seed
epsilon
clean and attacked full-validation accuracy
delta artifact + SHA-256
per-image ADS NPZ + SHA-256
runtime provenance
script/config hashes
aggregation/statistical procedure
```

Delta artifacts must receive `protocol_version` and `execution_implementation` from the caller; stale hard-coded labels are forbidden.

Resume is fail-closed: a completed cell may be skipped only after all protocol/provenance fields and artifact SHA values verify. Verified scientific artifacts are never overwritten automatically.

## 9. Scope boundary

This lock governs:

1. canonical PE-only fixed-absolute primary grids;
2. the derived damage-matched cross-family analysis.

It does **not** automatically lock branch-specific details for:

- corrected comparison baselines;
- specificity (`all_non_pe_weights`);
- any additional fine-threshold/ROC-only epsilon points;
- adaptive/ref-evasion/profile-aware attacks.

Those branches require their own explicit lock while inheriting the canonical ADS/reference/runtime safeguards.

## 10. Execution order after this lock

```text
1. corrected comparison pilot
2. corrected comparison n=6 if pilot passes
3. canonical fixed-absolute primary grid
4. damage-matched derivation + bracketing audit
5. specificity / fine-threshold / ROC under branch-specific locks
6. figures / tables / statistics
7. manuscript numerical rewrite
```

No downstream execution package may silently change the common epsilon grid, attack generator, operator, reference subset, PE surfaces, or damage-matched estimand without issuing a new protocol-lock version.
