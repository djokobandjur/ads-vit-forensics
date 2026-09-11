# ADS-TIFS Source-Code-to-Manuscript Semantic Audit v1.0

**Date:** 2026-09-11  
**Scope:** source semantics and provenance for the current canonical ADS-TIFS manuscript lineage  
**Status:** `PASS_CORE / PARTIAL_ARCHIVAL_COVERAGE`  
**New PGD:** `NO`  
**New scientific estimand:** `NO`

## Why this audit exists

Numerical replication from stored result artifacts can establish

`stored artifact -> derived table -> manuscript number`.

It cannot by itself establish

`manuscript formula/protocol -> executed source code -> stored artifact`.

This audit therefore checks the source layer explicitly, with special emphasis on the
parts of the pipeline whose semantic mismatch could invalidate the scientific claim.

## Exact production-source identities recovered

The canonical primary full-grid aggregate records the exact executed identities:

- `ads_primary_full_grid_v2_1.py` SHA-256
  `d0cc6e0ef1e914b5efea249dd22097d57b5da17a09906226e90a51a082f12c00`
- `ads_attack_engine_v2_1.py` SHA-256
  `e3c506a34cb27c10556bf62b2fefbf2b324b3ba5fc66f3d953f48ecc3bf2bf77`
- `ads_canonical_operator_v2_1.py` SHA-256
  `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6`
- `full_scale_experiment_v1_6.py` SHA-256
  `4aa884cffc0afbb64b9b08c776d2e94173264340b5a92a84c927869c3db7e8ee`

The files extracted for this audit match those identities exactly.

The final Stage-B execution metadata independently records:
- `stage_b_saved_delta_reeval_v2_8_2.py` SHA-256
  `d83219e566ab4829de57ac875d561b8e4f6b47aa19b61531a77ed5b4f00e1add`;
- the same canonical attack-engine/operator/model hashes as above.

## Independent oracle/unit tests

`run_source_oracle_tests.py` executes **36/36 PASS** checks.

The tests include:
- an independent NumPy log-softmax/KL oracle for clean-to-test canonical ADS;
- a synthetic case proving mean(per-image KL) rather than KL(mean attention);
- a nonlinear CE objective that detects stale-delta-gradient ordering;
- a missing-gradient fail-closed test;
- mock attack-surface membership tests;
- exact-hit / first-crossing / no-extrapolation damage-match tests;
- an independent profile-LSE gradient check;
- a query-mean attention-feature test rejecting the historical degenerate mean-over-keys feature;
- diagonal-Mahalanobis formula checks;
- a hand-computable tie-aware AUC oracle;
- exact-tested-epsilon boundary checks.

## Findings

### Canonical ADS operator - PASS

The source implements native pre-softmax clean-to-test KL, float64 `log_softmax`,
no additive probability floor, no probability renormalization, per-image head/query
reduction, then image averaging.

A small implementation safeguard is visible in source: row-KL values below `-1e-11`
fail closed; remaining tiny negative roundoff is clamped to zero. This is not a
probability floor and does not alter the locked operator semantics, but it should remain
visible in reproducibility documentation.

### Corrected CE projected ascent - PASS

The engine uses `eval()`, applies `base + current_delta` before each gradient
evaluation, accumulates exact full-reference mean CE, performs sign ascent, explicitly
projects each attacked group, fails closed on absent gradients, and saves deltas with
protocol metadata. The nonlinear oracle specifically validates the current-delta
ordering.

### Attack surfaces - PASS

The source matches the declared Learned, Sinusoidal, RoPE, ALiBi, QKV, MLP, and broad
all-non-PE surfaces, including RoPE `inv_freq` exclusion and the inclusion of ordinary
biases in the broad non-PE control.

### Full-grid runner / deterministic provenance - PASS

The runner contains the locked seed/epsilon grids, exact transformed-cache hashes,
reference-index hash, deterministic settings, and source implementation manifest.

### Damage-matched derivation - PASS

First crossing, exact hit direct, linear-in-damage interpolation, common interpolation
weight, and no extrapolation all match the manuscript and pass independent tests.

### Corrected comparison/AUC implementation - PASS

The Stage-B source exposes the corrected query-mean key-salience feature, distinct
Attn-L2 and diagonal Mahalanobis scores, exact variance regularizer, clean-to-test
LogitKL, tie-aware rank AUC, and exact-tested-epsilon all-six-seed boundary rule.

### Stage-B non-reference/holdout logic - PASS

The implementation creates the 4,744-image non-reference partition, verifies and reuses
saved deltas, reconstructs the original 5,000-image accuracy, and evaluates the locked
disjoint holdout without regenerating attacks.

### Adaptive source - semantic PASS, archival coverage incomplete

The locally recovered `canonical_adaptive_engine_v2_7.py` from the locked v2.7.1 pilot
implements `CE - lambda*ADS(L4)` and
`CE - lambda*T*logsumexp(ADS_l/T)` with `T=0.01`, using the canonical differentiable
ADS path and projected current-delta ascent. Its formula and gradient are independently
tested.

The exact v2.7.2 confirmatory runner/source archive bytes are not present in the
currently mounted authority set. This audit therefore records
`PASS_WITH_PROVENANCE_LIMIT` rather than assuming byte identity.

### Specificity source - partial archival coverage

The common corrected attack engine fully specifies the QKV/MLP/all-non-PE surfaces and
CE ascent used by the canonical specificity branch. The exact v2.6 specificity
execution notebook/runner is not in the current local authority set, so full
runner-level source conformance remains an archival gap.

## Decision

`PASS_CORE / PARTIAL_ARCHIVAL_COVERAGE`.

No source-to-manuscript semantic discrepancy was found in the core ADS operator,
corrected attack engine, canonical full-grid runner, damage-matching derivation,
corrected comparison/AUC implementation, or Stage-B reevaluation.

**Submission implication:** the highest-risk historical failure mode - divergence
between manuscript/protocol semantics and the executed primary attack/ADS code - is now
directly audited for the canonical primary pipeline.

**Archival action before public reproducibility freeze:** recover and add the exact
confirmatory adaptive v2.7.2 execution source and exact specificity v2.6 runner if they
still exist in FMLE/project archives. Their absence is a reproducibility-packaging gap,
not evidence of a numerical contradiction.

See:
- `MANUSCRIPT_SOURCE_CONFORMANCE_MATRIX.csv`
- `SOURCE_ORACLE_TEST_RESULTS.json`
- `SOURCE_IDENTITY_MANIFEST.csv`
