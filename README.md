# Attention Divergence Score (ADS)

**Attention Divergence Score: A Forensic Metric for Characterizing Parameter-Level Attacks in Vision Transformers**

This repository is the public code/source/provenance companion for the corrected IEEE TIFS manuscript and its canonical reproducibility audit.

> **Release state (2026-09-16): POST-PHASE-2 RELEASE CANDIDATE.** The experimental program is closed and the manuscript is frozen as submission v1.5.2 (12-page main + 6-page supplement). This repository update advances the public authority from the pre-Phase-2 v1.31 freeze to Ledger v1.62. Packaging introduces no new experiment or scientific result.

## Canonical ADS operator

ADS is the mean of **per-image** KL divergences computed from native pre-softmax attention logits. The canonical implementation uses float64 divergence arithmetic, `log_softmax` in the logit domain, no additive probability epsilon floor, no probability renormalization, a mean reduction over heads/query rows within each image, then a mean over reference images. It fails closed for row-wise KL below `-1e-11` and clamps only residual negative roundoff in `[-1e-11,0)` to zero.

- ADS semantic-spec SHA-256: `093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`
- ADS implementation SHA-256: `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6`
- fixed 256-image ImageNet-100 reference-index SHA-256: `1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509`
- transformed 256-image reference-cache SHA-256: `01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4`

## Corrected attack protocol

Canonical attacks use iterative projected CE ascent with `model.eval()`, the full fixed 256-image reference objective, current delta applied before every gradient evaluation, explicit L-infinity projection, saved deltas, and fail-closed behavior if attacked gradients are absent.

Canonical PE surfaces are Learned `pos_embed`; Sinusoidal PE buffer/object; RoPE `cos_cached + sin_cached` with `inv_freq` excluded; and per-layer ALiBi slopes. Fixed absolute epsilon is an implementation-space stress budget, not an intrinsic cross-family robustness ranking. Cross-family interpretation therefore uses prespecified damage matching, with 5 percentage points of full-validation accuracy drop as the primary operating point.

## Final scientific state

The scientific authority is Ledger v1.62:

`ADS_TIFS_SCIENTIFIC_RESULTS_AND_INTERPRETATION_LEDGER_POST_AUDIT_POSTHOC_v1_62_20260915.md`

SHA-256: `c3cc0a791517671e1e819c7bbd54518236d61a5a80bb4ec0a452b8f4ee4872d1`.

The closed manuscript submission is v1.5.2. Submission ZIP SHA-256:
`da888c7c6befa3c9d1b19432def11689ac4397496c1abbcff9f177a3480add0a`.

Phase-2 provenance under `provenance/phase2/` distinguishes byte-identical Git copies from canonical SHA pointers to exact artifacts carried in the archival payload. The final manuscript closure audit is under `provenance/manuscript/`.

## Interpretation boundary

ADS is presented as a **forensic characterization / triage metric**. The final evidence does not support a universal detector threshold, a defense, adaptive-proof behavior, or unique PE attribution. Layer 4 remains a fixed operational coordinate rather than a universally privileged layer.

## Repository layout

```text
canonical/        exact recovered/audited source for the corrected lineage
protocols/        locked protocol and analysis-specification material
reproducibility/  source conformance and release-audit pointers
provenance/       scientific authority, Phase-2 closure, and manuscript audit records
release/          post-Phase-2 release candidate, artifact map, archival gaps, Zenodo handoff
scripts/, data/   historical pre-correction material retained as provenance
```

Historical `scripts/`, older `data/`, and earlier paper material remain intentionally preserved and must not be treated as the corrected methodological authority.

## Archival boundary

This compact public release does **not** claim byte recovery of every execution artifact. The exact confirmatory adaptive v2.7.2 runner/source archive, exact matched-specificity execution runner/notebook, and original comparison-protocol JSON bytes remain explicit archival gaps. The 2,387 densified-stress saved deltas (approximately 401 GB) remain separately retained by hash rather than duplicated in Git/Zenodo. See `release/ARCHIVAL_GAPS.md`.

## Zenodo

Zenodo concept DOI: **10.5281/zenodo.19844729**.

Prepared post-Phase-2 archival payload:
`ADS_TIFS_CANONICAL_REPRO_RELEASE_POST_PHASE2_v3_0_0_20260916.zip`

SHA-256: `708d36074f40447f12705c24dbee36c7095e0dfe9bcac891f689cbcb7f1e6396`.

A new version-specific DOI must be recorded only after that Zenodo version is actually published; the concept DOI must not be substituted for it.

## Source identity

The earlier immutable corrected source/protocol snapshot remains `ac580b9524c287882d97f4660fd7bf7791ea0c73`. The post-Phase-2 release tag will identify the repository state that adds final authority, Phase-2 audit, manuscript, and archival metadata without rewriting historical source provenance.
