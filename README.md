# Attention Divergence Score (ADS)

**Attention Divergence Score: A Forensic Metric for Characterizing Model-State Tampering in Vision Transformers**

This repository is the public code, source, and provenance companion for the corrected IEEE TIFS manuscript and its audited reproducibility release.

> **Release state (2026-09-17): v3.1.0 PUBLICATION-RESULT REPRODUCIBILITY SYNCHRONIZATION.** The experimental program is closed. The current scientific authority is Ledger v1.68 / AUD-049. The externally synchronized reader-facing manuscript package is v1.6.12 (12-page main + 6-page supplement). This repository/release synchronization introduces no new experiment and changes no scientific result.

## ADS operator

ADS is the mean of **per-image** KL divergences computed from native pre-softmax attention logits. The audited implementation uses float64 divergence arithmetic, `log_softmax` in the logit domain, no additive probability epsilon floor, no probability renormalization, a mean reduction over heads/query rows within each image, then a mean over reference images. It fails closed for row-wise KL below `-1e-11` and clamps only residual negative roundoff in `[-1e-11,0)` to zero.

- ADS semantic-spec SHA-256: `093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`
- ADS implementation SHA-256: `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6`
- fixed 256-image ImageNet-100 reference-index SHA-256: `1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509`
- transformed 256-image reference-cache SHA-256: `01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4`

## Corrected attack protocol

Attack experiments used iterative projected CE ascent with `model.eval()`, the full fixed 256-image reference objective, current delta applied before every gradient evaluation, explicit L-infinity projection, saved deltas, and fail-closed behavior if attacked gradients are absent.

The positional-encoding/state surfaces are Learned `pos_embed`; Sinusoidal PE buffer/object; RoPE `cos_cached + sin_cached` with `inv_freq` excluded; and per-layer ALiBi slopes. Fixed absolute epsilon is an implementation-space stress budget, not an intrinsic cross-family robustness ranking. Cross-family interpretation therefore uses prespecified damage matching, with 5 percentage points of full-validation accuracy drop as the primary operating point.

## Scientific authority

Current scientific authority:

`ADS_TIFS_SCIENTIFIC_RESULTS_AND_INTERPRETATION_LEDGER_POST_AUD049_v1_68_20260917.md`

SHA-256: `17b87ab675c0e6ff8d7ae79dbdbf5dcbc4468322496ffce50df57734501b7ba8`.

`provenance/LATEST_SCIENTIFIC_LEDGER.md` records this pointer. The full append-only ledger is distributed in the v3.1.0 reproducibility archive rather than reconstructed from manuscript prose.

## Interpretation boundary

ADS is presented as a **forensic characterization / triage metric**. The evidence does not support a universal detector threshold, a defense, adaptive-proof behavior, or unique PE attribution for an arbitrary candidate state. Layer 4 is a fixed operational coordinate rather than a universally privileged layer.

## Reproducibility release v3.1.0

Zenodo reproducibility archive:

`ADS_TIFS_COMPACT_COMPLETE_REPRODUCIBILITY_v3_1_0_20260917.zip`

SHA-256: `1097fd77274c5fff54d9e13e56da2bc9b5d0c0f104ad9e5501536cc01fb831e1`  
Size: `223252213` bytes.

The archive supports publication-result numerical and figure reproducibility. Its fail-closed verifier passes **29/29 direct gates**, including direct recomputation of the `3/72` coarse endpoint sensitivity, the strict-disjoint dense-control `24/24 > 1` result with minimum `1.1681245478`, and source verification of the ROT180 target-KL direction.

It deliberately excludes manuscript/supplement documents, ImageNet image bytes, model checkpoint binaries, the approximately 401-GB densified-stress saved-delta store, and P2-D `.pt` delta tensors.

## Repository layout

```text
canonical/        recovered/audited source for the corrected lineage
protocols/        protocol and analysis-specification material
reproducibility/  source conformance and release-audit pointers
provenance/       scientific-authority and audit records
release/          current and historical public-release metadata
scripts/, data/   historical pre-correction material retained as provenance
```

Historical `scripts/`, older `data/`, and earlier release material remain intentionally preserved and must not be treated as the corrected methodological authority.

## Zenodo

- Concept DOI (stable reader-facing identifier): **10.5281/zenodo.19844729**
- v3.1.0 version DOI: **10.5281/zenodo.22802588**

The concept DOI resolves to the latest published Zenodo version; the version DOI identifies the exact v3.1.0 deposit.

## Source identity

The earlier immutable corrected source/protocol snapshot remains `ac580b9524c287882d97f4660fd7bf7791ea0c73`. The public v3.0.0 release and its metadata remain historical provenance and are not rewritten by v3.1.0.
