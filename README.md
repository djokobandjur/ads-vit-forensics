# Attention Divergence Score (ADS)

**Attention Divergence Score: A Forensic Metric for Characterizing Parameter-Level Attacks in Vision Transformers**

This repository is the public code/provenance companion for the corrected, source-audited IEEE TIFS manuscript lineage prepared in September 2026.

> **Canonical-status note.** Pre-2026-09-11 scripts and result JSON files are preserved as historical provenance. They are not the methodological or numerical authority for the corrected manuscript. The corrected source set is under [`canonical/`](canonical/); locked protocol material is under [`protocols/`](protocols/); conformance records are under [`reproducibility/`](reproducibility/) and [`provenance/`](provenance/).
>
> The synchronization branch `tifs-canonical-20260911` and draft PR #1 remain fail-closed until repository-to-package conformance is recorded. Do not cite an arbitrary branch HEAD as the final public-code identity before that gate is closed.

## Canonical ADS operator

ADS is the mean of **per-image** KL divergences computed from native pre-softmax attention logits. The corrected operator uses float64 divergence arithmetic, `log_softmax` in the logit domain, no additive probability epsilon floor, no probability renormalization, a mean reduction over heads/query rows within each image, then a mean over reference images.

- ADS semantic-spec SHA-256: `093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`
- ADS implementation SHA-256: `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6`
- Fixed 256-image ImageNet-100 reference-index SHA-256: `1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509`

## Corrected attack protocol

Canonical attack experiments use iterative projected CE ascent with `model.eval()`, the complete fixed 256-image reference objective, the current delta applied before every gradient evaluation, explicit coordinatewise L-infinity projection, saved deltas, and fail-closed behavior when an attacked gradient is absent.

Canonical PE surfaces are Learned `pos_embed`; Sinusoidal PE buffer/object; RoPE `cos_cached + sin_cached` with `inv_freq` excluded; and per-layer ALiBi slopes.

Fixed absolute epsilon is an implementation-space tampering budget, not an intrinsic cross-family robustness ranking. Cross-family interpretation uses prespecified damage-matched analysis, with a 5-percentage-point accuracy drop as the primary operating point.

## Reproducibility chain

The public reproducibility design traces

`manuscript specification -> exact source -> source SHA-256 -> locked protocol -> independent oracle/unit test -> canonical artifact -> reported claim`.

A separate numerical-replication layer reconstructs manuscript numbers from stored canonical result artifacts/per-image arrays. These two layers are complementary and are not silently substituted for one another.

The source-conformance audit includes 36/36 passing independent synthetic/oracle checks for the audited core pipeline. Large checkpoints, saved deltas, and completed-result archives remain hash-bound archival dependencies rather than being duplicated in Git.

## Canonical source identities

| Source | SHA-256 | Status |
|---|---|---|
| `canonical/ads_canonical_operator_v2_1.py` | `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6` | canonical |
| `canonical/ads_attack_engine_v2_1.py` | `e3c506a34cb27c10556bf62b2fefbf2b324b3ba5fc66f3d953f48ecc3bf2bf77` | canonical |
| `canonical/ads_primary_full_grid_v2_1.py` | `d0cc6e0ef1e914b5efea249dd22097d57b5da17a09906226e90a51a082f12c00` | canonical |
| `canonical/derive_damage_matched_v2_1.py` | `2d58a57523a3aa51c02de9abc4af9af161addb587dded26bd1afa85ef1ec28b7` | canonical |
| `canonical/full_scale_experiment_v1_6.py` | `4aa884cffc0afbb64b9b08c776d2e94173264340b5a92a84c927869c3db7e8ee` | canonical model/source lineage |
| `canonical/stage_b_saved_delta_reeval_v2_8_2.py` | `d83219e566ab4829de57ac875d561b8e4f6b47aa19b61531a77ed5b4f00e1add` | exact executed saved-delta reevaluation source; see recovery note |
| `canonical/canonical_adaptive_engine_v2_7.py` | `304e0159230520a181d31abcc1f3a9083f410657297ef85b10a0a60e4c7449bc` | adaptive pilot implementation lineage |
| `canonical/run_adaptive_pilot_v2_7_1.py` | `e881f02e9a2ea0289e6221cdeb22d3c5e18886641e9410213d8c999d09aac821` | pilot runner only |

The exact confirmatory adaptive v2.7.2 runner/source archive and the exact specificity v2.6 execution runner remain explicit archival source gaps. No byte identity is inferred for them from downstream artifacts.

## Repository layout

```text
canonical/        exact recovered/audited source for the corrected lineage
protocols/        locked protocol and analysis-specification material
reproducibility/  source-to-manuscript audit, oracle tests, and source identities
provenance/       scientific-authority pointer and repository-provenance records
paper/            historical/pre-correction paper material unless explicitly marked otherwise
scripts/          historical pre-correction scripts retained for provenance
```

Historical `scripts/` files intentionally remain in Git history/current tree. Several implement superseded ADS/attack semantics and must not be used to reproduce corrected claims; see [`HISTORICAL_CODE_NOTICE.md`](HISTORICAL_CODE_NOTICE.md).

## Scientific authority

The current scientific-authority filename/hash and archival policy are recorded in [`provenance/SCIENTIFIC_AUTHORITY.md`](provenance/SCIENTIFIC_AUTHORITY.md). The full append-only ledger is distributed with the archival reproducibility payload rather than duplicated wholesale in the source tree.

## Archival DOI

Zenodo concept DOI: **10.5281/zenodo.19844729**. This concept DOI represents all versions and resolves to the latest published Zenodo release.

For exact reproducibility, the final release record will map the audited Git source snapshot to the version-specific Zenodo release DOI and the SHA-256 of the archived release payload. A version-specific DOI is recorded only after that release is actually published.

## Repository provenance correction

`djokobandjur/vit-positional-adversarial` is a separate related project and is not the ADS/TIFS canonical repository. The previously cited commit `952ff4e7b81a220c40bc63483d332dc4d25277a2` is retired for ADS/TIFS provenance. The correction record is in [`provenance/ADS_TIFS_PUBLIC_REPOSITORY_PROVENANCE_AUDIT_v1_0_20260911.md`](provenance/ADS_TIFS_PUBLIC_REPOSITORY_PROVENANCE_AUDIT_v1_0_20260911.md).
