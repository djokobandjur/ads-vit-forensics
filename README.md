# Attention Divergence Score (ADS)

**Attention Divergence Score: A Forensic Metric for Characterizing Parameter-Level Attacks in Vision Transformers**

This repository is being synchronized to the corrected, source-audited IEEE TIFS manuscript lineage prepared on 2026-09-11.

> **Canonical-status note.** The pre-2026-09-11 scripts and result JSON files that remain in the repository are preserved as historical provenance. They must not be treated as the numerical or methodological authority for the corrected manuscript. The canonical source set is under [`canonical/`](canonical/), and the current protocol/provenance material is under [`protocols/`](protocols/), [`reproducibility/`](reproducibility/), and [`provenance/`](provenance/).

## Canonical ADS operator

The corrected ADS operator computes the **mean of per-image KL divergences from native pre-softmax attention logits**. It uses float64 divergence arithmetic, `log_softmax` in the logit domain, no additive probability epsilon floor, no probability renormalization, mean reduction over heads/query rows per image, then mean over reference images.

- Canonical ADS semantic-spec SHA-256: `093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`
- Canonical implementation SHA-256: `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6`
- Fixed ImageNet-100 256-image reference-index SHA-256: `1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509`

## Corrected attack protocol

Canonical attack experiments use corrected iterative projected CE ascent with `model.eval()`, the full fixed 256-image reference objective, the current delta applied before every gradient evaluation, explicit L-infinity projection, saved deltas, and fail-closed behavior when attacked gradients are absent.

Canonical PE attack surfaces are:

- Learned: `pos_embed`
- Sinusoidal: PE buffer/object
- RoPE: `cos_cached + sin_cached`, with `inv_freq` excluded
- ALiBi: per-layer `slopes`

Fixed absolute epsilon is an implementation-space tampering budget and is **not** interpreted as an intrinsic cross-family robustness ranking. Cross-family interpretation uses the prespecified damage-matched analyses, with 5-percentage-point accuracy drop as the primary operating point.

## Reproducibility chain

The corrected reproducibility package is organized around the traceability chain

`manuscript specification -> exact source -> source SHA-256 -> locked protocol -> independent oracle/unit test -> canonical artifact -> reported claim`.

The source-conformance audit in [`reproducibility/`](reproducibility/) includes 36/36 passing independent synthetic/oracle checks for the audited core pipeline.

Large checkpoints, saved deltas, and result archives are not duplicated in this Git repository. They remain hash-bound archival dependencies and are intended to be distributed through the versioned Zenodo release.

## Repository layout

```text
canonical/        exact recovered canonical source used/audited for corrected results
protocols/        protocol locks and analysis specifications
reproducibility/  source-to-manuscript conformance audit, tests, and manifests
provenance/       scientific ledger and repository-provenance audit
paper/            current provisional corrected manuscript/supplement source and bibliography
scripts/          historical pre-correction scripts preserved in Git history/current tree
```

The files in `canonical/` are the authoritative code entry points for the corrected manuscript lineage. Historical `scripts/` files are retained to preserve provenance; several implement superseded operator/attack semantics and should not be used to reproduce final claims.

## Current source identities

| Source | SHA-256 |
|---|---|
| `canonical/ads_canonical_operator_v2_1.py` | `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6` |
| `canonical/ads_attack_engine_v2_1.py` | `e3c506a34cb27c10556bf62b2fefbf2b324b3ba5fc66f3d953f48ecc3bf2bf77` |
| `canonical/ads_primary_full_grid_v2_1.py` | `d0cc6e0ef1e914b5efea249dd22097d57b5da17a09906226e90a51a082f12c00` |
| `canonical/derive_damage_matched_v2_1.py` | `2d58a57523a3aa51c02de9abc4af9af161addb587dded26bd1afa85ef1ec28b7` |
| `canonical/full_scale_experiment_v1_6.py` | `4aa884cffc0afbb64b9b08c776d2e94173264340b5a92a84c927869c3db7e8ee` |
| `canonical/stage_b_saved_delta_reeval_v2_8_2.py` | `d83219e566ab4829de57ac875d561b8e4f6b47aa19b61531a77ed5b4f00e1add` |

The exact confirmatory adaptive v2.7.2 runner/source archive and the exact specificity v2.6 execution runner remain explicitly documented archival source gaps; no byte identity is inferred for them.

## Paper source status

`paper/` currently contains the **provisional** post-repository-audit manuscript/supplement source. The final paper source will be updated only after this canonical repository branch is audited and the public commit/release identity is frozen.

## Archival DOI

Zenodo concept DOI: **10.5281/zenodo.19844729**. This DOI represents all versions and resolves to the latest Zenodo release.

## Citation / release policy

For exact reproducibility, cite the version-specific Zenodo release together with its archived Git commit and release-archive SHA-256. The concept DOI above is the stable pointer to the latest version.

## Historical repository state

The repository `djokobandjur/vit-positional-adversarial` is a separate related project and is **not** the ADS/TIFS canonical source repository. The previously cited commit `952ff4e7b81a220c40bc63483d332dc4d25277a2` has been retired for ADS/TIFS provenance.

See [`provenance/ADS_TIFS_PUBLIC_REPOSITORY_PROVENANCE_AUDIT_v1_0_20260911.md`](provenance/ADS_TIFS_PUBLIC_REPOSITORY_PROVENANCE_AUDIT_v1_0_20260911.md) for the correction record.
