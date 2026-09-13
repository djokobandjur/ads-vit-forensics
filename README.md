# Attention Divergence Score (ADS)

**Attention Divergence Score: A Forensic Metric for Characterizing Parameter-Level Attacks in Vision Transformers**

This repository is the public code/source/provenance companion for the corrected IEEE TIFS manuscript and its frozen reproducibility audit.

> **Release state (2026-09-13): PRE-ZENODO RELEASE READY.** The scientific and manuscript-number audit is closed. The immutable canonical source/protocol snapshot cited by the manuscript is [`ac580b9524c287882d97f4660fd7bf7791ea0c73`](https://github.com/djokobandjur/ads-vit-forensics/commit/ac580b9524c287882d97f4660fd7bf7791ea0c73). Later repository commits add release/citation metadata only; they do not replace that audited source identity.

## Canonical ADS operator

ADS is the mean of **per-image** KL divergences computed from native pre-softmax attention logits. The canonical implementation uses float64 divergence arithmetic, `log_softmax` in the logit domain, no additive probability epsilon floor, no probability renormalization, a mean reduction over heads/query rows within each image, then a mean over reference images. It fails closed for row KL below `-1e-11` and clamps only residual negative roundoff in `[-1e-11,0)` to zero.

- ADS semantic-spec SHA-256: `093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`
- ADS implementation SHA-256: `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6`
- fixed 256-image ImageNet-100 reference-index SHA-256: `1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509`

## Corrected attack protocol

Canonical attacks use iterative projected CE ascent with `model.eval()`, the full fixed 256-image reference objective, current delta applied before every gradient evaluation, explicit L-infinity projection, saved deltas, and fail-closed behavior if attacked gradients are absent.

Canonical PE surfaces are Learned `pos_embed`; Sinusoidal PE buffer/object; RoPE `cos_cached + sin_cached` with `inv_freq` excluded; and per-layer ALiBi slopes. Fixed absolute epsilon is an implementation-space stress budget, not an intrinsic cross-family robustness ranking. Cross-family interpretation therefore uses prespecified damage matching, with 5 percentage points of full-validation accuracy drop as the primary operating point.

## Frozen scientific state

At the primary 5-pp coordinate, the recovered original EXP-019 artifacts reproduce all 72 seed-level PE/control whole-profile comparisons above one; the paired family/control mean ratio range is **6.55--15.88**. A separate resolved adaptive RoPE state at 5.17-pp strict damage falls inside the conservative non-PE envelope (`R_upper = 0.802`; bootstrap 2.5% bound `0.660`), so ADS is presented as a forensic characterization/triage metric rather than a unique attack-surface classifier.

The final compact replication archive independently passes the dense-grid, adaptive-bootstrap, and EXP-019 headline reproduction gates. Exact public release identities are under [`release/`](release/) and [`reproducibility/`](reproducibility/).

## Repository layout

```text
canonical/        exact recovered/audited source for the corrected lineage
protocols/        locked protocol and analysis-specification material
reproducibility/  source conformance and final release-audit pointers
provenance/       scientific-authority and repository-provenance records
release/          pre-Zenodo freeze manifest and handoff instructions
scripts/, data/   historical pre-correction material retained as provenance
```

Historical `scripts/`, older `data/`, and earlier paper material remain intentionally preserved and must not be treated as the corrected methodological authority.

## Scientific authority

Current authority: `ADS_TIFS_SCIENTIFIC_RESULTS_AND_INTERPRETATION_LEDGER_POST_EXP019_RECOVERY_v1_31_20260913.md`, SHA-256 `50efb67906fa9bc8058c3843cd2036dce25169e0fdf7720b190271881423a9b4`. The full append-only ledger is distributed with the archival replication payload rather than duplicated wholesale in Git.

## Release payload prepared for Zenodo

Final compact replication package:

- `ADS_TIFS_SUBMISSION_REPLICATION_PACKAGE_v1_5_POST_v1_31_FREEZE_20260913.zip`
- bytes: `368622268`
- SHA-256: `80f421a22c878a95809c62fea8ae6a1711cef9db4547bc10b1835650fed82943`

Final submission wrapper SHA-256: `ef34d7e899bd0c1e05dd528a53de7f52af21779c8b68862ec505b91c3f374f0b`.

Zenodo concept DOI: **10.5281/zenodo.19844729**. The version-specific DOI is intentionally not claimed until the new Zenodo version is actually published.

## Repository provenance correction

`djokobandjur/vit-positional-adversarial` is a separate related project and is not the ADS/TIFS canonical repository. The previously cited commit `952ff4e7b81a220c40bc63483d332dc4d25277a2` is retired for ADS/TIFS provenance.
