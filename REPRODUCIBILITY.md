# Reproducibility boundary — ADS/TIFS v3.1.0

## Authority order

Use the following precedence when artifacts conflict:

1. latest explicit protocol lock;
2. latest completed canonical result artifact with verified provenance;
3. latest Scientific Results and Interpretation Ledger;
4. repository/release prose;
5. historical manuscript text.

Current scientific authority is Ledger v1.68, SHA-256 `17b87ab675c0e6ff8d7ae79dbdbf5dcbc4468322496ffce50df57734501b7ba8`.

## ADS numerical/operator lock

ADS is the mean of per-image KL divergences from native pre-softmax attention logits, with float64 divergence arithmetic, `torch.log_softmax`, no additive epsilon floor, no probability renormalization, mean reduction over heads/query rows per image, then mean over reference images.

Operator semantic SHA-256: `093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`.

Implementation SHA-256: `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6`.

Reference-index SHA-256: `1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509`.

Transformed 256-image reference-cache SHA-256: `01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4`.

## Attack protocol

Attack experiments use corrected iterative projected CE ascent, `model.eval()`, the full fixed 256-image reference objective, current delta applied before every gradient evaluation, explicit L-infinity projection, saved deltas, and fail-closed behavior when attacked gradients are absent.

The final manuscript does not claim optimizer-horizon invariance of the primary 5-pp matched-damage estimand.

## Damage matching

The primary cross-family estimand is 5 percentage points of full-validation accuracy drop. Damage matching uses the first adjacent crossing, exact hits directly, otherwise linear interpolation in damage, identical interpolation weight for epsilon/ADS/profile/per-image ADS, and no extrapolation.

## Defined v3.1.0 scope

The Zenodo v3.1.0 archive supports publication-result numerical and figure reproducibility: reported tables, summary statistics, post-hoc diagnostics, decision criteria, figure-source data, and final figure outputs can be recomputed from included canonical or explicitly manifested compact artifacts.

The archive is:
`ADS_TIFS_COMPACT_COMPLETE_REPRODUCIBILITY_v3_1_0_20260917.zip`

SHA-256: `1097fd77274c5fff54d9e13e56da2bc9b5d0c0f104ad9e5501536cc01fb831e1`.

The fail-closed verifier passes 29/29 direct gates and the full-coverage release gate.

## v3.1.0 provenance closure

- exact P2-D confirmatory execution/source package is co-released and hash-verified;
- ROT180 target-KL direction is source-traceable as `D_KL(ROT180(clean attention) || attacked attention)`;
- current figure generators/provenance and final figure outputs are included;
- obsolete figure-production material remains separately labeled historical;
- `source_identity/` is populated;
- direct gates rederive the `3/72` coarse endpoint sensitivity and strict-disjoint dense-control `24/24 > 1`, minimum `1.1681245478` result.

## Deliberate exclusions / remaining execution-provenance boundary

The reproducibility archive intentionally excludes ImageNet image bytes, model checkpoint binaries, approximately 401 GB of densified-stress saved deltas, the 432 P2-D `.pt` delta tensors, and multi-GB intermediates when compact sufficient statistics are present.

The exact confirmatory-adaptive runner/source archive and exact matched-specificity runner/notebook remain historical execution-provenance gaps where recorded. The original comparison-protocol JSON bytes were not recovered; the corrected textual protocol/specification and completed results are retained. The P2-C raw result bundle remains externally hash-bound while publication-used sufficient statistics and audit records are included.

No missing byte is silently reconstructed or represented as original.

## Historical material

Historical pseudo-PGD, aggregate-first `KL(mean||mean)`, degenerate `attn.mean(-1)` attention comparison, leakage-prone probing, and asymmetric historical `all_weights` specificity remain provenance only and must not be used for final claims.

## Release identities

Scientific authority: `ADS_TIFS_SCIENTIFIC_RESULTS_AND_INTERPRETATION_LEDGER_POST_AUD049_v1_68_20260917.md`, SHA-256 `17b87ab675c0e6ff8d7ae79dbdbf5dcbc4468322496ffce50df57734501b7ba8`.

External reader-facing manuscript package: `ADS_TIFS_MANUSCRIPT_v1_6_12_S13_RULE_AND_SUPPLEMENT_HYGIENE_20260917.zip`, SHA-256 `da31916dec2a9c90d65fb4e5e405d660220e1f0f79a3d5c38d23de9e994c3779`.

Zenodo concept DOI: `10.5281/zenodo.19844729`.

Zenodo v3.1.0 DOI: `10.5281/zenodo.22802588`.
