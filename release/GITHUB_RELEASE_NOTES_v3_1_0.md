# v3.1.0 — Publication-result reproducibility synchronization

Version 3.1.0 synchronizes the public ADS/TIFS repository with the final reader-facing manuscript state and the compact-complete reproducibility archive prepared for Zenodo.

## What changed

- Scientific-authority pointer advanced to Ledger v1.68 / AUD-049 (`17b87ab675c0e6ff8d7ae79dbdbf5dcbc4468322496ffce50df57734501b7ba8`).
- Reader-facing title/terminology synchronized to **model-state tampering** rather than the older parameter-level wording.
- External manuscript traceability synchronized to v1.6.12 (12-page main + 6-page supplement), package SHA-256 `da31916dec2a9c90d65fb4e5e405d660220e1f0f79a3d5c38d23de9e994c3779`.
- Exact P2-D confirmatory execution/source package recovered and co-released in the Zenodo archive; the ROT180 target-KL direction is now source-traceable as `D_KL(ROT180(clean attention) || attacked attention)`.
- Current figure source/generator provenance and final figure outputs synchronized; obsolete figure-production material remains separately labeled historical.
- Source-identity metadata populated.
- Fail-closed verification expanded to 29/29 direct gates, including direct recomputation of the coarse `3/72` endpoint sensitivity and the strict-disjoint dense-control `24/24 > 1` result with minimum `1.1681245478`.
- Added a top-level MIT `LICENSE` to match the public software release metadata.

## Zenodo archive

`ADS_TIFS_COMPACT_COMPLETE_REPRODUCIBILITY_v3_1_0_20260917.zip`

SHA-256: `1097fd77274c5fff54d9e13e56da2bc9b5d0c0f104ad9e5501536cc01fb831e1`  
Size: `223252213` bytes.

Concept DOI: `10.5281/zenodo.19844729`  
v3.1.0 DOI: `10.5281/zenodo.22802588`

The archive is reproducibility-only: it does not redistribute manuscript/supplement documents, ImageNet image bytes, model checkpoint binaries, the approximately 401-GB densified-stress saved-delta store, or P2-D `.pt` delta tensors.

## Scientific status

**No new experiment and no scientific result change.** Version 3.1.0 is a reproducibility, provenance, and reader-facing synchronization release. Historical v3.0.0 and earlier release states remain preserved and are not rewritten.
