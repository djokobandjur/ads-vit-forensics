# Zenodo handoff — v3.1.0 reproducibility release

Concept DOI: `10.5281/zenodo.19844729`

v3.1.0 version DOI: `10.5281/zenodo.22802588`

Target release version: `3.1.0`

Prepared payload:
`ADS_TIFS_COMPACT_COMPLETE_REPRODUCIBILITY_v3_1_0_20260917.zip`

SHA-256:
`1097fd77274c5fff54d9e13e56da2bc9b5d0c0f104ad9e5501536cc01fb831e1`

Size: `223252213` bytes.

Companion checksum file:
`ADS_TIFS_COMPACT_COMPLETE_REPRODUCIBILITY_v3_1_0_20260917.zip.sha256.txt`

## Publication sequence

1. Synchronize GitHub `main` with the v3.1.0 reader-facing/release metadata.
2. Publish a new version under the existing Zenodo concept record using the exact ZIP and checksum file above.
3. Verify that the published v3.1.0 DOI resolves and that the files match the expected identities.
4. Create Git tag/release `v3.1.0` on the synchronization commit; do not rewrite the historical v3.0.0 tag/release.

## DOI use

Reader-facing manuscript text uses the stable concept DOI `10.5281/zenodo.19844729`. Release/provenance metadata may additionally record the exact v3.1.0 DOI `10.5281/zenodo.22802588`.

## Scientific rule

Zenodo/GitHub publication is a release/provenance action only. It does not authorize scientific reinterpretation, ledger modification, new optimizer work, or replacement of completed-result identities.

## Scope

The v3.1.0 archive supports publication-result numerical and figure reproducibility. It intentionally excludes manuscript/supplement documents, ImageNet image bytes, checkpoint binaries, the approximately 401-GB densified-stress saved-delta store, and P2-D `.pt` delta tensors.
