# v3.0.0 — Canonical post-audit reproducibility release

This release synchronizes the public ADS/TIFS repository with the completed canonical experimental program and the closed v1.5.2 IEEE TIFS submission.

## Added / advanced

- Scientific authority advanced from the pre-Phase-2 v1.31 freeze to Ledger v1.62.
- Added byte-identical P2-C step-horizon final audit/decision in Git, plus a SHA-bound pointer to the exact post-hoc trajectory memo carried in the archival payload.
- Added a SHA-bound P2-D canonical pointer; the byte-identical final audit/decision are carried in the archival payload rather than manually transcribed under original filenames in Git.
- Added byte-identical final adaptive n=6 external audit and final v1.5.2 manuscript closure audit.
- Added post-Phase-2 artifact map, machine-readable release manifest, archival-gap boundary, and Zenodo v3.0.0 metadata.

## Scientific status

No new experiment was run and no scientific result was recomputed by this release-packaging step. Historical, corrected, canonical, superseded, retired, pilot, and deferred states remain distinct.

## Key identities

- ADS operator SHA-256: `093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`
- reference-index SHA-256: `1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509`
- transformed reference-cache SHA-256: `01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4`
- Ledger v1.62 SHA-256: `c3cc0a791517671e1e819c7bbd54518236d61a5a80bb4ec0a452b8f4ee4872d1`
- submission v1.5.2 ZIP SHA-256: `da888c7c6befa3c9d1b19432def11689ac4397496c1abbcff9f177a3480add0a`
- prepared archival payload SHA-256: `708d36074f40447f12705c24dbee36c7095e0dfe9bcac891f689cbcb7f1e6396`

## Scope boundary

ADS remains a forensic characterization/triage metric. Fixed absolute epsilon is not an intrinsic cross-family robustness scale. This release does not claim universal detection, defense, adaptive-proof behavior, or unique PE attribution.

## Archival boundary

The approximately 401 GB densified-stress saved-delta payload is not duplicated in the compact release. Exact unrecovered source/protocol byte gaps are enumerated in `release/ARCHIVAL_GAPS.md`.

Zenodo concept DOI: `10.5281/zenodo.19844729`. Record the new version-specific DOI only after actual publication.
