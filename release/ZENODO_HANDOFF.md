# Zenodo handoff — post-Phase-2 canonical release

Concept DOI: `10.5281/zenodo.19844729`

Target release version: `3.0.0`

Prepared payload:
`ADS_TIFS_CANONICAL_REPRO_RELEASE_POST_PHASE2_v3_0_0_20260916.zip`

SHA-256:
`708d36074f40447f12705c24dbee36c7095e0dfe9bcac891f689cbcb7f1e6396`

Prepared metadata: `release/ZENODO_METADATA_v3_0_0.json`.

## Publication sequence

1. Freeze the GitHub repository state and create tag/release `v3.0.0`.
2. Create a **new version** under the existing Zenodo concept record; do not create an unrelated concept record.
3. Upload the exact payload above and apply the prepared metadata.
4. Publish the Zenodo version.
5. Record the resulting **version-specific DOI** in a metadata-only GitHub commit and in `CITATION.cff`/`CITATION.md` as appropriate.
6. Cross-check: Git tag, Git commit, Zenodo version DOI, payload filename, byte count, payload SHA-256, Ledger v1.62 SHA-256, and submission v1.5.2 SHA-256.

## DOI rule

The concept DOI is not a substitute for the version-specific DOI. Until publication, repository prose must state that the version-specific DOI is pending.

## Scientific rule

Zenodo publication is a release/provenance action only. It does not authorize scientific reinterpretation, ledger modification, new optimizer work, or replacement of canonical result identities.

## Archival gaps

See `release/ARCHIVAL_GAPS.md`. The compact deposit does not claim inclusion of the approximately 401 GB densified-stress saved-delta payload or byte recovery of explicitly unrecovered runners/protocol bytes.
