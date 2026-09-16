# Release status

Date: 2026-09-16

Status: **POST-PHASE-2 RELEASE CANDIDATE / READY FOR GITHUB TAG + ZENODO VERSION PUBLICATION**

## Closed internal gates

- scientific authority advanced to Ledger v1.62;
- v1.5.2 manuscript closure PASS (12-page main, 6-page supplement);
- Phase-2 P2-C and P2-D closure artifacts identified and added to public provenance;
- release artifact index and archival-gap boundary prepared;
- local Zenodo payload assembled: 52 files total, 51 manifest entries, 0 manifest failures;
- local payload ZIP SHA-256: `708d36074f40447f12705c24dbee36c7095e0dfe9bcac891f689cbcb7f1e6396`;
- no scientific number/result is created by this release pass.

## Remaining external publication actions

1. merge/freeze the post-Phase-2 repository update and create Git tag/release `v3.0.0`;
2. publish a new version under Zenodo concept DOI `10.5281/zenodo.19844729` using the prepared payload/metadata;
3. record the actual version-specific DOI in a metadata-only follow-up commit and cross-check it against the Git tag and payload SHA.

A version-specific DOI must not be invented before publication.
