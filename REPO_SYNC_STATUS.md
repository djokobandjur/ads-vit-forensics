# Canonical repository synchronization status

Branch: `tifs-canonical-20260911`

Base: historical `main` at `cb0709a78c88602a5a8fed314508d369f9849fcd`.

Draft PR: #1 — **DO NOT MERGE YET**.

## Source/protocol synchronization state

Repository-to-canonical-package conformance is now closed for the synchronized source/protocol snapshot.

Immutable audited source/protocol snapshot:

`ac580b9524c287882d97f4660fd7bf7791ea0c73`

Conformance results:

- source-byte conformance: **8/8 PASS**;
- exact available protocol conformance: **17/17 PASS**;
- source-to-manuscript independent oracle/unit checks: **36/36 PASS**;
- GitHub Actions exact protocol restore/verification run `34637439733`: **SUCCESS**.

The exact canonical source files required for the audited primary/source-conformance chain are present and byte-bound to their canonical SHA-256 identities. The exact executed Stage-B source is preserved unchanged; its known metadata-only finalization exception is documented in `canonical/STAGE_B_FINALIZATION_RECOVERY_NOTE.md`, and recovery performed no new PGD, inference, score, or AUC computation.

Adaptive v2.7 engine/runner files are explicitly retained as **pilot implementation lineage** only. They are not represented as the unrecovered confirmatory adaptive v2.7.2 runner.

Machine-readable public-repository conformance records are under `provenance/`.

## Current scientific authority

See `provenance/SCIENTIFIC_AUTHORITY.md`.

Current ledger:

`ADS_TIFS_SCIENTIFIC_RESULTS_AND_INTERPRETATION_LEDGER_UPDATED_POST_GITHUB_CONFORMANCE_v1_21_20260911.md`

SHA-256:

`916865096723af369d51a81d648a3d19491b76bc598914509f5b3b99db008938`

Status:

`PASS_SOURCE_AND_PROTOCOL_CONFORMANCE / ZENODO_AND_MANUSCRIPT_FINALIZATION_PENDING`

## Persistent archival identifier

Zenodo concept DOI:

`10.5281/zenodo.19844729`

This DOI represents all versions and resolves to the latest published Zenodo release. No new version-specific DOI is claimed until a new-version draft actually reserves/mints it.

## Explicit archival gaps

The following remain explicit and must not be reconstructed or inferred from downstream artifacts:

- exact confirmatory adaptive v2.7.2 runner/source archive;
- exact specificity v2.6 execution runner/notebook;
- original comparison LOCK-002 JSON bytes.

## Remaining gates before final citation / merge

1. Prepare the synchronized Zenodo new-version release payload from the audited source snapshot and current reproducibility authority.
2. Create a **new version** from the existing Zenodo record and reserve the real version-specific DOI before publication.
3. Insert the exact public identifiers into main manuscript and supplement:
   - `https://github.com/djokobandjur/ads-vit-forensics`;
   - audited source snapshot `ac580b9524c287882d97f4660fd7bf7791ea0c73`;
   - Zenodo concept DOI `10.5281/zenodo.19844729`;
   - actual reserved/minted version-specific DOI.
4. Rebuild the final manuscript/supplement and replace any provisional paper payload in the Zenodo draft with the DOI-complete final files.
5. Run the final manuscript + supplement + GitHub + Zenodo cross-audit.
6. Only after that audit passes: publish the Zenodo version, finalize/merge PR #1, and record the published release identities.

Historical files in `scripts/`, `data/`, notebooks, and older paper material are intentionally preserved as provenance and explicitly non-authoritative for corrected claims.
