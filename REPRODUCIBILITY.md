# Reproducibility and provenance

The corrected ADS/TIFS workflow separates two verification layers.

1. **Source conformance:** manuscript equation/prose -> exact source function -> source SHA-256 -> locked protocol -> independent oracle/unit test.
2. **Numerical replication:** canonical stored artifact/per-image arrays -> derived analysis/table -> manuscript number/claim.

Together these form the broader traceability chain

`manuscript -> source -> source SHA-256 -> protocol -> oracle/unit test -> artifact -> claim`.

Neither layer is silently substituted for the other.

## Canonical source layer

The exact recovered/audited source files are under `canonical/`. Their canonical SHA-256 identities are recorded in `reproducibility/SOURCE_IDENTITY_MANIFEST.csv` and summarized in the repository README.

The source-conformance audit is `reproducibility/ADS_TIFS_SOURCE_TO_MANUSCRIPT_SEMANTIC_AUDIT_v1_0_20260911.md`; the independent synthetic/oracle record is `reproducibility/SOURCE_ORACLE_TEST_RESULTS.json` and reports 36/36 PASS.

`canonical/STAGE_B_FINALIZATION_RECOVERY_NOTE.md` documents the metadata-only finalization exception in the exact executed Stage-B source. Scientific outputs were completed before that exception; the documented recovery performed no new PGD, inference, score, or AUC computation.

The adaptive files presently under `canonical/` are explicitly pilot implementation lineage. They are not represented as the missing confirmatory v2.7.2 runner.

## Protocol layer

`protocols/` contains the available locked full-grid, damage-matched, corrected comparison, specificity, adaptive, and post-audit protocol material. The synchronization workflow reconstructed this protocol payload from an archived exact bundle and verified both the bundle SHA-256 and every individual protocol-file SHA-256 before committing the files.

The original comparison LOCK-002 JSON bytes remain unavailable; no reconstructed JSON is substituted for them. The locked Markdown semantics and completed corrected comparison artifacts remain preserved.

## Scientific authority

The current ledger identity is recorded in `provenance/SCIENTIFIC_AUTHORITY.md`:

`ADS_TIFS_SCIENTIFIC_RESULTS_AND_INTERPRETATION_LEDGER_UPDATED_POST_REPO_AUDIT_v1_20_20260911.md`

SHA-256:

`0c232fb8f4e74e5baa8d13f48482e3ae71c2a795e8cbc189130c9d4d6be421e7`

The full append-only ledger is part of the archival reproducibility payload/Zenodo distribution. Git carries the authority pointer plus the source/protocol/provenance material needed for public code conformance.

## External archival payloads

Large checkpoints, saved deltas, per-cell score arrays, and completed-result ZIPs remain external hash-bound dependencies. Their exact identities are recorded in completed audit/reproducibility material. The versioned public archival distribution is Zenodo; persistent concept DOI: `10.5281/zenodo.19844729`.

## Known archival source gaps

The following are not silently reconstructed:

- exact confirmatory adaptive v2.7.2 runner/source archive;
- exact specificity v2.6 execution runner/notebook;
- original comparison LOCK-002 JSON bytes.

These are archival reproducibility gaps, not detected numerical contradictions.

## Final release gate

Before the synchronized branch becomes the citable public state:

- verify exact Git blob/byte identity for every canonical source file against the execution/source package;
- verify the protocol bundle and individual protocol hashes;
- record repository-to-package conformance;
- freeze the audited Git source snapshot;
- archive the final release/reproducibility payload to Zenodo;
- record the version-specific Zenodo DOI and release-archive SHA-256, while retaining concept DOI `10.5281/zenodo.19844729`;
- only then insert the audited Git snapshot and DOI identifiers into the reader-facing manuscript/supplement.

The final manuscript may be committed after the source snapshot is frozen; it should cite that immutable audited source snapshot rather than attempting to self-reference the later manuscript-containing repository HEAD.
