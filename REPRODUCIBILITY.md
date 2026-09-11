# Reproducibility and provenance

The corrected ADS/TIFS workflow distinguishes two verification layers:

1. **Source conformance**: manuscript equation/prose -> exact source function -> source SHA-256 -> locked protocol -> independent oracle/unit test.
2. **Numerical replication**: canonical stored artifact/per-image arrays -> derived analyses/tables -> manuscript number/claim.

Neither layer is silently substituted for the other.

## Canonical operator and attack

The authoritative source files are stored under `canonical/`. Their SHA-256 identities are listed in `reproducibility/SOURCE_IDENTITY_MANIFEST.csv` and the repository README.

The source audit is `reproducibility/ADS_TIFS_SOURCE_TO_MANUSCRIPT_SEMANTIC_AUDIT_v1_0_20260911.md`; its independent test record is `reproducibility/SOURCE_ORACLE_TEST_RESULTS.json` (36/36 PASS).

## Scientific authority

The current scientific-results ledger is:

`provenance/ADS_TIFS_SCIENTIFIC_RESULTS_AND_INTERPRETATION_LEDGER_UPDATED_POST_REPO_AUDIT_v1_20_20260911.md`

This ledger explicitly retires the erroneous provenance claim that pointed to `vit-positional-adversarial@952ff...`.

## External archival payloads

Large checkpoints, saved deltas, and completed-result ZIPs remain external hash-bound dependencies. Their exact hashes are recorded in the scientific ledger and completed audit material. The versioned public archival distribution is intended to be Zenodo; concept DOI: `10.5281/zenodo.19844729`.

## Final release gate

Before merging the canonical branch and minting the next Zenodo version:

- verify every file under `canonical/` against the execution-package SHA-256 identities;
- verify protocol/operator/reference hashes;
- verify the provisional manuscript and supplement use only corrected claims;
- freeze the Git commit;
- archive that commit/reproducibility payload to Zenodo;
- record the version-specific Zenodo DOI and archive SHA-256;
- then update the reader-facing paper source with the exact public identifiers.
