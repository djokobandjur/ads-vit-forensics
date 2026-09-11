# ADS-TIFS GitHub Repository-to-Canonical-Package Conformance Audit v1.0

**Date:** 2026-09-11  
**Repository:** `djokobandjur/ads-vit-forensics`  
**Historical base:** `cb0709a78c88602a5a8fed314508d369f9849fcd`  
**Audited source snapshot:** `ac580b9524c287882d97f4660fd7bf7791ea0c73`  
**Branch:** `tifs-canonical-20260911`  
**Draft PR:** `#1`  
**Decision:** `PASS_SOURCE_AND_PROTOCOL_CONFORMANCE / ZENODO_AND_MANUSCRIPT_FINALIZATION_PENDING`

## Scope

This audit closes the public-repository source/protocol synchronization gate. It does not claim that a new Zenodo version has already been published and it does not elevate known archival gaps into reconstructed source.

The verification target is the immutable Git commit `ac580b9524c287882d97f4660fd7bf7791ea0c73`. Later commits may add this audit record, the next ledger pointer, manuscript source, or release metadata; those later commits must not be substituted for the audited source snapshot without a new conformance check.

## Source-byte conformance

All eight source files represented in the source identity manifest were checked by Git blob identity against the exact local canonical/audited bytes. A matching Git blob SHA-1 means the repository byte stream is identical to the local file used to compute the recorded canonical SHA-256.

Result: **8/8 PASS**.

The primary corrected pipeline is byte-bound in Git to the canonical ADS operator, corrected projected-CE attack engine, primary full-grid runner, damage-matched derivation, exact model/source implementation used by canonical runs, and exact executed Stage-B saved-delta reevaluation source.

The adaptive engine and v2.7.1 runner are retained only as **pilot implementation lineage**. They are not represented as the unrecovered confirmatory v2.7.2 runner.

## Protocol conformance

The available protocol set was archived locally as a deterministic tar.gz payload with SHA-256

`21325d92ae1275058d332a53ca4e20be414ab36d9a391d35622df42ed15740d1`.

GitHub Actions run `34637439733` reconstructed the protocol payload on the branch, verified the archive SHA-256, then verified all **17/17 individual protocol-file SHA-256 identities** before committing the restored files.

Result: **17/17 PASS**.

This includes the exact JSON protocol locks for the full-grid, specificity, adaptive, and post-audit branches plus the available locked Markdown specifications for corrected comparison and damage-matched analysis.

The original comparison LOCK-002 JSON bytes remain unavailable; no reconstructed JSON has been substituted.

## Historical-state preservation

The pre-correction `scripts/`, `data/`, notebooks, and older paper material remain in the repository/history. They are explicitly labeled historical/non-authoritative for corrected final claims. In particular, the known superseded ADS/attack semantics in the old public scripts are disclosed rather than silently rewritten.

The erroneous ADS/TIFS provenance citation to `djokobandjur/vit-positional-adversarial@952ff4e7b81a220c40bc63483d332dc4d25277a2` is explicitly retired in provenance records. The correct public repository is `djokobandjur/ads-vit-forensics`.

## Scientific authority

At the time of this audit, the latest scientific-results authority is:

`ADS_TIFS_SCIENTIFIC_RESULTS_AND_INTERPRETATION_LEDGER_UPDATED_POST_REPO_AUDIT_v1_20_20260911.md`

SHA-256:

`0c232fb8f4e74e5baa8d13f48482e3ae71c2a795e8cbc189130c9d4d6be421e7`

The full append-only ledger remains an archival reproducibility payload; Git contains a reader-facing authority pointer rather than duplicating the entire internal ledger.

## Persistent archival identifier

The repository records Zenodo concept DOI `10.5281/zenodo.19844729` with the correct semantics: it represents all versions and resolves to the latest published Zenodo release.

No version-specific DOI is claimed yet. That identifier must be recorded only after the new synchronized release is actually published.

## Known source/archive gaps retained

The following remain explicit and unresolved:

- exact confirmatory adaptive v2.7.2 runner/source archive;
- exact specificity v2.6 execution runner/notebook;
- original comparison LOCK-002 JSON bytes.

They are archival reproducibility gaps, not detected numerical contradictions.

## Decision

`PASS_SOURCE_AND_PROTOCOL_CONFORMANCE`.

The public Git repository now contains byte-identical canonical primary source and the available exact protocol set required for the corrected/source-audited lineage.

The next gates are archival/editorial rather than new scientific computation:

1. advance the scientific ledger with this completed repository-conformance audit;
2. preserve `ac580b9524c287882d97f4660fd7bf7791ea0c73` as the immutable **audited source snapshot**;
3. publish the corresponding new Zenodo version and record its real version-specific DOI plus release-archive SHA-256;
4. update main manuscript and supplement to cite the correct GitHub URL, audited source snapshot, Zenodo concept DOI, and actual version-specific DOI after minting;
5. run the final main+supplement+GitHub+Zenodo cross-audit before TIFS upload.
