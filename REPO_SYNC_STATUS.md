# Canonical repository synchronization status

Branch: `tifs-canonical-20260911`

Base: historical `main` at `cb0709a78c88602a5a8fed314508d369f9849fcd`.

Draft PR: #1 — **DO NOT MERGE YET**.

## Source/protocol synchronization state

The exact canonical source files required for the audited primary/source-conformance chain are now present on the branch and byte-bound to the canonical source package:

- `canonical/ads_canonical_operator_v2_1.py` — SHA-256 `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6`
- `canonical/ads_attack_engine_v2_1.py` — SHA-256 `e3c506a34cb27c10556bf62b2fefbf2b324b3ba5fc66f3d953f48ecc3bf2bf77`
- `canonical/ads_primary_full_grid_v2_1.py` — SHA-256 `d0cc6e0ef1e914b5efea249dd22097d57b5da17a09906226e90a51a082f12c00`
- `canonical/derive_damage_matched_v2_1.py` — SHA-256 `2d58a57523a3aa51c02de9abc4af9af161addb587dded26bd1afa85ef1ec28b7`
- `canonical/full_scale_experiment_v1_6.py` — SHA-256 `4aa884cffc0afbb64b9b08c776d2e94173264340b5a92a84c927869c3db7e8ee`
- `canonical/stage_b_saved_delta_reeval_v2_8_2.py` — SHA-256 `d83219e566ab4829de57ac875d561b8e4f6b47aa19b61531a77ed5b4f00e1add`
- `canonical/canonical_adaptive_engine_v2_7.py` — SHA-256 `304e0159230520a181d31abcc1f3a9083f410657297ef85b10a0a60e4c7449bc` — pilot implementation lineage
- `canonical/run_adaptive_pilot_v2_7_1.py` — SHA-256 `e881f02e9a2ea0289e6221cdeb22d3c5e18886641e9410213d8c999d09aac821` — pilot runner only

The exact executed Stage-B source is intentionally preserved unchanged. Its known metadata-only finalization exception is documented in `canonical/STAGE_B_FINALIZATION_RECOVERY_NOTE.md`; recovery performed no new PGD, inference, score, or AUC computation.

The available full-grid, damage-matched, corrected-comparison, specificity, adaptive, and post-audit protocol material has also been restored from an exact archived bundle. The restore workflow verified bundle SHA-256 `21325d92ae1275058d332a53ca4e20be414ab36d9a391d35622df42ed15740d1` and each individual protocol-file SHA-256 before committing the payload.

GitHub Actions run `34637439733` completed **SUCCESS** for the repaired exact source+protocol restore/verification flow.

## Current scientific authority

See `provenance/SCIENTIFIC_AUTHORITY.md` for the latest ledger identity:

`ADS_TIFS_SCIENTIFIC_RESULTS_AND_INTERPRETATION_LEDGER_UPDATED_POST_REPO_AUDIT_v1_20_20260911.md`

SHA-256 `0c232fb8f4e74e5baa8d13f48482e3ae71c2a795e8cbc189130c9d4d6be421e7`.

## Explicit archival gaps

The exact confirmatory adaptive v2.7.2 runner/source archive and exact specificity v2.6 execution runner remain unrecovered. They are not reconstructed or inferred. The original comparison LOCK-002 JSON bytes also remain unavailable; the locked Markdown semantics/results are preserved instead.

## Remaining gates before merge/final citation

1. Complete and record repository-to-canonical-package conformance for the synchronized source/protocol snapshot.
2. Freeze the audited immutable Git source snapshot that the manuscript will cite.
3. Keep PR #1 draft until that conformance record is committed and reviewed.
4. Publish the corresponding new Zenodo version; record its version-specific DOI and release-archive SHA-256 while retaining concept DOI `10.5281/zenodo.19844729`.
5. Update the manuscript/supplement with the exact GitHub URL, audited source-snapshot commit, Zenodo concept DOI, and actual version-specific DOI.
6. Run the final main+supplement/repository/Zenodo cross-audit before TIFS upload.

Historical files in `scripts/`, `data/`, and older paper material are intentionally preserved as provenance and explicitly non-authoritative for corrected claims.
