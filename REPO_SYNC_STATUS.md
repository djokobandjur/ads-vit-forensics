# Canonical repository synchronization status

Branch: `tifs-canonical-20260911`

Base: historical `main` at `cb0709a78c88602a5a8fed314508d369f9849fcd`.

Status: **IN PROGRESS — DO NOT MERGE / DO NOT CITE AS FINAL YET**.

## Already synchronized and byte-bound

The following corrected canonical files have been added from the audited execution/source package and are tracked by their canonical SHA-256 identities:

- `canonical/ads_canonical_operator_v2_1.py` — `14d3c110c7a6219ae8b486c1e8a56360815d0b3d7069ce703ebbc726e25b4fd6`
- `canonical/ads_attack_engine_v2_1.py` — `e3c506a34cb27c10556bf62b2fefbf2b324b3ba5fc66f3d953f48ecc3bf2bf77`
- `canonical/ads_primary_full_grid_v2_1.py` — `d0cc6e0ef1e914b5efea249dd22097d57b5da17a09906226e90a51a082f12c00`
- `canonical/derive_damage_matched_v2_1.py` — `2d58a57523a3aa51c02de9abc4af9af161addb587dded26bd1afa85ef1ec28b7`

The repository also now contains the corrected reproducibility overview, source-conformance audit, 36/36 oracle-test record, manuscript-to-source conformance matrix, public-repository provenance audit, and the full-grid protocol lock.

## Remaining before merge/final citation

1. Add the exact canonical model implementation `full_scale_experiment_v1_6.py`.
2. Add the exact saved-delta reevaluation runner `stage_b_saved_delta_reeval_v2_8_2.py` together with a prominent finalization-recovery note: the executed runner has a known metadata-only NameError after all scientific outputs were already written; recovery v2.8.2.1 performed no new PGD, inference, or score recomputation.
3. Add the adaptive pilot-lineage source with explicit PILOT/PROVENANCE labeling; do not imply it is the missing confirmatory v2.7.2 runner.
4. Add the remaining available protocol-lock/provenance files and current scientific ledger.
5. Add the provisional corrected manuscript/supplement source only after repository wording is synchronized.
6. Run repository-to-canonical-package byte/hash conformance for every exact source file.
7. Freeze the final Git commit only after all conformance gates pass.
8. Archive the frozen release to Zenodo and record its version-specific DOI, while retaining concept DOI `10.5281/zenodo.19844729`.
9. Only then insert the exact Git commit and DOI identifiers into the reader-facing manuscript/supplement.

Historical files in `scripts/` and `data/` are intentionally preserved and explicitly marked non-authoritative rather than silently overwritten.
