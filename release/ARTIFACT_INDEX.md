# Artifact Index — current public state

This map preserves file → protocol/result → interpretation/release/manuscript traceability for the current publication-result reproducibility release and final reader-facing manuscript package.

| Layer | Public artifact / pointer | Role |
|---|---|---|
| Scientific authority | `provenance/LATEST_SCIENTIFIC_LEDGER.md` | Points to Ledger v1.68 / AUD-049, SHA-256 `17b87ab675c0e6ff8d7ae79dbdbf5dcbc4468322496ffce50df57734501b7ba8`. Full ledger is included in the v3.1.0 Zenodo archive. |
| ADS semantics | `canonical/ads_canonical_operator_v2_1.py` + protocol locks | Native pre-softmax logits; float64 KL; no floor/renormalization; semantic SHA `093e0e...7362`. |
| Attack generator | `canonical/ads_attack_engine_v2_1.py`, `canonical/ads_primary_full_grid_v2_1.py`, `protocols/FULL_GRID_v2_0__*` | Corrected projected CE ascent and fixed-stress/damage-matched design. |
| Primary damage matching | `protocols/FULL_GRID_v2_0__DAMAGE_MATCHED_ANALYSIS_SPEC.md` | First adjacent crossing; exact hit direct; linear interpolation in damage; identical weight; no extrapolation. |
| Comparison/ROC | `protocols/COMPARISON_ROC_v2_3__*` | Corrected same-image ADS/Attn-L2/diagonal-Mahalanobis/LogitKL definitions and benign transforms. |
| Specificity | v2.6 protocol/output material + v3.1.0 compact result archive | Primary 5-pp PE/control comparison, corrected non-PE controls, and direct `3/72` / dense-control verification. |
| Adaptive | `protocols/ADAPTIVE_v2_7__ADAPTIVE_PROTOCOL_LOCK_v2_7.json` + adaptive external audit | ADS-aware reoptimization and damage–stealth limitation; no adaptive-proof claim. |
| P2-C | Phase-2 final audit/closure records + post-hoc sufficient statistics in the Zenodo archive | Public-model fixed-step horizon sensitivity; raw result bundle remains externally hash-bound. |
| P2-D | Phase-2 final audit/closure records + recovered exact execution source in the Zenodo archive | 432-state targeted structural analysis; exact ROT180 target-KL direction source identity closed in v3.1.0. |
| Figure production | Current figure-source/provenance material in the v3.1.0 Zenodo archive | Final figure source data, active generators/provenance, and figure outputs; obsolete production material retained separately as historical. |
| Reader-facing manuscript traceability | external package identity below | Final 12-page main + 6-page supplement; not redistributed in the reproducibility-only archive. |
| Reproducibility archive | `release/REPRODUCIBILITY_RELEASE_v3_1_0.json` | Exact published ZIP identity, DOI identities, authority identity, scope and verifier state. |

## Final external reader-facing manuscript package

`ADS_TIFS_MANUSCRIPT_v1_6_13_FINAL_PUBLIC_RELEASE_SYNC_20260917.zip`  
SHA-256: `7bbc433dda9b829e412085797a33e1e8691457c2cb940d6077eed880624fd00b`.

Version v1.6.13 is a final public-release identity synchronization of v1.6.12 and changes no scientific result or interpretation.

## Zenodo reproducibility archive

`ADS_TIFS_COMPACT_COMPLETE_REPRODUCIBILITY_v3_1_0_20260917.zip`  
SHA-256: `1097fd77274c5fff54d9e13e56da2bc9b5d0c0f104ad9e5501536cc01fb831e1`  
Size: `223252213` bytes.

Concept DOI: `10.5281/zenodo.19844729`  
v3.1.0 DOI: `10.5281/zenodo.22802588`

GitHub tag/release `v3.1.0` remains the published release snapshot at commit `30bc80108f102a0687d8a7989a909abae2e9c066`. Current-main manuscript-traceability metadata may advance without rewriting that tag/release.

See `ARCHIVAL_GAPS.md` for intentionally omitted or still-unrecovered execution material. Historical v3.0.0 release metadata remains preserved under its version-specific files and Git tag.
