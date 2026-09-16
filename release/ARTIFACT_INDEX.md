# Canonical Artifact Index — Post-Phase-2

This map preserves file → experiment/protocol → result → manuscript/release traceability for the compact public release.

| Layer | Public artifact / pointer | Canonical role |
|---|---|---|
| Scientific authority | `provenance/LATEST_SCIENTIFIC_LEDGER.md` | Points to Ledger v1.62, SHA-256 `c3cc0a791517671e1e819c7bbd54518236d61a5a80bb4ec0a452b8f4ee4872d1`. |
| ADS semantics | `canonical/ads_canonical_operator_v2_1.py` + protocol locks | Native pre-softmax logits; float64 KL; no floor/renormalization; operator SHA `093e...7362`. |
| Attack generator | `canonical/ads_attack_engine_v2_1.py`, `canonical/ads_primary_full_grid_v2_1.py`, `protocols/FULL_GRID_v2_0__*` | Corrected projected CE ascent and canonical fixed-stress/damage-matched design. |
| Primary damage matching | `protocols/FULL_GRID_v2_0__DAMAGE_MATCHED_ANALYSIS_SPEC.md` and damage-matched decision/spec material | First adjacent crossing; exact hit direct; linear interpolation in damage; identical weight; no extrapolation. |
| Comparison/ROC | `protocols/COMPARISON_ROC_v2_3__*` | Corrected same-image ADS/Attn-L2/diagonal-Mahalanobis/LogitKL definitions and benign transforms. |
| Specificity | locked v2.6 protocol/output material + Ledger v1.62 | Primary 5-pp PE/control comparison; dense attribution gate and conservative controls kept distinct. |
| Adaptive | `protocols/ADAPTIVE_v2_7__ADAPTIVE_PROTOCOL_LOCK_v2_7.json` + final adaptive audit | Canonical ADS-aware reoptimization; damage–stealth limitation; no adaptive-proof claim. |
| P2-C | `provenance/phase2/ADS_TIFS_PHASE2_P2_C_STEP_HORIZON_FINAL_EXTERNAL_AUDIT_v1_0_20260915.md`, decision JSON, post-hoc memo | Public-model fixed-step horizon sensitivity; both models nonconverged by step 200 under the locked criterion. |
| P2-D | `provenance/phase2/ADS_TIFS_PHASE2_P2_D_TARGETED_STRUCTURAL_CONFIRMATORY_FINAL_AUDIT_v1_0_20260915.md` + decision JSON | 432-state target-redirection confirmatory; modest low-damage target progress; ADS tracks independent target. |
| Manuscript | `provenance/manuscript/ADS_TIFS_SUBMISSION_CLOSURE_v1_5_2_FINAL_AUDIT_20260916.md` | Final 12-page main + 6-page supplement closure; scientific change NONE. |
| Archival payload | `release/POST_PHASE2_RELEASE_CANDIDATE.json` | Payload filename, byte count, SHA-256, DOI state, and archival-gap boundary. |

## Final manuscript package

`ADS_TIFS_TIFS_SUBMISSION_PACKAGE_v1_5_2_20260916.zip`  
SHA-256: `da888c7c6befa3c9d1b19432def11689ac4397496c1abbcff9f177a3480add0a`.

## Prepared Zenodo payload

`ADS_TIFS_CANONICAL_REPRO_RELEASE_POST_PHASE2_v3_0_0_20260916.zip`  
SHA-256: `708d36074f40447f12705c24dbee36c7095e0dfe9bcac891f689cbcb7f1e6396`.

See `ARCHIVAL_GAPS.md` for payloads intentionally not represented as included/recovered.
