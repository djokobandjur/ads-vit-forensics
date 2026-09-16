# Reproducibility boundary — canonical ADS/TIFS release

## Authority order

Use the following precedence when artifacts conflict:

1. latest explicit protocol lock;
2. latest completed canonical result artifact with verified provenance;
3. latest Scientific Results and Interpretation Ledger;
4. repository/release prose;
5. historical manuscript text.

Current scientific authority is Ledger v1.62, SHA-256 `c3cc0a791517671e1e819c7bbd54518236d61a5a80bb4ec0a452b8f4ee4872d1`.

## Canonical numerical/operator lock

Canonical ADS is the mean of per-image KL divergences from native pre-softmax attention logits, with float64 divergence arithmetic, `torch.log_softmax`, no additive epsilon floor, no probability renormalization, mean reduction over heads/query rows per image, then mean over reference images.

Operator SHA-256: `093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`.

Reference-index SHA-256: `1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509`.

Transformed 256-image reference-cache SHA-256: `01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4`.

## Canonical attack lock

Canonical attack experiments use corrected iterative projected CE ascent, `model.eval()`, the full fixed 256-image reference objective, current delta applied before every gradient evaluation, explicit per-group L-infinity projection, saved deltas, and fail-closed behavior when attacked gradients are absent.

The final manuscript does not claim optimizer-horizon invariance of the primary 5-pp matched-damage estimand.

## Damage matching

The primary cross-family estimand is 5 percentage points of full-validation accuracy drop. Damage matching uses the first adjacent crossing, exact hits directly, otherwise linear interpolation in damage, identical interpolation weight for epsilon/ADS/profile/per-image ADS, and no extrapolation.

## Historical material

Historical pseudo-PGD, aggregate-first `KL(mean||mean)`, degenerate `attn.mean(-1)` attention comparison, leakage-prone probing, and asymmetric historical `all_weights` specificity are retained only as provenance and must not be used for final claims.

## Phase-2 closure

P2-C public-model step-horizon diagnostic: complete; both public models are classified `FIXED_STEP_TRAJECTORY_NOT_CONVERGED_BY_200` under the locked fixed-step criterion.

P2-D targeted structural confirmatory: complete / partial positive; 432-state tree verified, all 12 checkpoint units estimable, low-damage target progress modest, and ADS tracks the independent target with per-seed Spearman association around 0.95.

Exact final audit/decision artifacts are under `provenance/phase2/`.

## Compact release boundary

The compact GitHub/Zenodo release contains authority, audit, protocol/specification, manuscript, and figure-source material needed to trace final claims. It does not duplicate every large execution payload.

Explicit archival gaps:

- exact confirmatory adaptive v2.7.2 runner/source archive;
- exact matched-specificity execution runner/notebook;
- original comparison-protocol JSON bytes;
- P2-C/P2-D protocol-lock byte files absent from the compact project upload (audited hashes retained in closure artifacts);
- 2,387 densified-stress saved deltas, approximately 401 GB, retained separately by hash.

No missing byte is silently reconstructed or represented as original.

## Release identities

Final submission v1.5.2 ZIP SHA-256: `da888c7c6befa3c9d1b19432def11689ac4397496c1abbcff9f177a3480add0a`.

Prepared post-Phase-2 archival payload SHA-256: `708d36074f40447f12705c24dbee36c7095e0dfe9bcac891f689cbcb7f1e6396`.

Zenodo concept DOI: `10.5281/zenodo.19844729`. Record a new version-specific DOI only after actual publication.
