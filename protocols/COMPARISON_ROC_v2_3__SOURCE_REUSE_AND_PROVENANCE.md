# Source reuse and provenance rules

## Canonical attack-state source
The n=6 comparison/ROC execution does **not** run PGD. It reuses the exact delta artifacts from:

`ADS_TIFS_CANONICAL_PRIMARY_FULL_GRID_v2_1_20260909`

Source aggregate SHA-256:

`dcd2fa22d27a9a983dfadca3d49b56f776d316add01d201a2fe5563d757203d1`

Expected cells for this branch: 4 PE x 6 seeds x 8 common eps = **192**.

For every cell, before applying a delta:
1. locate the cell in the source aggregate;
2. verify PE, seed, epsilon, checkpoint SHA, reference SHA, operator hash and delta SHA;
3. hash the on-disk delta artifact and require an exact match;
4. reconstruct the attacked model from the clean checkpoint + delta;
5. never chain deltas across epsilon cells.

## Reference / operator locks
Reference-index SHA-256:
`1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509`

Transformed canonical 256-image reference stream SHA-256 (unmodified input cache):
`01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4`

Canonical ADS operator hash:
`093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`

## New artifacts required
- per-condition transformed-cache SHA manifest;
- per-image score arrays for every method, PE, seed, epsilon and condition;
- seed-level AUC summary JSON/CSV;
- source-delta verification ledger (192/192 required);
- clean feature variance/lambda provenance for diagonal Mahalanobis;
- runtime/package/script hashes;
- fail-closed decision file.

Historical `ads_roc_v2.json` and historical comparison results are provenance-only and may not be copied into final result tables.
