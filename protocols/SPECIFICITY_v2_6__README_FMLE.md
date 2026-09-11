# FMLE execution — Canonical Specificity n=6 v2.6

Install/unzip under:
`/home/djoko.bandjur.ftnkm/Notebooks/ADS/`

Open:
`ADS_TIFS_CANONICAL_SPECIFICITY_N6_v2_6_20260910/notebooks/01_CANONICAL_SPECIFICITY_N6_v2_6.ipynb`

Run **Cell 1 first** (UID/cache bootstrap before any torch import), then execute cells in order.

Required existing source result:
`/home/djoko.bandjur.ftnkm/Notebooks/ADS/results/ads_tifs_canonical_primary_full_grid_v2_1_20260909/ads_tifs_canonical_primary_full_grid_v2_1.json`

The run creates 864 new non-PE attack cells, losslessly gzip-compresses each float32 delta artifact, measures full validation accuracy and canonical per-image ADS, derives 5/10/20-pp matched specificity, and finally creates a compact `_RESULTS.zip`. Before packaging, all 864 gzip deltas are re-verified both by compressed SHA-256 and decompressed raw SHA-256. The default upload ZIP omits the very large `deltas_gzip/` directory but includes its complete manifest; the exact lossless deltas remain in the FMLE result folder. Run the packer with `--include_deltas` only if a full local archival ZIP is desired.

**Storage note:** QKV/MLP/all-non-PE delta artifacts are much larger than PE deltas. The package uses lossless gzip level 1 and records both the raw pre-compression SHA and compressed SHA. Do not delete the result folder until the final artifact audit is complete.

Protocol lock SHA-256: `c3fc549f2a4110e08fe6a12d276195b44d6083cba76f3c33b6d6cd10bbe7565e`
