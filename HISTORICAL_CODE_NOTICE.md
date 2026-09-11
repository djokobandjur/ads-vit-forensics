# Historical pre-correction code notice

Files already present under the legacy `scripts/`, `data/`, notebook, and historical figure paths predate the September 2026 canonical reproducibility correction. They are preserved for provenance and to avoid silently rewriting project history.

They are **not** the authority for the corrected IEEE TIFS manuscript unless a file is explicitly cross-listed by exact SHA-256 in the current scientific ledger or reproducibility manifest.

Known superseded behaviors in the historical public `scripts/ads_experiment.py` include dataset-averaged attention before KL, additive probability epsilon/renormalization, a one-batch reference attack objective, PGD in `train()` mode, and inclusion of RoPE `inv_freq`. These behaviors must not be used for corrected final claims.

Use `canonical/` and `protocols/` for the corrected source/protocol lineage.
