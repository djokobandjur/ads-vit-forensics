# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-05-18

First stable release prepared for IEEE TIFS resubmission of *"Attention
Divergence Score: A Forensic Metric for Characterizing Parameter-Level
Attacks in Vision Transformers"*.

### Added

- 11 production experiment scripts with consistent Tier 1 CLI interface:
  required paths (`--models_dir`, `--val_dir`, `--output_path`,
  `--ref_indices_path`) plus optional `--pe_types` and `--seeds`
  arguments for selective execution (single PE type, single seed, or
  any subset of the paper configuration).
- `reproduce.py`: CPU-only verification script that regenerates every
  numerical claim, table value, and statistical test in the paper
  directly from the shipped JSON artifacts in `data/`. No GPU or
  PyTorch required.
- `generate_ads_figures.py`: regenerates all paper figures from the
  shipped JSONs. Gracefully handles partial outputs (n<3 seeds) by
  omitting error bands.
- `colab_quickstart.ipynb`: end-to-end Colab pipeline (dataset prep →
  experiments → figures → verification) organized in ten sections,
  each re-runnable independently.
- Shipped JSON artifacts under `data/`: full 3-seed × 4 PE results for
  all 11 experiments, sufficient to reproduce every paper number
  offline.

### Notes

- Trained model checkpoints are hosted on Google Drive and reused from
  the companion work [vit-positional-adversarial](https://github.com/djokobandjur/vit-positional-adversarial)
  (Zenodo Concept DOI: [`10.5281/zenodo.19154465`](https://doi.org/10.5281/zenodo.19154465)).
- This work uses a *per-buffer δ* attack convention (each of the 12
  transformer blocks receives its own perturbation), matched to the
  weight-attacks it characterizes. The companion work uses *shared-δ*
  (single perturbation broadcast across all blocks); both are
  legitimate threat models. See README §*Relationship to companion
  work* for details.
- Dual licensing: MIT for code, CC BY 4.0 for data and documentation;
  matches the Zenodo deposit metadata.
