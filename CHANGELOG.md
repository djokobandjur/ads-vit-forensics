# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — 2026-09-16 post-Phase-2 canonical release candidate

### Added

- Byte-identical Phase-2 P2-C step-horizon external audit and closure decision, with a SHA-bound pointer to the exact post-hoc trajectory-dynamics memo in the archival payload.
- SHA-bound public pointer to the byte-identical P2-D targeted-structural confirmatory external audit and closure decision carried in the archival payload.
- Final canonical adaptive n=6 external audit in public provenance.
- Final manuscript v1.5.2 closure audit (12-page main + 6-page supplement).
- Machine-readable post-Phase-2 release manifest, artifact index, explicit archival-gap boundary, GitHub release notes, and prepared Zenodo v3.0.0 metadata.

### Changed

- Scientific-authority pointer advanced from Ledger v1.31 to Ledger v1.62 (`c3cc0a791517671e1e819c7bbd54518236d61a5a80bb4ec0a452b8f4ee4872d1`).
- Repository status advanced from pre-Zenodo v1.31 freeze to post-Phase-2 release candidate.
- Reproducibility wording now distinguishes compact public payloads from unrecovered or separately hash-retained execution artifacts.
- Zenodo handoff now targets a new version under concept DOI `10.5281/zenodo.19844729`; no version-specific DOI is claimed before publication.

### Scientific status

No experiment, manuscript scientific number, attack result, interpretation, or attribution decision was created by this release-packaging pass. Historical and superseded states remain preserved rather than overwritten.

## [Unreleased] — 2026-09-13 pre-Zenodo freeze

### Added

- Final release/provenance manifest with the frozen manuscript, submission-wrapper, replication-package, and scientific-authority hashes.
- Final release-freeze status and Zenodo handoff instructions.
- `CITATION.cff` for GitHub citation support without inventing a version-specific DOI.
- Recovered EXP-019 canonical artifact identities and final independent numerical-gate results in the public provenance documentation.

### Changed

- Scientific-authority pointer advanced to Ledger v1.31 (`50efb67906fa9bc8058c3843cd2036dce25169e0fdf7720b190271881423a9b4`).
- Repository status advanced from synchronization-in-progress to pre-Zenodo release ready.
- Historical draft PR #1 classified as superseded rather than a merge target.

### Scientific status

No manuscript number, attack result, interpretation, or attribution decision changes in this repository-release pass.

## [2.1.0] — 2026-07-06

Final repository update for the 13-page IEEE TIFS resubmission of
*"Attention Divergence Score: A Forensic Metric for Characterizing
Parameter-Level Attacks in Vision Transformers"*.

### Changed

- Updated the README to match the final TIFS resubmission narrative and numbers:
  20×--200× saturation-budget asymmetry, AUC ≥ 0.99 operational detection
  boundary for Learned PE / RoPE, 95.8% full-profile PE fingerprinting, and
  explicit PE-state terminology for RoPE/ALiBi buffers and slopes.
- Updated `generate_ads_figures.py` to the final figure-generation script.
  In particular, Fig. 2 now renders exact-zero ADS values at the lower log-scale
  limit instead of using Matplotlib's default masked-value color for `LogNorm`.
- Updated `reproduce.py` for the final primary n=6 protocol with seeds
  `42, 123, 456, 789, 1011, 1213`.
- Updated the shared-δ vs per-buffer wording in repository documentation to avoid
  implying close agreement across all ε values. The shared-δ sweep is documented
  as a tied-buffer control; the per-buffer sweep is the primary protocol.
- Updated the adaptive/reference-evasion wording to match the final sign convention
  used in the manuscript: maximize CE − λ·ADS under PGD ascent.
- Clarified that ADS is a parameter-free metric definition, while deployment still
  requires design choices such as reference-set selection, layer choice, and
  benign-calibrated thresholds.

### Added

- Final 13-page submission PDF and manuscript source under `paper/` in the
  GitHub/submission update package.
- Final Fig. 1--Fig. 4 PDF and PNG artifacts under `output/figures/`.
- `REPO_UPDATE_AND_SUBMISSION_CHECKLIST.md` with local sanity-check commands for
  README, requirements, reproduction, figure generation, and final PDF submission.
- Optional verification hooks in `reproduce.py` for the independent CIFAR-100
  ALiBi-style protocol-robustness cohort when the canonical n=12 JSON files are
  present.

### Fixed

- Removed stale n=3 expectations from `reproduce.py`; primary statistics now treat
  seed as the unit of independence with n=6.
- Replaced the old 17×--200× repository phrasing with the final 20×--200×
  ImageNet/CIFAR saturation-budget range.
- Corrected the Learned/ImageNet collapse budget to ε*=0.300 and the CIFAR ALiBi
  multiplier to 30× wherever those quantities are summarized.
- Removed the undefined `TFS` shorthand from repository-facing documentation.
- Avoided the older claim that weight-attack trajectories are visually demonstrated
  in Fig. 4; weight-level contrast is summarized quantitatively in the specificity
  table.

## [2.0.0] — 2026-05-18

Code, data artifacts, and reproducibility scripts for the paper
*"Attention Divergence Score: A Forensic Metric for Characterizing
Parameter-Level Attacks in Vision Transformers"*, prepared for IEEE
TIFS resubmission.

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
- Shipped JSON artifacts under `data/`: full seed × PE results for
  all experiments, sufficient to reproduce paper numbers offline.

### Notes

- Trained model checkpoints are hosted on Google Drive and reused from
  the companion work [vit-positional-adversarial](https://github.com/djokobandjur/vit-positional-adversarial)
  (Zenodo Concept DOI: [`10.5281/zenodo.19154465`](https://doi.org/10.5281/zenodo.19154465)).
- This work uses a *per-buffer δ* attack convention matched to the
  weight-attacks it characterizes. The companion work uses *shared-δ*
  as a distinct tied-buffer threat model.
- Dual licensing: MIT for code, CC BY 4.0 for data and documentation;
  matches the Zenodo deposit metadata.
