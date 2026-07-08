# Attention Divergence Score (ADS): A Forensic Metric for Characterizing Parameter-Level Attacks in Vision Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djokobandjur/ads-vit-forensics/blob/main/colab_quickstart.ipynb)
[![Paper Status](https://img.shields.io/badge/paper-TIFS%20resubmission%20ready-green)](#paper-and-citation)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

Code, result files, figures, and reproducibility scripts for the paper
*"Attention Divergence Score: A Forensic Metric for Characterizing
Parameter-Level Attacks in Vision Transformers"*.

The repository is aligned with the final 13-page IEEE TIFS resubmission
version of the manuscript.

---

## What is this repository?

Vision Transformers are increasingly deployed in security-sensitive image
pipelines such as biometric verification, content authentication, and
medical imaging. This work studies whether parameter-level attacks leave
different attention-space forensic signatures depending on which part of
the model state is perturbed.

We introduce the **Attention Divergence Score (ADS)**, a layer-wise KL
divergence between attention distributions under clean and potentially
compromised model state. ADS is used to characterize four attack surfaces
(PE-only, Q/K/V, MLP, all-weights) across four positional-encoding (PE)
strategies: Learned, Sinusoidal, RoPE, and ALiBi.

Central findings:

1. **Differentiated forensic signature.** PE-only attacks require
   **20×–200× larger perturbation budgets** than MLP-only weight-level
   attacks to reach the 50%-of-clean accuracy criterion. This is a
   grid-limited operational saturation summary: PE-only collapse points come
   from the primary PE sweep, while the MLP-only collapse point comes from the
   specificity sweep. The ADS-per-damage ratio bifurcates by PE operation space:
   embedding-space PE (Learned, Sinusoidal) stays in the low weight-attack range,
   with Learned modestly above its MLP-only baseline, whereas attention-space PE
   (RoPE, ALiBi) is roughly an order of magnitude above the corresponding MLP-only
   baselines on ImageNet-100. The same qualitative bifurcation appears on CIFAR-100
   with dataset- and PE-dependent ratios.

2. **Operational detection boundary.** The low-budget ADS self-baseline
   trigger is an absolute-budget sensitivity diagnostic, not a deployment
   threshold. Against realistic benign distribution shifts (JPEG, blur,
   noise) on the ImageNet-100 calibration cohort, the operational ADS
   boundary is **ε ≥ 0.1 for Learned PE** and **ε ≥ 0.2 for RoPE**, where
   the threshold-grid operating AUC is at least 0.99 under that protocol; a
   rank-based sensitivity check gives the same perfect-separation boundary.

3. **Hierarchical PE fingerprint.** A layer-profile attenuation dichotomy
   separates input-injected/rotary PE (slopes roughly −0.15 to −0.26 per layer)
   from ALiBi (slope ≈ +0.01). The 12-layer ADS profile classifies PE
   types via 1-NN leave-one-out at **79.2%–95.8%** accuracy (chance = 25%).
   Single-layer L4 is useful as an operational sentinel; individual layers can also be
   highly discriminative in dataset-specific settings, so the profile should be read as a
   compact multi-layer fingerprint rather than a pointwise optimal classifier.

4. **Protocol robustness.** The ALiBi separator is validated on an
   independent CIFAR-100 canonical n=12 cohort and under grid-aware 2D
   ALiBi-style variants. The robustness-cohort ratios are within-cohort
   stress-test magnitudes and should not be read as replacements for the
   primary ImageNet-100/CIFAR-100 ratios.

ADS is best understood as a forensic characterization tool, not as a
universal early-warning detector.

---

## Repository layout

```text
ads-vit-forensics/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── colab_quickstart.ipynb
├── colab_quickstart_README.md
│
├── data/
│   ├── ads_results.json
│   ├── ads_results_cifar100.json
│   ├── ads_specificity.json
│   ├── ads_specificity_cifar.json
│   ├── ads_adaptive.json
│   ├── ads_comparison.json
│   ├── ads_threshold_fine.json
│   ├── ads_ref_evasion.json
│   ├── ads_roc_v2.json
│   ├── ads_roc_rank_auc_sensitivity.json
│   ├── ads_probing_residual.json
│   ├── ads_probing_residual_cifar.json
│   ├── ads_ref_indices.json
│   ├── ads_ref_indices_imagenet100.json
│   ├── ads_ref_indices_cifar100.json
│   ├── ads_shared_delta_imagenet100.json
│   ├── ads_shared_delta_cifar100.json
│   └── robustness/
│       ├── ads_results_cifar100_canonical_n12.json
│       ├── ads_specificity_cifar100_canonical_n12.json
│       ├── ads_probing_residual_cifar100_canonical_n12.json
│       └── *.log
│
├── figures/
│   ├── ads_fig1_l4_vs_epsilon.{png,pdf}
│   ├── ads_fig2_per_layer_heatmap.{png,pdf}
│   ├── ads_fig3_layer_profile.{png,pdf}
│   └── ads_fig4_early_warning_combined.{png,pdf}
│
├── paper/
│   └── refs_ads.bib
│
└── scripts/
    ├── full_scale_experiment.py
    ├── cifar100_experiment.py
    ├── ads_experiment.py
    ├── ads_experiment_cifar.py
    ├── ads_specificity.py
    ├── ads_specificity_cifar.py
    ├── ads_adaptive.py
    ├── ads_comparison.py
    ├── ads_threshold_fine.py
    ├── ads_ref_evasion.py
    ├── ads_roc_analysis.py
    ├── compute_roc_rank_auc_sensitivity.py
    ├── ads_probing_residual.py
    ├── ads_probing_residual_cifar.py
    ├── ads_shared_delta_attack_convention.py
    ├── generate_ads_figures.py
    ├── reproduce.py
    └── robustness/
        ├── README_canonical_protocol_robustness_n12.md
        ├── ads_experiment_cifar_protocol_robustness_n12.py
        ├── ads_specificity_cifar_protocol_robustness_n12.py
        ├── ads_probing_residual_cifar_protocol_robustness_n12.py
        └── auxiliary protocol-robustness scripts
```

`data/` contains archived JSON outputs used by the paper. `scripts/` contains
GPU experiment scripts plus CPU-only verification and figure generation. The
`ads_shared_delta_imagenet100.json` and `ads_shared_delta_cifar100.json`
reproduce the tied-buffer shared-δ control columns in the attack-convention
table. `ads_roc_rank_auc_sensitivity.json` is a derived post-processing artifact from `ads_roc_v2.json` that compares exact single-positive rank AUCs with the stored threshold-grid operating AUCs. The `data/robustness/` and `scripts/robustness/` folders are auxiliary:
they support the independent CIFAR-100 canonical n=12 protocol-robustness
analysis and are not part of the primary n=6 ImageNet-100/CIFAR-100 sweep.

---

## Checkpoint layout

Model checkpoints are **not redistributed in this repository**. They are hosted
separately on Google Drive and should be passed to scripts via `--models_dir`.

- **Checkpoint root:** [Google Drive — `ads-vit-forensics` checkpoints](https://drive.google.com/drive/folders/1UojvGk3oeoQui7jy8DSFM_0U6Xkwx-IB?usp=drive_link)

Each script expects the checkpoint folders directly inside the corresponding
`--models_dir`, so use the dataset-specific subfolder rather than the parent
checkpoint root.

Example Google Drive layout used for the paper runs:

```text
ads-vit-forensics/                    # Drive folder with checkpoints, not the GitHub repo clone
├── ImageNet100/
│   ├── learned_seed42/best_model.pth
│   ├── learned_seed123/best_model.pth
│   ├── ...
│   └── alibi_seed1213/best_model.pth
│
├── CIFAR100/
│   ├── learned_seed42/best_model.pth
│   ├── learned_seed123/best_model.pth
│   ├── ...
│   └── alibi_seed1213/best_model.pth
│
└── CIFAR100_canonical/
    ├── alibi_seed1/best_model.pth
    ├── alibi_seed5/best_model.pth
    ├── ...
    ├── alibi_2d_seed31337/best_model.pth
    └── alibi_2d_matched_seed31337/best_model.pth
```

Therefore, use paths such as:

```text
/path/to/ImageNet100_checkpoints
/path/to/CIFAR100_checkpoints
/path/to/CIFAR100_canonical_checkpoints
```

Do **not** pass the parent folder that contains all three checkpoint groups as
`--models_dir`; pass the dataset-specific checkpoint folder.

---

## Quick verification and figure regeneration

The fastest reproducibility check does not require GPU or checkpoint files. It verifies the primary numerical tables, figure inputs, and statistical tests; the shared-δ attack-convention columns are backed by the archived shared-δ JSON files and can be regenerated with the GPU script below:

```bash
pip install -r requirements.txt
python scripts/reproduce.py --data-dir data --output-dir output --no-figures
```

To regenerate the paper figures from the archived JSON files:

```bash
python scripts/generate_ads_figures.py --data-dir data --output-dir .
```

To regenerate the ROC rank-AUC sensitivity artifact from the archived ROC JSON:

```bash
python scripts/compute_roc_rank_auc_sensitivity.py \
  --roc-path data/ads_roc_v2.json \
  --output-path data/ads_roc_rank_auc_sensitivity.json
```

The figure script writes Fig. 1–4 files to `figures/`. The Fig. 2 heatmap uses
a display-only fix for the ε=0 column: exact-zero ADS values are rendered at the
lower log-scale limit because `LogNorm` cannot display non-positive values.

---

## Full pipeline

For a complete end-to-end run, use the Colab notebook:

```text
colab_quickstart.ipynb
```

The notebook contains the detailed workflow for mounting Drive, setting
checkpoint paths, running the GPU experiments, generating JSON files,
regenerating figures, running `reproduce.py`, and zipping final artifacts.
The README intentionally keeps CLI examples short; the notebook is the
recommended place for the full step-by-step pipeline.

---

## Minimal CLI examples

All experiment scripts use the same path-oriented interface:

```bash
python scripts/ads_experiment.py \
  --models_dir "/path/to/ImageNet100_checkpoints" \
  --val_dir "/path/to/imagenet100/val" \
  --output_path "data/ads_results.json"
```

```bash
python scripts/ads_experiment_cifar.py \
  --models_dir "/path/to/CIFAR100_checkpoints" \
  --val_dir "/tmp/cifar100" \
  --output_path "data/ads_results_cifar100.json"
```

The optional `--pe_types` and `--seeds` arguments can restrict a run to a subset:

```bash
python scripts/ads_experiment.py \
  --models_dir "/path/to/ImageNet100_checkpoints" \
  --val_dir "/path/to/imagenet100/val" \
  --output_path "output/smoke_test.json" \
  --pe_types learned \
  --seeds 42
```

Primary n=6 seeds are:

```text
42 123 456 789 1011 1213
```

---

## Shared-δ attack-convention control

The attack-convention table in the paper compares the primary independent
per-buffer PGD-PE sweep with a tied-buffer **shared-δ** control. The shared-δ
control is regenerated by:

```bash
python scripts/ads_shared_delta_attack_convention.py \
  --models_dir "/path/to/ImageNet100_checkpoints" \
  --val_dir "/path/to/imagenet100/val" \
  --dataset imagenet \
  --output_path "data/ads_shared_delta_imagenet100.json"
```

```bash
python scripts/ads_shared_delta_attack_convention.py \
  --models_dir "/path/to/CIFAR100_checkpoints" \
  --val_dir "/tmp/cifar100" \
  --dataset cifar \
  --output_path "data/ads_shared_delta_cifar100.json"
```

These two JSON files contain seed-level clean and attacked accuracies for the
same six seeds and budget grid used in the paper:

```text
ε = 0.05, 0.1, 0.2, 0.5, 1.0
```

The attack metadata records the pattern as `shared_delta_all_12_blocks`:
gradients are aggregated across transformer blocks for each PE buffer name and
one shared perturbation tensor is applied to all replicated buffers. For RoPE, this ties
the replicated `cos_cached` and `sin_cached` cache perturbations; implementations that expose
`inv_freq` tie it by the same per-buffer-name rule and record it in the sanity fields. These
results are a tied-buffer control only; the main forensic analysis and
ADS-per-damage ratios use the independent per-buffer convention.

---

## Protocol-robustness cohort

The independent canonical CIFAR-100 n=12 analysis is stored separately from the
primary data and scripts:

```text
data/robustness/
scripts/robustness/
```

The three publication-relevant robustness outputs are:

```text
data/robustness/ads_results_cifar100_canonical_n12.json
data/robustness/ads_specificity_cifar100_canonical_n12.json
data/robustness/ads_probing_residual_cifar100_canonical_n12.json
```

Generic example:

```bash
python scripts/robustness/ads_experiment_cifar_protocol_robustness_n12.py \
  --models_dir "/path/to/CIFAR100_canonical_checkpoints" \
  --val_dir "/tmp/cifar100" \
  --output_path "data/robustness/ads_results_cifar100_canonical_n12.json"
```

Canonical n=12 seeds are:

```text
1 5 7 11 13 21 42 99 123 456 2024 31337
```

Canonical conditions are:

```text
alibi              # 1D-ALiBi canonical
alibi_2d           # grid-aware 2D-ALiBi-style canonical
alibi_2d_matched   # matched 2D-ALiBi-style variant
```

These results are within-cohort stress-test magnitudes. They should not be
compared directly to, or read as replacements for, the primary n=6 ImageNet-100
or CIFAR-100 ratios.

---

## Reading the reported numbers

A few protocol distinctions are important for interpreting the paper:

- **PE state vs ordinary model weights.** For Learned PE, the attacked tensor
  has `197 × 768 = 151,296` entries, less than 0.2% of ViT-Base parameters.
  For RoPE and ALiBi, the attacked objects are non-trainable positional buffers
  or slopes rather than ordinary learned weights.

- **ROC operating AUC.** The ROC table reports a threshold-grid operating
  AUC from `ads_roc_v2.json`. For each seed and attack budget, the positive
  side is one attacked ADS score and the negative side contains 29 clean/benign
  scores. The derived `ads_roc_rank_auc_sensitivity.json` file compares this
  stored operating AUC with exact single-positive rank AUC. The high-confidence
  boundary is unchanged: Learned reaches AUC = 1.0 at ε = 0.1 and RoPE at
  ε = 0.2 by both estimators.

- **Self-baseline ADS trigger vs operational detection.** The 10× self-baseline
  ADS trigger is useful for sensitivity analysis, but it is not a
  deployment-ready threshold. The interpolated trigger comes from
  `ads_threshold_fine.json`, whereas operational detection is evaluated against benign
  shifts through the ImageNet-100 ROC analysis for Learned PE and RoPE.

- **Shared-δ vs per-buffer.** Shared-δ is a tied-buffer control and can be
  stronger for some PE types and ε ranges. The archived files
  `ads_shared_delta_imagenet100.json` and `ads_shared_delta_cifar100.json`
  reproduce the shared-δ columns in the attack-convention table. The primary
  forensic comparison uses independent per-buffer perturbations to match the
  granularity of weight-level attacks.

- **ADS-per-damage ratio.** The ratio `ADS(L4) / accuracy-drop` is useful for
  comparing attack surfaces at the same budget, but it becomes less informative
  when accuracy drop is extremely small. The manuscript reports absolute ADS,
  accuracy drop, and benign-calibrated ROC results alongside this ratio.

---

## Data and checkpoint notes

- ImageNet-100 validation images are not redistributed and remain subject to the
  ImageNet terms of access.
- CIFAR-100 is downloaded by the CIFAR scripts through `torchvision` when needed.
- Fixed reference-set indices are shipped under `data/` for exact reproduction of
  ADS reference measurements. The reference/adaptive evasion experiments are stress tests
  on the 256-image reference protocol; they do not implement the stronger split-objective
  attacker that optimizes CE on the validation complement while regularizing ADS only on
  the reference set.
- LaTeX sources reference PDF figures; regenerated figures should include the PDF versions
  alongside the PNG previews.
- Shared-δ tied-buffer controls are shipped under `data/` as
  `ads_shared_delta_imagenet100.json` and `ads_shared_delta_cifar100.json`;
  they are not replacements for the primary per-buffer PE-only sweep.
- `ads_roc_rank_auc_sensitivity.json` is a derived verification artifact, not a
  new model experiment. It is regenerated from `ads_roc_v2.json` by
  `scripts/compute_roc_rank_auc_sensitivity.py`.
- A whole-checkpoint hash that includes PE tensors would detect Learned-PE
  `pos_embed` tampering. The PE-only threat model concerns settings where
  PE/adapters are supplied separately, excluded from a main-weight hash,
  legitimately updated after hashing, or controlled by the same supply-chain
  actor that controls verification metadata.
- Checkpoint files are external artifacts and are intentionally not committed to
  this repository.

---

## Paper and citation

The paper is prepared for resubmission to IEEE Transactions on Information
Forensics and Security. Citation details will be updated once the manuscript
status is finalized.

For citing this repository, use:

```bibtex
@software{bandjur2026ads,
  author       = {Bandjur, Djoko and Bandjur, Milos},
  title        = {{ADS: Attention Divergence Score — A Forensic
                   Metric for Characterizing Parameter-Level Attacks
                   in Vision Transformers}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19844729},
  url          = {https://github.com/djokobandjur/ads-vit-forensics}
}
```

---

## License

This repository uses a dual-licensing scheme:

- **Source code** (`scripts/*.py`, `scripts/robustness/*.py`, and
  `colab_quickstart.ipynb`) is released under the **MIT License**; see
  [`LICENSE`](LICENSE).

- **Result files and documentation** (`data/`, `figures/`, README, CHANGELOG,
  and the Zenodo deposit) are released under the **Creative Commons Attribution
  4.0 International License** (CC BY 4.0).

- **ImageNet-100 data** are not redistributed and remain subject to the ImageNet
  terms of access.

If you use the code, cite the repository under MIT terms. If you use the
results or figures in a derivative work, cite under CC BY 4.0 terms.
