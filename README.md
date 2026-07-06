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
   attacks to reach the 50%-of-clean accuracy criterion. The ADS-per-damage
   ratio bifurcates by PE operation space: it is comparable to weight-attack
   baselines for embedding-space PE (Learned, Sinusoidal), but roughly an
   order of magnitude above the corresponding MLP-only baselines for
   attention-space PE (RoPE, ALiBi) on ImageNet-100. The same qualitative
   bifurcation appears on CIFAR-100 with dataset- and PE-dependent ratios.

2. **Operational detection boundary.** The low-budget ADS self-baseline
   trigger is an absolute-budget sensitivity diagnostic, not a deployment
   threshold. Against realistic benign distribution shifts (JPEG, blur,
   noise), the operational ADS boundary is **ε ≥ 0.1 for Learned PE** and
   **ε ≥ 0.2 for RoPE**, where ROC AUC is at least 0.99.

3. **Hierarchical PE fingerprint.** A layer-profile attenuation dichotomy
   separates input-injected/rotary PE (average slope ≈ −0.22 per layer)
   from ALiBi (slope ≈ +0.01). The full 12-layer ADS profile classifies PE
   types via 1-NN leave-one-out at up to **95.8%** accuracy (chance = 25%).
   Single-layer L4 is useful as an operational sentinel, but the full
   profile is the stronger fingerprint.

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
│   ├── ads_probing_residual.json
│   ├── ads_probing_residual_cifar.json
│   ├── ads_ref_indices.json
│   ├── ads_ref_indices_imagenet100.json
│   ├── ads_ref_indices_cifar100.json
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
    ├── ads_probing_residual.py
    ├── ads_probing_residual_cifar.py
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
`data/robustness/` and `scripts/robustness/` folders are auxiliary: they support
the independent CIFAR-100 canonical n=12 protocol-robustness analysis and are
not part of the primary n=6 ImageNet-100/CIFAR-100 sweep.

---

## Checkpoint layout

Model checkpoints are **not redistributed in this repository**. They are hosted
separately on Google Drive and should be passed to scripts via `--models_dir`.

- **Checkpoint root:** [Google Drive — `ads-vit-forensics` checkpoints]([PASTE_GOOGLE_DRIVE_FOLDER_LINK_HERE](https://drive.google.com/drive/folders/1UojvGk3oeoQui7jy8DSFM_0U6Xkwx-IB?usp=drive_link))

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

The fastest reproducibility check does not require GPU or checkpoint files:

```bash
pip install -r requirements.txt
python scripts/reproduce.py --data-dir data --output-dir output --no-figures
```

To regenerate the paper figures from the archived JSON files:

```bash
python scripts/generate_ads_figures.py --data-dir data --output-dir .
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

- **Self-baseline ADS trigger vs operational detection.** The 10× self-baseline
  ADS trigger is useful for sensitivity analysis, but it is not a
  deployment-ready threshold. Operational detection is evaluated against benign
  shifts through the ROC analysis.

- **Shared-δ vs per-buffer.** Shared-δ is a tied-buffer control and can be
  stronger for some PE types and ε ranges. The primary forensic comparison uses
  independent per-buffer perturbations to match the granularity of weight-level
  attacks.

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
  ADS reference measurements.
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
