# Attention Divergence Score (ADS): A Forensic Metric for Characterizing Parameter-Level Attacks in Vision Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djokobandjur/ads-vit-forensics/blob/main/colab_quickstart.ipynb)
[![Paper Status](https://img.shields.io/badge/paper-TIFS%20resubmission%20ready-green)](#paper-and-citation)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

Code, result files, and reproducibility scripts for the paper
*"Attention Divergence Score: A Forensic Metric for Characterizing
Parameter-Level Attacks in Vision Transformers"*.

The repository is aligned with the final 13-page IEEE TIFS resubmission
version of the manuscript.

---

## What is this repository?

Vision Transformers are increasingly deployed in security-sensitive image
pipelines such as biometric verification, content authentication, and
medical imaging. Existing detection literature has largely treated
parameter-level attacks homogeneously, applying a single threshold or
summary statistic regardless of which architectural component is
perturbed. This work questions that homogeneity.

We introduce the **Attention Divergence Score (ADS)**, a layer-wise KL
divergence between attention distributions under clean and potentially
compromised model state. ADS is used to characterize four attack surfaces
(PE-only, Q/K/V, MLP, all-weights) across four PE strategies (Learned,
Sinusoidal, RoPE, ALiBi) on ViT-Base models trained on ImageNet-100 and
CIFAR-100.

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
   noise), the operational ADS boundary is
   **ε ≥ 0.1 for Learned PE** and **ε ≥ 0.2 for RoPE**, where ROC AUC is
   at least 0.99. Adaptive and reference-set evasion tests show that
   systematic stealth is confined to low-damage or near-threshold regimes.

3. **Hierarchical PE fingerprint.** A layer-profile attenuation dichotomy
   separates input-injected/rotary PE (average slope ≈ −0.22 per layer)
   from ALiBi (slope ≈ +0.01). The full 12-layer ADS profile classifies PE
   types via 1-NN leave-one-out at up to **95.8%** accuracy (chance = 25%).
   Single-layer L4 is useful as an operational sentinel, but the full
   profile is the stronger fingerprint.

4. **Protocol robustness.** The ALiBi separator is validated on an
   independent CIFAR-100 cohort with 12 checkpoints and under grid-aware
   2D ALiBi-style variants. The robustness-cohort ratios are within-cohort
   stress-test magnitudes and should not be read as replacements for the
   primary ImageNet-100/CIFAR-100 ratios.

ADS is best understood as a forensic characterization tool, not as a
universal early-warning detector.

---

## Relationship to companion work

This work uses the same overall ViT-Base training setup as the companion
repository [vit-positional-adversarial](https://github.com/djokobandjur/vit-positional-adversarial)
(Concept DOI: [`10.5281/zenodo.19154465`](https://doi.org/10.5281/zenodo.19154465)).
The companion work studies the PE attack surface as a *performance
robustness problem*; the present work reframes PE perturbation as a
*forensic monitoring problem*.

**Attack convention note.** The companion work uses a *shared-δ* attack
convention, where a single perturbation is tied across replicated PE
buffers. This work uses the *per-buffer independent δᵢ* convention as the
primary protocol, because weight-level attack surfaces (Q/K/V, MLP,
all-weights) operate on per-block parameters by definition. The manuscript
reports the shared-δ convention separately as a tied-buffer control; it is
not treated as a replacement for the primary per-buffer forensic protocol.

---

## Repository layout

```text
ads-vit-forensics/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
│
├── ads_experiment.py              # Main PE-only PGD attack + ADS (ImageNet-100)
├── ads_experiment_cifar.py        # Main PE-only PGD attack + ADS (CIFAR-100)
├── ads_specificity.py             # 4 attack surfaces × 4 PE (ImageNet-100)
├── ads_specificity_cifar.py       # 4 attack surfaces × 4 PE (CIFAR-100)
├── ads_adaptive.py                # Adaptive PGD with ADS-minimization objective
├── ads_comparison.py              # Shared-δ vs per-buffer attack-convention comparison
├── ads_threshold_fine.py          # Fine ε grid + interpolated self-baseline trigger
├── ads_ref_evasion.py             # Reference-set evasion experiment
├── ads_probing_residual.py        # Residual-stream probing (ImageNet-100)
├── ads_probing_residual_cifar.py  # Residual-stream probing (CIFAR-100)
├── ads_roc_analysis.py            # ROC AUC vs benign shifts
│
├── reproduce.py                   # CPU-only verification of paper claims
├── generate_ads_figures.py        # Figure generation from JSON results
│
├── colab_quickstart.ipynb         # End-to-end Colab pipeline notebook
│
├── data/                          # Archived JSON result files
│   ├── ads_results.json
│   ├── ads_results_cifar100.json
│   ├── ads_specificity.json
│   ├── ads_specificity_cifar.json
│   ├── ads_adaptive.json
│   ├── ads_comparison.json
│   ├── ads_threshold_fine.json
│   ├── ads_ref_evasion.json
│   ├── ads_ref_indices.json
│   ├── ads_probing_residual.json
│   ├── ads_probing_residual_cifar.json
│   ├── ads_roc_v2.json
│   ├── ads_results_cifar100_canonical_n12.json
│   ├── ads_specificity_cifar100_canonical_n12.json
│   ├── ads_probing_residual_cifar100_canonical_n12.json
│   └── additional auxiliary files
│
├── output/figures/                # Generated by generate_ads_figures.py
│
└── paper/
    ├── main.pdf                   # Final 13-page TIFS resubmission PDF
    ├── main.tex                   # Manuscript source used to build main.pdf
    └── refs_ads.bib
```

### Script-to-output mapping

| Script | Purpose | Output |
| --- | --- | --- |
| `ads_experiment.py` | Main ADS PE-only attack on ImageNet-100; 4 PE × 6 seeds × ε grid; logs accuracy, ADS(mean), ADS(L4), and per-layer ADS | `ads_results.json` |
| `ads_experiment_cifar.py` | Same on CIFAR-100 | `ads_results_cifar100.json` |
| `ads_specificity.py` | 4 attack surfaces (PE-only, QKV-only, MLP-only, all-weights) × 4 PE on ImageNet-100; same-budget ADS/drop specificity analysis | `ads_specificity.json` |
| `ads_specificity_cifar.py` | Same on CIFAR-100 | `ads_specificity_cifar.json` |
| `ads_adaptive.py` | Adaptive PGD attacker that maximizes `CE − λ·ADS` under PGD ascent to test ADS-aware evasion | `ads_adaptive.json` |
| `ads_comparison.py` | Shared-δ versus per-buffer PE attack-convention comparison | `ads_comparison.json` |
| `ads_threshold_fine.py` | Fine ε grid with log-log interpolation of the 10× self-baseline ADS crossing | `ads_threshold_fine.json` |
| `ads_ref_evasion.py` | Reference-set evasion: attacker knows the 256 reference images and optimizes `CE − λ·ADS(ref)` | `ads_ref_evasion.json` |
| `ads_probing_residual.py` | Layer-wise positional probing on residual-stream activations for ImageNet-100 | `ads_probing_residual.json` |
| `ads_probing_residual_cifar.py` | Same on CIFAR-100 | `ads_probing_residual_cifar.json` |
| `ads_roc_analysis.py` | ROC AUC of ADS detector against realistic benign shifts (JPEG, blur, noise) | `ads_roc_v2.json` |
| `reproduce.py` | CPU-only verification; regenerates numerical claims, tables, and statistical tests from JSON data with PASS/FAIL tolerances | Console output + `output/tables/` |
| `generate_ads_figures.py` | Generates Fig. 1–4 from JSON result files; renders ε=0 heatmap entries at the lower log-scale limit for display | `output/figures/*.png`, `output/figures/*.pdf` |

---

## Trained models

The primary evaluation uses ViT-Base checkpoints for four PE types
(Learned, Sinusoidal, RoPE, ALiBi), six seeds, and two datasets
(ImageNet-100 and CIFAR-100). The checkpoints are hosted on Google Drive
and documented through the companion repository:

- **ImageNet-100 models:** [drive.google.com/.../1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9](https://drive.google.com/drive/folders/1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9)
- **CIFAR-100 models:** [drive.google.com/.../1HBiOjNfuRsh2H0ZGRP4rIdBeydedCBJL](https://drive.google.com/drive/folders/1HBiOjNfuRsh2H0ZGRP4rIdBeydedCBJL)

For the model architecture, training procedure, and clean-accuracy
benchmarks, see the companion repository
[vit-positional-adversarial](https://github.com/djokobandjur/vit-positional-adversarial)
(Concept DOI: [`10.5281/zenodo.19154465`](https://doi.org/10.5281/zenodo.19154465)).

The folder structure expected by the scripts is:

```text
<models_dir>/
├── learned_seed42/best_model.pth
├── learned_seed123/best_model.pth
├── learned_seed456/best_model.pth
├── learned_seed789/best_model.pth
├── learned_seed1011/best_model.pth
├── learned_seed1213/best_model.pth
├── ...
└── alibi_seed1213/best_model.pth
```

---

## Reproducing results

### Verification only (no GPU needed)

To verify the numerical claims, table values, and statistical tests in the
paper directly from the archived JSON files:

```bash
pip install -r requirements.txt
python reproduce.py
```

This runs on CPU, requires no model checkpoints, and prints PASS/FAIL with
explicit tolerances for each claim. Output tables are written to
`output/tables/`.

### Selective execution (single PE and/or seed, GPU needed)

To run any experiment script for a specific PE type and/or seed instead of
the full paper configuration, use the optional `--pe_types` and `--seeds`
CLI arguments:

```bash
python ads_experiment.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "output/partial.json" \
    --pe_types learned --seeds 42
```

Argument reference:

```text
--pe_types  Optional. PE types to evaluate. If omitted, uses the
            hardcoded PE_TYPES list defined in the script. Values:
            learned, sinusoidal, rope, alibi.

--seeds     Optional. Random seeds to evaluate. If omitted, uses the
            hardcoded SEEDS list defined in the script. Paper seeds:
            42, 123, 456, 789, 1011, 1213.
```

The resulting JSON contains only the requested `(PE, seed)` cells but
shares the same structure as the full output. `generate_ads_figures.py`
handles partial outputs gracefully where possible. `reproduce.py` is for
paper-number verification and requires the full JSON files shipped under
`data/`.

### Full pipeline (GPU needed)

The fastest path is the [Colab Quickstart notebook](colab_quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/djokobandjur/ads-vit-forensics/blob/main/colab_quickstart.ipynb)).
It clones this repo, mounts Drive for model and dataset access, and walks
through the full pipeline: dataset preparation, ADS experiments,
specificity, adaptive attacker, probing, ROC, figures, and verification.
Each section can be re-run independently if prior JSON outputs are present.

### Manual CLI workflow

Each script supports a consistent `--models_dir / --val_dir / --output_path`
interface. The command sequence below assumes Colab-style paths and runs
the full paper configuration.

**1. Main ADS PE-only attack**

```bash
python ads_experiment.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "data/ads_results.json"

python ads_experiment_cifar.py \
    --models_dir "/path/to/Trained models_CIFAR100" \
    --output_path "data/ads_results_cifar100.json"
```

**2. Specificity: 4 attack surfaces × 4 PE**

```bash
python ads_specificity.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "data/ads_specificity.json"

python ads_specificity_cifar.py \
    --models_dir "/path/to/Trained models_CIFAR100" \
    --output_path "data/ads_specificity_cifar.json"
```

**3. Adaptive attacker**

```bash
python ads_adaptive.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "data/ads_adaptive.json"
```

**4. Threshold and reference-evasion experiments**

```bash
python ads_threshold_fine.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "data/ads_threshold_fine.json"

python ads_ref_evasion.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "data/ads_ref_evasion.json"
```

**5. Residual-stream probing**

```bash
python ads_probing_residual.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "data/ads_probing_residual.json"

python ads_probing_residual_cifar.py \
    --models_dir "/path/to/Trained models_CIFAR100" \
    --output_path "data/ads_probing_residual_cifar.json"
```

**6. ROC analysis**

```bash
python ads_roc_analysis.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "data/ads_roc_v2.json"
```

**7. Attack-convention comparison**

```bash
python ads_comparison.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "data/ads_comparison.json"
```

**8. Generate figures**

```bash
python generate_ads_figures.py \
    --data-dir data \
    --output-dir output
```

**9. Verify**

```bash
python reproduce.py
```

---

## Reading the reported numbers

A few protocol distinctions are important for interpreting the paper and
for comparing this repository with the companion work.

- **PE state vs ordinary model weights.** For Learned PE, the attacked
  tensor has `197 × 768 = 151,296` entries, less than 0.2% of ViT-Base
  parameters. For RoPE and ALiBi, the attacked objects are non-trainable
  positional buffers or slopes rather than ordinary learned weights.

- **Self-baseline ADS trigger vs operational detection.** The 10×
  self-baseline ADS trigger is useful for sensitivity analysis, but it is
  not a deployment-ready threshold. Operational detection is evaluated
  against benign shifts through the ROC analysis.

- **Shared-δ vs per-buffer.** Shared-δ is a tied-buffer control and can be
  stronger for some PE types and ε ranges. The primary forensic comparison
  uses independent per-buffer perturbations to match the granularity of
  weight-level attacks.

- **ADS-per-damage ratio.** The ratio `ADS(L4) / accuracy-drop` is useful
  for comparing attack surfaces at the same budget, but it becomes less
  informative when accuracy drop is extremely small. The manuscript reports
  absolute ADS, accuracy drop, and benign-calibrated ROC results alongside
  this ratio.

- **Figure 2 heatmap.** The ε=0 column is rendered at the lower log-scale
  limit for display, because exact zero ADS cannot be shown directly with
  `LogNorm`.

---

## Adapting for local execution

All experiment scripts accept their input and output paths as CLI arguments.
Adapt them to your local filesystem by changing `--models_dir`, `--val_dir`,
and `--output_path`. No code changes are required for ordinary local runs.

Practical notes:

- **ImageNet-100 dataset structure** follows the standard ImageFolder
  layout: one subdirectory per class, with image files inside. Pass the
  parent directory as `--val_dir`.

- **CIFAR-100** requires no manual dataset setup. CIFAR scripts invoke
  `torchvision.datasets.CIFAR100(download=True)` and cache locally.

- **Reference image indices** (`ads_ref_indices.json`) are shipped in
  `data/` for exact reproducibility. Most scripts read this file directly;
  if absent, they regenerate it using a fixed seed. Keep the shipped file
  to reproduce paper numbers exactly.

- **`reproduce.py` requires no GPU and no PyTorch.** It reads only the
  archived JSON files and uses `numpy`, `scipy`, and `pandas` for
  statistical verification.

---

## Paper and citation

The paper is prepared for resubmission to IEEE Transactions on Information
Forensics and Security. Citation details will be added once the manuscript
status is finalized.

For citing this repository, use the Zenodo deposit:

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

When citing the companion work, use:

```bibtex
@software{bandjur2026adversarial,
  author       = {Bandjur, Djoko and Bandjur, Milos},
  title        = {{Adversarial Vulnerability of Positional Encoding
                   in Vision Transformers}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19154465},
  url          = {https://github.com/djokobandjur/vit-positional-adversarial}
}
```

---

## License

This repository uses a dual-licensing scheme:

- **Source code** (`.py` files and `colab_quickstart.ipynb`) is released
  under the **MIT License**; see [`LICENSE`](LICENSE).

- **Result files and documentation** (`data/`, generated figures,
  README, CHANGELOG, and the Zenodo deposit) are released under the
  **Creative Commons Attribution 4.0 International License** (CC BY 4.0).

- **Trained model checkpoints** are hosted externally and released under
  CC BY 4.0 for research purposes by the companion work. ImageNet-100
  models are derivative artifacts of ImageNet-1k and remain subject to the
  [ImageNet terms of access](https://www.image-net.org/download.php).

- **ImageNet-1k validation images** required to reproduce the ImageNet-100
  experiments are governed by the ImageNet terms of access and are not
  redistributed in this repository.

If you use the code, cite the repository under MIT terms. If you use the
results, figures, or trained models in a derivative work, cite under
CC BY 4.0 terms.
