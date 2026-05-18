# Attention Divergence Score (ADS): A Forensic Metric for Characterizing Parameter-Level Attacks in Vision Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djokobandjur/ads-vit-forensics/blob/main/colab_quickstart.ipynb)
[![Paper Status](https://img.shields.io/badge/paper-resubmission-orange)](#paper-and-citation)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

Code, result files, and reproducibility scripts for the paper
*"Attention Divergence Score: A Forensic Metric for Characterizing
Parameter-Level Attacks in Vision Transformers"* (under resubmission to
IEEE TIFS).

---

## What is this repository?

Vision Transformers are increasingly deployed in security-sensitive image
pipelines (biometric verification, content authentication, medical imaging).
Existing detection literature has largely treated parameter-level attacks
homogeneously, applying a single threshold regardless of which architectural
component is perturbed. This work questions that homogeneity.

We introduce the **Attention Divergence Score (ADS)** — the KL divergence
between attention distributions under clean and potentially compromised
parameters — and use it to characterize four attack surfaces (PE, Q/K/V,
MLP, all-weights) across four PE strategies (Learned, Sinusoidal, RoPE,
ALiBi) on 24 ViT-Base models trained on ImageNet-100 and CIFAR-100.

Central findings:

1. **Differentiated forensic signature.** PE-only attacks require
   17×–200× larger perturbation budgets than weight attacks to compromise
   the model (50% clean accuracy threshold). The ADS-per-damage ratio
   bifurcates by PE operation space: at or below weight-attack baselines
   for embedding-space PE (sub-critical evasion), but 16×–25× above for
   attention-space PE (forensically conspicuous). Replicates across both
   datasets.

2. **Hierarchical PE fingerprint.** A universal attenuation dichotomy
   separates input-injected/rotary PE (slope ≈ −0.22 per layer) from
   ALiBi (slope ≈ +0.01) with near-perfect cross-dataset stability.
   Within-dataset, the full 12-layer profile classifies PE types via
   1-NN LOO at 66.7% (ImageNet-100) and 91.7% (CIFAR-100); chance = 25%.

3. **Operational detection boundary.** Against realistic benign shifts
   (JPEG, blur, noise), AUC ≥ 0.82 at ε ≥ 0.1 (Learned) / ε ≥ 0.2 (RoPE).
   An adaptive PGD attacker with ADS-minimization regularization cannot
   evade detection in this regime — ADS is a structural consequence of
   perturbation magnitude, verified with SPSA.

4. **Three-way convergence.** Layer profile, residual-stream probing,
   and ADS-per-damage signature all partition PE types along the same
   operation-space axis, providing cross-validation of the forensic
   signature as a genuine functional property.

---

## Relationship to companion work

This work uses the same 24 trained ViT-Base checkpoints as the companion
work [vit-positional-adversarial](https://github.com/djokobandjur/vit-positional-adversarial)
(resubmitted in parallel; Zenodo Concept DOI:
[`10.5281/zenodo.19154465`](https://doi.org/10.5281/zenodo.19154465)).
The companion work establishes the PE attack surface as a *performance
problem*; the present work reframes it as a *forensic problem*.

**Attack convention note.** The companion work uses a *shared-δ* attack
convention (single perturbation applied identically to all 12 transformer
blocks), while this work uses a *per-buffer independent δ_i* convention
(each replicated PE buffer receives its own optimized perturbation). The
per-buffer convention is methodologically matched to weight-level attacks
(Q/K/V, MLP), which operate on per-block parameters by definition. Both
conventions are valid threat models; the manuscript (Section II.E and
Table II) documents the convention explicitly and provides side-by-side
comparison with the companion work.

---

## Repository layout

```
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
├── ads_adaptive.py                # Adaptive PGD with ADS-min regularization
├── ads_comparison.py              # PE-to-PE comparison
├── ads_threshold_fine.py          # Fine ε grid + interpolated threshold
├── ads_ref_evasion.py             # Reference set evasion experiment
├── ads_probing_residual.py        # Residual stream probing (ImageNet-100)
├── ads_probing_residual_cifar.py  # Residual stream probing (CIFAR-100)
├── ads_roc_analysis.py            # ROC AUC vs benign shifts
│
├── reproduce.py                   # CPU-only verification of all paper claims
├── generate_ads_figures.py        # Figure generation from JSON results
│
├── colab_quickstart.ipynb         # End-to-end Colab pipeline notebook
│
├── data/                          # 13 JSON result files
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
│   └── (additional auxiliary files)
│
├── output/figures/                # Generated by generate_ads_figures.py
│
└── paper/
    └── refs_ads.bib
```

### Script-to-output mapping

| Script | Purpose | Output |
| --- | --- | --- |
| `ads_experiment.py` | Main ADS PE-only attack on ImageNet-100; 4 PE × 3 seeds × 11 ε values; logs accuracy, ADS(mean), ADS(L4), per-layer ADS | `ads_results.json` |
| `ads_experiment_cifar.py` | Same on CIFAR-100 | `ads_results_cifar100.json` |
| `ads_specificity.py` | 4 attack surfaces (PE-only, QKV-only, MLP-only, all-weights) × 4 PE on ImageNet-100; saturation budget asymmetry analysis (Section 6.2) | `ads_specificity.json` |
| `ads_specificity_cifar.py` | Same on CIFAR-100 | `ads_specificity_cifar.json` |
| `ads_adaptive.py` | Adaptive PGD attacker with L_evasion = -L_CE + λ·ADS regularization; tests whether knowledge of ADS metric enables evasion (Section 6.7) | `ads_adaptive.json` |
| `ads_comparison.py` | PE-to-PE ADS signature comparison | `ads_comparison.json` |
| `ads_threshold_fine.py` | Fine ε grid (14 values from 0.001 to 1.0) with log-log interpolation of 10×-baseline crossing; tests universality of detection threshold | `ads_threshold_fine.json` |
| `ads_ref_evasion.py` | Reference set evasion: attacker knows the 256 ref images and optimizes -L_CE + λ·ADS(ref) to evade detection while damaging held-out accuracy | `ads_ref_evasion.json` |
| `ads_probing_residual.py` | Layer-wise positional probing on residual stream activations (768-d per patch, image-level GroupKFold CV) for ImageNet-100 | `ads_probing_residual.json` |
| `ads_probing_residual_cifar.py` | Same on CIFAR-100 | `ads_probing_residual_cifar.json` |
| `ads_roc_analysis.py` | ROC AUC of ADS detector vs realistic benign shifts (JPEG, blur, noise) | `ads_roc_v2.json` |
| `reproduce.py` | CPU-only verification (no GPU/PyTorch required); regenerates every numerical claim, table value, and statistical test from JSON data with PASS/FAIL tolerances | Console output + `output/tables/` |
| `generate_ads_figures.py` | Generates Fig. 1–4 from JSON result files | `output/figures/*.png` |

---

## Trained models

This work reuses the 24 ViT-Base checkpoints (4 PE types × 3 seeds × 2
datasets) trained for the companion work. They are hosted on Google Drive
and documented in the companion repository:

- **ImageNet-100 models:** [drive.google.com/.../1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9](https://drive.google.com/drive/folders/1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9)
- **CIFAR-100 models:** [drive.google.com/.../1HBiOjNfuRsh2H0ZGRP4rIdBeydedCBJL](https://drive.google.com/drive/folders/1HBiOjNfuRsh2H0ZGRP4rIdBeydedCBJL)

For the model architecture, training procedure, and clean-accuracy
benchmarks, see the [companion repository](https://github.com/djokobandjur/vit-positional-adversarial)
(Concept DOI: [`10.5281/zenodo.19154465`](https://doi.org/10.5281/zenodo.19154465)).

The folder structure expected by the scripts is:

```
<models_dir>/
├── learned_seed42/best_model.pth
├── learned_seed123/best_model.pth
├── ...
└── alibi_seed456/best_model.pth
```

---

## Reproducing results

### Verification only (no GPU needed)

To verify every numerical claim, table value, and statistical test in the
paper directly from the archived JSON files:

```bash
pip install -r requirements.txt
python reproduce.py
```

This runs on CPU, requires no model checkpoints, and prints PASS/FAIL with
explicit tolerances for each claim. Output tables are written to
`output/tables/`.

### Selective execution (single PE and/or seed, GPU needed)

To run any experiment script for a specific PE type and/or seed instead
of the full paper configuration, use the optional `--pe_types` and
`--seeds` CLI arguments:

```bash
python ads_experiment.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "output/partial.json" \
    --pe_types learned --seeds 42
```

Argument reference:

```
--pe_types  Optional. PE types to evaluate. If omitted, uses the
            hardcoded PE_TYPES list defined in the script (paper
            configuration). Values: learned, sinusoidal, rope, alibi
            (space-separated for multiple).

--seeds     Optional. Random seeds to evaluate. If omitted, uses the
            hardcoded SEEDS list defined in the script (paper
            configuration). Values: 42, 123, 456 (space-separated for
            multiple).
```

The resulting JSON contains only the requested (PE, seed) cells but
shares the same structure as the full output. `generate_ads_figures.py`
handles partial outputs gracefully (omits error bands when n<3 seeds).
`reproduce.py` is paper-numbers verification and requires the full
3-seed JSONs shipped under `data/`; it aborts with a clear message if
pointed at a partial output directory.

### Full pipeline (GPU needed)

The fastest path is the [Colab Quickstart notebook](colab_quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/djokobandjur/ads-vit-forensics/blob/main/colab_quickstart.ipynb)).
It clones this repo, mounts Drive for model and dataset access, and walks
through the full pipeline (dataset prep → ADS experiments → specificity →
adaptive attacker → probing → ROC → figures → verification) in ten sections.
Each section can be re-run independently provided prior JSON outputs are
present on Drive.

### Manual CLI workflow

Each script supports a consistent `--models_dir / --val_dir / --output_path`
interface. The command sequence below assumes Colab-style paths and runs
the full paper configuration (4 PE × 3 seeds = 12 cells per script).

> **Selective execution.** All experiment scripts also accept optional
> `--pe_types` and `--seeds` arguments to restrict the run to a subset
> (see *Selective execution* section above for argument reference).

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

**2. Specificity (4 attack surfaces × 4 PE)**

```bash
python ads_specificity.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/path/to/imagenet100/val" \
    --output_path "data/ads_specificity.json"

python ads_specificity_cifar.py \
    --models_dir "/path/to/Trained models_CIFAR100" \
    --output_path "data/ads_specificity_cifar.json"
```

**3. Adaptive attacker (Section 6.7)**

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

**5. Residual stream probing**

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

**7. PE-to-PE comparison**

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

**9. Verify (no GPU required)**

```bash
python reproduce.py
```

---

## Adapting for local execution

All experiment scripts accept their input and output paths as CLI arguments
— adapt them to your local filesystem by changing the `--models_dir`,
`--val_dir`, and `--output_path` values when invoking the script. No code
changes are required for the experiment scripts themselves.

A few practical notes:

- **ImageNet-100 dataset structure** required by the scripts is the
  standard ImageFolder layout: one subdirectory per class, with image
  files inside. Pass the path to the parent of the class subdirectories
  as `--val_dir`. For Colab setup, see the companion repository's
  `00_setup_imagenet.py` utility.

- **CIFAR-100** requires no manual setup. The CIFAR-100 scripts invoke
  `torchvision.datasets.CIFAR100(download=True)` and cache to a local
  directory. The `--val_dir` argument selects the torchvision cache
  location; for some scripts it is optional.

- **Reference image indices** (`ads_ref_indices.json`, 256 images used as
  the ADS reference set) are shipped in `data/` for exact reproducibility.
  Most scripts read this file directly; if absent, they regenerate it
  using a fixed seed. To reproduce paper numbers exactly, keep the shipped
  file.

- **`reproduce.py` requires no GPU and no PyTorch.** It reads only the
  archived JSON files and uses `numpy` + `scipy` + `pandas` for statistical
  verification. Run this first to confirm the result files match the paper.

---

## Paper and citation

The paper is currently under resubmission to IEEE Transactions on
Information Forensics and Security, in parallel with the companion
work. Citation details will be added once the resubmission status
is finalized.

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

When citing the companion work (required if you reuse the trained
checkpoints), use:

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

This repository uses a dual-licensing scheme that reflects the different
nature of its contents:

- **Source code** (all `.py` files, the `colab_quickstart.ipynb` notebook)
  is released under the **MIT License** — see [`LICENSE`](LICENSE).

- **Result files and documentation** (the JSON files under `data/`, the
  generated figures under `output/figures/`, this README, `CHANGELOG.md`,
  and the Zenodo deposit of this repository) are released under the
  **Creative Commons Attribution 4.0 International License** (CC BY 4.0).
  Full text: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

- **Trained model checkpoints** (hosted on Google Drive) are released
  under **CC BY 4.0 for research purposes** by the companion work.
  ImageNet-100 models are derivative artifacts of the ImageNet-1k dataset
  and remain subject to the [ImageNet terms of access](https://www.image-net.org/download.php)
  for any redistribution or commercial use. CIFAR-100 models carry no
  such upstream restriction.

- **The ImageNet-1k validation images** required to reproduce the
  ImageNet-100 experiments are governed by the
  [ImageNet terms of access](https://www.image-net.org/download.php) and
  are not redistributed in this repository.

If you use the code, cite the repository under MIT terms. If you use the
results, figures, or trained models in a derivative work, cite under
CC BY 4.0 terms (attribution required).
