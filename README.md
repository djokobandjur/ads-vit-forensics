# Attention Divergence Score (ADS): Forensic Metric for ViT Parameter-Level Attacks

[![Status](https://img.shields.io/badge/status-under%20review-orange)](#)
[![Paper](https://img.shields.io/badge/paper-IEEE%20TIFS%20(submitted)-blue)](#)
[![License](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Data](https://img.shields.io/badge/data-CC--BY%204.0-lightgrey)](#data)

This repository accompanies the manuscript **"Attention Divergence Score: A
Forensic Metric for Characterizing Parameter-Level Attacks in Vision
Transformers"**, currently under review at *IEEE Transactions on Information
Forensics and Security*.

The repository contains everything needed to reproduce all numerical claims,
tables, and figures in the paper from the archived experimental data, without
requiring GPU access or model retraining. 

The repository is organized to make reviewer-side verification straightforward:
clone, install dependencies, place the data, run one script.

---

## What this repository provides

- **`reproduce.py`** — single-script reproduction of every table value, statistical
  test, and verification claim in the paper.
- **`generate_ads_figures.py`** — regeneration of all 4 paper figures from the
  same archived JSON data.
- **`data/`** — pre-computed experimental data: per-seed attention distributions,
  layer-wise ADS values, accuracy trajectories, attack-specificity measurements,
  residual-stream probing results, interpolated thresholds per PE type/seed, etc.
- **`paper/refs_ads.bib`** — verified bibliography file (23 entries, all cited in
  the manuscript).
  

> [!IMPORTANT]
> **Note on reproducibility verification**\
> **`reproduce.py`** includes built-in verification against published paper values,
  printing PASS/FAIL for each numerical claim with tolerances (e.g., Δ < 0.05 for confidence-interval widths).
  
---

## Repository layout

```
ads-vit-forensics/
├── README.md                       (this file)
├── reproduce.py                    (main reproducibility script)
├── generate_ads_figures.py         (figure regeneration)
├── requirements.txt                (Python dependencies)
├── data/
│   ├── ads_results.json                 ImageNet-100 main ADS experiment
│   ├── ads_results_cifar100.json        CIFAR-100 main ADS experiment
│   ├── ads_specificity.json             ImageNet-100 specificity (4 PE × 4 attacks)
│   ├── ads_specificity_cifar.json       CIFAR-100 specificity (4 PE × 4 attacks)
│   ├── ads_probing_residual.json        ImageNet-100 residual-stream probing
│   ├── ads_adaptive.json                Adaptive Attacker experiment 
│   ├── ads_comparison.json              Detection Method Comparison experiment
│   ├── ads_ref_evasion.json             Reference Set Evasion experiment
│   ├── ads_roc_v2.json                  ADS ROC Analysis
│   ├── ads_threshold_fine.json          ADS Fine-Grid Threshold experiment
│   ├── ads_ref_indices.json             256 reference images
│   └── ads_probing_residual_cifar.json  CIFAR-100 residual-stream probing
│
├── output/                             (created on first run)
│   ├── tables/                         ← 6 .txt files with statistical tables
│   ├── figures/                        ← 4 .png files matching paper figures
│   └── reproduce_log.txt               full reproduction log
└── paper/
    └── refs_ads.bib                bibliography
```

---

## Quick start

```bash
git clone https://github.com/djokobandjur/ads-vit-forensics.git
cd ads-vit-forensics
pip install -r requirements.txt

# Reproduce all numerical claims
python reproduce.py

# Generate all 4 figures
python generate_ads_figures.py
```
---

## Data provenance

The JSON files in `data/` were produced by training 24 ViT-Base models
(4 PE strategies × 3 random seeds × 2 datasets) on ImageNet-100 and CIFAR-100,
then computing layer-wise attention distributions under controlled parameter
perturbations. Training and instrumentation code is **not** in this repository
(it would require GPU resources and full datasets to use); however, full
training details are documented in the paper's experimental section, and the
training code is available on request from the corresponding author.

## Dependencies

Listed in `requirements.txt`:

- `numpy >= 1.20`
- `scipy >= 1.7`
- `matplotlib >= 3.4`

The reproducibility script is intentionally light: no PyTorch, no
GPU, no CUDA, no large model files. A standard scientific Python install is
sufficient.

Tested on:
- Python 3.10 / Ubuntu 22.04 LTS
- Python 3.11 / macOS 14
- Python 3.10 / Windows 11

---

## Citation

If you use this code or data, please cite both the paper and the dataset:

**Paper** (upon acceptance):

```bibtex
@article{bandjur2026ads,
  author  = {Bandjur, Djoko and Bandjur, Milos},
  title   = {Attention Divergence Score: A Forensic Metric for
             Characterizing Parameter-Level Attacks in Vision Transformers},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  note    = {Under review}
}
```

**Dataset** (placeholder for Zenodo DOI):

```bibtex
@dataset{bandjur2026ads_data,
  author    = {Bandjur, Djoko and Bandjur, Milos},
  title     = {ADS-ViT-Forensics: experimental data for
               attention-divergence score measurements on Vision Transformers},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {<TO BE FILLED>},
  url       = {https://doi.org/<TO BE FILLED>}
}
```

---

## Related work

This work builds on, and extends, the prior PE-specific vulnerability analysis:

```bibtex
@article{bandjur2026adversarial,
  author  = {Bandjur, Djoko and Bandjur, Milos},
  title   = {Adversarial Vulnerability of Positional Encoding in Vision Transformers: A Targeted Attack Analysis},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  note    = {Under review}
}

```

**Dataset** (10.5281/zenodo.19154465):

```bibtex
@dataset{bandjur2026adversarial_data,
  author    = {Bandjur, Djoko and Bandjur, Milos},
  title     = {ViT-Positional-Adversarial: experimental data for a targeted attack analysis},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19154465},
  url       = {https://doi.org/10.5281/zenodo.19154465}
}

```

The prior work established the *existence* of asymmetric PE vulnerability as a
performance phenomenon. The present work develops the *forensic framework*
(attention-space signature, operational detection boundary, adaptive-attacker
evaluation, operation-space taxonomy) around it.

---

## Contact

For questions about the paper, code, or data:

- Corresponding author: `<djoko.bandjur@pr.ac.rs>`
- Faculty of Technical Sciences, University of Pristina — Kosovska Mitrovica

---

## License

- **Code** (`*.py`): MIT License — see [LICENSE](LICENSE).
- **Data** (`data/*.json`): Creative Commons Attribution 4.0 (CC-BY 4.0) — see [DATA_LICENSE](DATA_LICENSE).
---

Last updated: April 2026.
