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

---

## What this repository provides

- **`reproduce.py`** — single-script reproduction of every table value, statistical
  test, and verification claim in the paper.
- **`generate_ads_figures.py`** — regeneration of all 4 paper figures from the
  same archived JSON data.
- **`data/`** — pre-computed experimental data: per-seed attention distributions,
  layer-wise ADS values, accuracy trajectories, attack-specificity measurements,
  and residual-stream probing results.
- **`paper/refs_ads.bib`** — verified bibliography file (23 entries, all cited in
  the manuscript).

The repository is organized to make reviewer-side verification straightforward:
clone, install dependencies, place the data, run one script.

---

## Repository layout

```
ads-vit-forensics/
├── README.md                       (this file)
├── LICENSE                         (MIT for code)
├── DATA_LICENSE                    (CC-BY 4.0 for data)
├── reproduce.py                    (main reproducibility script)
├── generate_ads_figures.py         (figure regeneration)
├── requirements.txt                (Python dependencies)
├── data/
│   ├── ads_results.json                ImageNet-100 main ADS experiment
│   ├── ads_results_cifar100.json       CIFAR-100 main ADS experiment
│   ├── ads_specificity.json            ImageNet-100 specificity (4 PE × 4 attacks)
│   ├── ads_specificity_cifar.json      CIFAR-100 specificity (4 PE × 4 attacks)
│   ├── ads_probing_residual.json       ImageNet-100 residual-stream probing
│   └── ads_probing_residual_cifar.json CIFAR-100 residual-stream probing
├── output/                         (created on first run)
│   ├── tables/                         per-table reproductions
│   ├── figures/                        regenerated PNGs
│   └── reproduce_log.txt               full reproduction log
└── paper/
    └── refs_ads.bib                bibliography
```

---

## Quick start

```bash
# Clone repo
git clone https://github.com/<USER>/ads-vit-forensics.git
cd ads-vit-forensics

# Set up environment
python -m venv venv
source venv/bin/activate            # Linux/macOS
# venv\Scripts\activate              # Windows
pip install -r requirements.txt

# Reproduce all numbers from paper
python reproduce.py

# Regenerate all 4 figures
python generate_ads_figures.py
```

Output appears in `output/`:

- `output/tables/table_VI_fingerprint.txt` — Table VI (fingerprint ratios)
- `output/tables/stats_ANOVA.txt` — Section 6.3 statistics
- `output/figures/ads_fig{1..4}_*.png` — paper figures
- `output/reproduce_log.txt` — full timestamped log with verification assertions

---

## What gets reproduced

`reproduce.py` regenerates every numerical claim in the paper that depends on
the experimental data. Verification against published values is included; each
reproduced number passes a tolerance check against the paper's reported value
(`Δ < 0.05` for confidence-interval widths, `Δ < 0.1` for ratio means).

| Paper section | Quantity | File |
|---|---|---|
| §6.2 (Table VII) | PE attack signatures @ ε=0.2 (4 PE × 4 attacks × 2 datasets) | `output/tables/table_VII_specificity.txt` |
| §6.2 | Damage asymmetry (MLP/PE drop ratio) | included in same file |
| §6.2 (Effect 1) | Saturation-budget ratios (50% and 5% thresholds, 4 PE × 2 datasets) | `output/tables/stats_saturation_budget.txt` |
| §6.3 (Table VI) | ADS(L4)/ADS(mean) fingerprint ± 95% CI, CV, ICC | `output/tables/table_VI_fingerprint.txt` |
| §6.3 (Tables VII, VIII) | Hierarchical fingerprint: slope (Level 1) and 12-d profile LOO (Level 2) | `output/tables/stats_hierarchical_fingerprint.txt` |
| §6.3 | ANOVA (ImageNet F=14.06 p=0.0015; CIFAR F=1.94 p=0.20) | `output/tables/stats_ANOVA.txt` |
| §6.3 | Welch t-tests (ALiBi vs.\ rest, embedding vs.\ attention-space) | same file |
| §6.3 | Pairwise Welch tests across all PE pairs | same file |
| §6.3 | 1-NN LOO classification (50% acc, perm. p=0.047) | logged in `reproduce_log.txt` |
| §6.3 | Cross-dataset profile transfer (50%, both directions) | logged in `reproduce_log.txt` |
| §7.1 (Table IX) | Residual-stream probing peaks (4 PE × 2 datasets) | `output/tables/table_VIII_probing.txt` |

The verification step at the end of `reproduce.py` prints PASS/FAIL for each
table value against the published number. All 8 fingerprint values pass with
`Δ = 0.00`, the saturation-budget range (17×–200×) matches the paper claim
exactly, and the hierarchical fingerprint slope dichotomy (ALiBi ≈ +0.013,
others ≈ -0.22; magnitude ratio 17×) reproduces with seed-level precision.

---

## Data

### Provenance

The JSON files in `data/` were produced by training 24 ViT-Base models
(4 PE strategies × 3 random seeds × 2 datasets) on ImageNet-100 and CIFAR-100,
then computing layer-wise attention distributions under controlled parameter
perturbations. Training and instrumentation code is **not** in this repository
(it would require GPU resources and full datasets to use); however, full
training details are documented in the paper's experimental section, and the
training code is available on request from the corresponding author.

### File schema

Each JSON has the structure:

```
{
  "<pe_type>": {                    # learned, sinusoidal, rope, alibi
    "<seed>": {                     # 42, 123, 456
      "epsilons":     [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
      "accuracies":   [<top-1 accuracy at each ε>],
      "ads_layer4":   [<KL divergence at L4 at each ε>],
      "ads_mean":     [<mean across all 12 layers at each ε>],
      "ads_per_layer":[[<12-element layer profile>] for each ε],
      "clean_acc":    <baseline accuracy with no perturbation>
    }
  }
}
```

For specificity files, the schema adds an attack-type level:

```
{
  "<pe_type>": {
    "<seed>": {
      "<attack>": {                 # pe_only, qkv_only, mlp_only, all_weights
        "epsilons":     [...],
        "accuracies":   [...],
        ...
      }
    }
  }
}
```

For probing files:

```
{
  "<pe_type>": {
    "<seed>": {
      "peak_layer":   <int>,
      "n_samples":    50176,        # ImageNet (256 imgs × 196 patches)
      "feature_dim":  768,          # residual stream dimension
      "layers": {
        "<layer_idx>": {            # "1" through "12"
          "r2_row":      <float>,
          "r2_col":      <float>,
          "r2_mean":     <float>,
          "r2_position": <float>
        }
      }
    }
  }
}
```

---

## Dependencies

Listed in `requirements.txt`:

- `numpy >= 1.20`
- `scipy >= 1.7`
- `matplotlib >= 3.4`

That's it. The reproducibility script is intentionally light: no PyTorch, no
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
@article{bandjur2026ads,
  author  = {Bandjur, Djoko and Bandjur, Milos},
  title   = {Adversarial Vulnerability of Positional Encoding in Vision Transformers: A Targeted Attack Analysis},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  note    = {Under review}
}

```

**Dataset** (10.5281/zenodo.19154465):

```bibtex
@dataset{bandjur2026ads_data,
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
- **Data** (`data/*.json`): Creative Commons Attribution 4.0 (CC-BY 4.0) — see
  [DATA_LICENSE](DATA_LICENSE).

---

## Acknowledgments

We thank the anonymous reviewers (when available) for their feedback on this
work. Computing resources were provided by `<acknowledgment placeholder>`.

---

## Status

| Item | Status |
|---|---|
| Paper submitted to IEEE TIFS | ✓ |
| Reproducibility script verified end-to-end | ✓ (all 8 table values pass) |
| Data archived on Zenodo | placeholder |
| Repository made public | upon acceptance |
| Camera-ready DOI | upon acceptance |

Last updated: April 2026.
