# Script → Output Map

This file maps each repository script/notebook to its role, required inputs, and expected outputs. It is intended as a quick reproducibility guide and as a sanity checklist for release packaging.

## Execution convention

Run commands from the repository root, both locally and in Colab:

```bash
cd /content/ads-vit-forensics   # Colab example after git clone
python scripts/<script_name>.py ...
```

If you do not `cd` into the repository root, prefix script paths explicitly, for example:

```bash
python /content/ads-vit-forensics/scripts/<script_name>.py ...
```

## Dataset setup helpers

| Script | Purpose | Main inputs | Expected output | GPU? | Notes |
|---|---|---|---|---:|---|
| `00_setup_imagenet.py` | Prepare ImageNet-100 validation set on Colab/local SSD. | `ILSVRC2012_img_val.tar`, `data/val_labels.txt`, `data/imagenet100_classes.txt` | `/content/imagenet100_resized/val/{synset_id}/*.JPEG` | No | Extracts 100 classes × 50 images = 5,000 validation images. Alternative fast path: unpack prebuilt `imagenet100_resized.tar`. |
| `00_setup_cifar100.py` | Prepare CIFAR-100 local cache when torchvision download is unavailable. | Cached `cifar-100-python` folder, typically from Drive | `/tmp/cifar100/cifar-100-python` | No | Pass `--val_dir /tmp/cifar100` to CIFAR scripts. The path must be the parent folder containing `cifar-100-python`. |
| `colab_quickstart.ipynb` | End-to-end Colab driver for setup, experiments, verification, figures, and packaging. | Google Drive paths for checkpoints/data; repo checkout | Multiple JSON/log/figure artifacts depending on selected cells | Mixed | Notebook wrapper; not a separate scientific experiment. |

## Core experiment scripts

| Script | Purpose | Dataset / protocol | Main inputs | Primary output(s) | GPU? |
|---|---|---|---|---|---:|
| `scripts/ads_experiment.py` | Primary PE-only ADS sweep. | ImageNet-100 | ImageNet-100 checkpoints, `/content/imagenet100_resized/val`, fixed reference indices | `data/ads_results.json` | Yes |
| `scripts/ads_experiment_cifar.py` | Primary PE-only ADS sweep. | CIFAR-100 | CIFAR-100 checkpoints, CIFAR root/cache | `data/ads_results_cifar100.json` | Yes |
| `scripts/ads_specificity.py` | Specificity stress test: PE-only vs QKV-only vs MLP-only vs all-weights. | ImageNet-100 | ImageNet-100 checkpoints and validation set | `data/ads_specificity.json` | Yes |
| `scripts/ads_specificity_cifar.py` | Specificity stress test: PE-only vs QKV-only vs MLP-only vs all-weights. | CIFAR-100 | CIFAR-100 checkpoints and test set/cache | `data/ads_specificity_cifar.json` | Yes |
| `scripts/ads_threshold_fine.py` | Fine-grid self-baseline trigger calibration. | ImageNet-100 | ImageNet-100 checkpoints, fine epsilon grid | `data/ads_threshold_fine.json` | Yes |
| `scripts/ads_roc_analysis.py` | Threshold-grid operating ROC/AUC calibration. | ImageNet-100; Learned/RoPE | ImageNet-100 checkpoints, reference/benign score protocol | `data/ads_roc_v2.json` | Yes |
| `scripts/ads_comparison.py` | ADS vs attention-distance / Mahalanobis-style / Logit-KL comparison. | Learned/RoPE diagnostic | Relevant checkpoints and validation data | `data/ads_comparison.json` | Yes |
| `scripts/ads_adaptive.py` | Adaptive ADS-regularized attack stress test. | Learned/RoPE diagnostic | Relevant checkpoints and validation/reference data | `data/ads_adaptive.json` or `data/ads_adaptive_eval.json` | Yes |
| `scripts/ads_ref_evasion.py` | Reference-set / holdout evasion stress test. | Learned/RoPE diagnostic | Relevant checkpoints, reference and holdout sets | `data/ads_ref_evasion.json` or `data/ads_ref_evasion_eval.json` | Yes |
| `scripts/ads_probing_residual.py` | Residual-stream probing sanity check. | ImageNet-100 | ImageNet-100 checkpoints and validation/reference data | `data/ads_probing_residual.json` | Yes |
| `scripts/ads_probing_residual_cifar.py` | Residual-stream probing sanity check. | CIFAR-100 | CIFAR-100 checkpoints and test/cache data | `data/ads_probing_residual_cifar.json` | Yes |
| `scripts/ads_shared_delta_attack_convention.py` | Shared-δ / tied-buffer PGD-PE attack-convention control. | ImageNet-100 or CIFAR-100 depending on `--dataset` | Dataset-specific checkpoints and validation/test data | `data/ads_shared_delta_imagenet100.json` or `data/ads_shared_delta_cifar100.json` | Yes |

## Derived verification and figure scripts

| Script | Purpose | Inputs | Outputs | GPU? | Notes |
|---|---|---|---|---:|---|
| `scripts/compute_roc_rank_auc_sensitivity.py` | CPU-only sensitivity check for ROC protocol. | `data/ads_roc_v2.json` | `data/ads_roc_rank_auc_sensitivity.json` | No | Compares exact single-positive rank AUC against stored 11-threshold operating AUC. |
| `scripts/generate_ads_figures.py` | Regenerate paper figures from archived JSON files. | Primary JSON files in `data/` | `figures/ads_fig1_l4_vs_epsilon.{png,pdf}`, `figures/ads_fig2_per_layer_heatmap.{png,pdf}`, `figures/ads_fig3_layer_profile.{png,pdf}`, `figures/ads_fig4_early_warning_combined.{png,pdf}` | No | Figure-generation only; does not rerun model attacks. |
| `scripts/reproduce.py` | CPU-only verification of paper tables, statistics, shared-δ control, ROC sensitivity, and optional figures. | Archived JSON files in `data/` | `output/reproduce_log.txt`, `output/tables/*.txt`, optional regenerated figures | No | Preferred final release check: `python scripts/reproduce.py --data-dir data --output-dir output --no-figures`. |

## Protocol-robustness scripts

| Script | Purpose | Inputs | Outputs | GPU? |
|---|---|---|---|---:|
| `scripts/robustness/ads_experiment_cifar_protocol_robustness_n12.py` | CIFAR-100 canonical/protocol-robustness PE sweep. | CIFAR-100 robustness checkpoints/cache | `data/robustness/ads_results_cifar100_canonical_n12.json` | Yes |
| `scripts/robustness/ads_specificity_cifar_protocol_robustness_n12.py` | CIFAR-100 canonical/protocol-robustness specificity sweep. | CIFAR-100 robustness checkpoints/cache | `data/robustness/ads_specificity_cifar100_canonical_n12.json` | Yes |
| `scripts/robustness/ads_probing_residual_cifar_protocol_robustness_n12.py` | CIFAR-100 canonical/protocol-robustness probing. | CIFAR-100 robustness checkpoints/cache | `data/robustness/ads_probing_residual_cifar100_canonical_n12.json` | Yes |

## Static / metadata files consumed by scripts

| File | Role |
|---|---|
| `data/val_labels.txt` | ImageNet validation image-to-synset mapping used by `00_setup_imagenet.py`. |
| `data/imagenet100_classes.txt` | 100 synset IDs defining the ImageNet-100 split. |
| `data/ads_ref_indices.json` | Fixed reference indices used by legacy/general reference-set checks. |
| `data/ads_ref_indices_imagenet100.json` | Fixed ImageNet-100 reference indices. |
| `data/ads_ref_indices_cifar100.json` | Fixed CIFAR-100 reference indices, if used by CIFAR diagnostics. |

## Recommended minimal release checks

```bash
# From repository root
python scripts/reproduce.py --data-dir data --output-dir output --section attack_convention --no-figures
python scripts/reproduce.py --data-dir data --output-dir output --section roc_sensitivity --no-figures
python scripts/reproduce.py --data-dir data --output-dir output --no-figures
python scripts/generate_ads_figures.py --data-dir data --output-dir .
```

## Artifact categories

- **Primary experimental JSONs:** `ads_results*.json`, `ads_specificity*.json`, `ads_threshold_fine.json`, `ads_roc_v2.json`, `ads_probing_residual*.json`.
- **Stress-test / auxiliary JSONs:** `ads_adaptive*.json`, `ads_ref_evasion*.json`, `ads_comparison.json`, `ads_shared_delta_*.json`.
- **Derived verification JSONs:** `ads_roc_rank_auc_sensitivity.json`.
- **Static metadata:** `val_labels.txt`, `imagenet100_classes.txt`, fixed reference-index JSONs.
- **Generated publication artifacts:** `figures/*.png`, `figures/*.pdf`, `output/tables/*.txt`, `output/reproduce_log.txt`.
