# CIFAR-100 canonical protocol-robustness n=12 cohort

This folder contains the auxiliary scripts and result-file conventions for the
independent CIFAR-100 canonical protocol-robustness cohort used in the ADS/TIFS
resubmission. These scripts are **not** part of the primary n=6
ImageNet-100/CIFAR-100 sweep. They support the protocol-robustness analysis for
1D-ALiBi and grid-aware 2D-ALiBi-style variants.

## Scope

The canonical robustness cohort evaluates three CIFAR-100 conditions:

- `alibi` — 1D-ALiBi canonical
- `alibi_2d` — 2D-ALiBi canonical
- `alibi_2d_matched` — 2D-ALiBi matched

Canonical seeds:

```text
1 5 7 11 13 21 42 99 123 456 2024 31337
```

The corresponding checkpoint folders are expected to follow the same naming
pattern used by the training pipeline:

```text
<condition>_seed<seed>/best_model.pth
```

For example:

```text
alibi_seed1/best_model.pth
alibi_2d_seed1/best_model.pth
alibi_2d_matched_seed1/best_model.pth
```

## Implementation convention

Checkpoint and result labels remain:

```text
alibi
alibi_2d
alibi_2d_matched
```

The `alibi_2d_matched` condition uses the same 2D-ALiBi model architecture as
`alibi_2d`; only the matched slope/bias configuration differs. In the scripts,
this is handled through `MODEL_PE_ALIAS`, which maps:

```python
"alibi_2d_matched" -> "alibi_2d"
```

The repository version of the model-definition file should already include
support for both `alibi_2d` and `alibi_2d_matched` labels. No legacy versioned
model-definition filename is required.

## Scripts

Primary protocol-robustness scripts:

```text
ads_experiment_cifar_protocol_robustness_n12.py
ads_specificity_cifar_protocol_robustness_n12.py
ads_probing_residual_cifar_protocol_robustness_n12.py
```

These generate the canonical n=12 JSON files used by the paper's robustness
analysis. Adaptive/ref-evasion variants are not part of the publication-grade
canonical robustness result unless separate benign/noise-floor thresholds are
calibrated for this cohort.

Recommended repository placement:

```text
scripts/robustness/
├── README.md
├── ads_experiment_cifar_protocol_robustness_n12.py
├── ads_specificity_cifar_protocol_robustness_n12.py
└── ads_probing_residual_cifar_protocol_robustness_n12.py
```

Recommended output placement:

```text
data/robustness/
├── ads_results_cifar100_canonical_n12.json
├── ads_specificity_cifar100_canonical_n12.json
└── ads_probing_residual_cifar100_canonical_n12.json
```

## Colab/local configuration

Set these paths at the top of your Colab notebook or shell session:

```bash
MODELS_DIR="/content/drive/MyDrive/pe_experiment/results_cifar100"
VAL_DIR="/tmp/cifar100"
OUT_DIR="/content/drive/MyDrive/ads_protocol_robustness_cifar_n12/data"
mkdir -p "$OUT_DIR"
```

Use your actual Drive folder name for `MODELS_DIR`. The path above is only a
template.

## Smoke test

Run one condition and one seed first:

```bash
python -u scripts/robustness/ads_experiment_cifar_protocol_robustness_n12.py \
  --models_dir "$MODELS_DIR" \
  --val_dir "$VAL_DIR" \
  --output_path "$OUT_DIR/test_alibi2d_matched_seed1.json" \
  --pe_types alibi_2d_matched \
  --seeds 1
```

## Full canonical n=12 runs

### 1. Main ADS trajectory

```bash
python -u scripts/robustness/ads_experiment_cifar_protocol_robustness_n12.py \
  --models_dir "$MODELS_DIR" \
  --val_dir "$VAL_DIR" \
  --output_path "$OUT_DIR/ads_results_cifar100_canonical_n12.json"
```

### 2. Specificity / ADS-per-damage comparison

```bash
python -u scripts/robustness/ads_specificity_cifar_protocol_robustness_n12.py \
  --models_dir "$MODELS_DIR" \
  --val_dir "$VAL_DIR" \
  --output_path "$OUT_DIR/ads_specificity_cifar100_canonical_n12.json"
```

### 3. Residual-stream probing

```bash
python -u scripts/robustness/ads_probing_residual_cifar_protocol_robustness_n12.py \
  --models_dir "$MODELS_DIR" \
  --val_dir "$VAL_DIR" \
  --output_path "$OUT_DIR/ads_probing_residual_cifar100_canonical_n12.json"
```

## Expected interpretation in the paper

This cohort is an auxiliary robustness check. The resulting values should be
read as **within-cohort stress-test magnitudes** for independently trained
canonical CIFAR-100 checkpoints and ALiBi-style attack surfaces. They should not
be compared directly to, or read as replacements for, the primary n=6
ImageNet-100/CIFAR-100 ratios.

The intended claims supported by this cohort are:

- 1D-ALiBi preserves the near-flat layer-profile separator, with slope around
  `+0.01`.
- Grid-aware 2D-ALiBi-style variants have decaying profiles and are therefore
  separable from 1D-ALiBi by profile shape.
- The protocol-robustness analysis supports the qualitative ALiBi separator;
  the exact ADS-per-damage multipliers are cohort-specific.

## Figure generation notes

The final ADS figure-generation code supports the `alibi_2d` and
`alibi_2d_matched` condition labels. Figures should be exported at 300 dpi. For
heatmaps that use log scaling, exact-zero ADS values should be rendered at the
lower log-scale bound rather than as the colormap's missing/invalid color.
