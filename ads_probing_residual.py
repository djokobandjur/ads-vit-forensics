"""
Layer-wise Positional Probing — Residual Stream
=================================================
For each Transformer layer, trains a linear probe to predict
patch position (row, col) from patch token residual stream activations.

This is the methodologically correct approach for positional probing:
- Residual stream at layer l encodes the accumulated representation
  after l Transformer blocks, including both positional and semantic info
- Feature dimension: 768 (full embedding) per patch token
- Much richer signal than CLS→patch attention (12 dims)

Mechanistic connection to ADS:
  PE perturbation
    → disrupts residual stream positional encoding
    → disrupts Q/K projections (which read from residual stream)
    → disrupts attention distributions
    → high ADS

If Layer 4 residual stream has peak positional decodability,
the causal chain is empirically grounded.

Protocol:
  - Register forward hooks on each Transformer block output
  - Extract patch token activations (tokens 1..196, skip CLS)
  - Feature: residual stream vector at layer l, shape (768,)
  - Target: patch row ∈ [0,13] and col ∈ [0,13]
  - Classifier: Ridge regression, image-level GroupKFold CV (5 folds)
  - Metric: R²(row), R²(col), R²(mean)
  - n = 256 images × 196 patches = 50,176 samples per layer

Image-level CV: all 196 patches from the same image go into the
same fold, preventing data leakage from correlated activations.

Output: ads_probing_residual.json
"""

import os, sys, json
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR      = '/content/drive/MyDrive/pe_experiment/results'
DATA_DIR         = '/content/imagenet100'
SAVE_PATH        = '/content/drive/MyDrive/pe_experiment/results/ADS/ads_probing_residual.json'
REF_INDICES_PATH = '/content/drive/MyDrive/pe_experiment/results/ADS/ads_ref_indices.json'

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

PE_TYPES   = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS      = [42, 123, 456]
N_LAYERS   = 12
IMG_SIZE   = 224
PATCH_SIZE = 16
GRID_SIZE  = IMG_SIZE // PATCH_SIZE   # 14
EMBED_DIM  = 768
N_PATCHES  = GRID_SIZE ** 2           # 196
N_FOLDS    = 5
RIDGE_ALPHA = 1.0

# ============================================================
# PATCH POSITION LABELS
# ============================================================
patch_rows = np.array([i // GRID_SIZE for i in range(N_PATCHES)], dtype=np.float32)
patch_cols = np.array([i %  GRID_SIZE for i in range(N_PATCHES)], dtype=np.float32)

print(f"Grid: {GRID_SIZE}×{GRID_SIZE} = {N_PATCHES} patches")
print(f"Embed dim: {EMBED_DIM}, Feature per patch: {EMBED_DIM}")

# ============================================================
# DATA
# ============================================================
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transform)

with open(REF_INDICES_PATH) as f:
    ref_indices = json.load(f)

ref_loader = DataLoader(
    Subset(val_dataset, ref_indices),
    batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)

print(f"Reference set: {len(ref_indices)} images")

# ============================================================
# MODEL
# ============================================================
def load_model(pe_type, seed):
    model = VisionTransformer(
        img_size=224, patch_size=16, num_classes=100,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
    )
    ckpt = os.path.join(RESULTS_DIR, f'{pe_type}_seed{seed}', 'best_model.pth')
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
    model.eval().to(device)
    return model

# ============================================================
# RESIDUAL STREAM EXTRACTION VIA HOOKS
# ============================================================
@torch.no_grad()
def extract_residual_stream(model, loader):
    """
    Extract patch token residual stream activations at each layer.

    Uses forward hooks on each Transformer block. The hook captures
    the block OUTPUT (post-LayerNorm + MLP), which is the residual
    stream entering the next block.

    Returns:
      activations: dict {layer_idx (0-based): np.array (N_img*N_patches, D)}
      rows:        np.array (N_img*N_patches,)
      cols:        np.array (N_img*N_patches,)
      image_ids:   np.array (N_img*N_patches,) for GroupKFold
    """
    # Storage for hook outputs
    hook_outputs = {l: [] for l in range(N_LAYERS)}
    hooks = []

    # Register hooks on each Transformer block
    # Model structure: model.blocks is a list of TransformerBlock modules
    for l, block in enumerate(model.blocks):
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output shape: (B, N, D) where N = N_patches + 1 (includes CLS)
                # Take patch tokens only (skip CLS at index 0)
                patch_acts = output[:, 1:, :]  # (B, N_patches, D)
                hook_outputs[layer_idx].append(patch_acts.cpu())
            return hook
        h = block.register_forward_hook(make_hook(l))
        hooks.append(h)

    all_rows = []
    all_cols = []
    all_image_ids = []
    image_counter = 0

    for imgs, _ in loader:
        imgs = imgs.to(device)
        B = imgs.shape[0]

        # Forward pass triggers hooks
        _ = model(imgs)

        # Position labels for this batch
        rows_batch = np.tile(patch_rows, B)   # (B * N_patches,)
        cols_batch = np.tile(patch_cols, B)
        all_rows.append(rows_batch)
        all_cols.append(cols_batch)

        # Image IDs: all patches of image i get id = image_counter + i
        ids = np.repeat(np.arange(image_counter, image_counter + B), N_PATCHES)
        all_image_ids.append(ids)
        image_counter += B

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate and reshape hook outputs
    # Each hook_outputs[l] is a list of (B, N_patches, D) tensors
    activations = {}
    for l in range(N_LAYERS):
        acts = torch.cat(hook_outputs[l], dim=0)  # (N_img, N_patches, D)
        # Reshape to (N_img * N_patches, D)
        acts = acts.reshape(-1, EMBED_DIM).numpy()
        activations[l] = acts

    rows      = np.concatenate(all_rows)
    cols      = np.concatenate(all_cols)
    image_ids = np.concatenate(all_image_ids)

    return activations, rows, cols, image_ids

# ============================================================
# LINEAR PROBING WITH IMAGE-LEVEL CV
# ============================================================
def probe_layer(features, targets, image_ids, alpha=RIDGE_ALPHA):
    """
    Ridge regression with image-level GroupKFold CV.
    All patches from the same image are in the same fold.

    Returns: (mean_R², std_R²)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    clf = Ridge(alpha=alpha)
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = cross_val_score(clf, X, targets, groups=image_ids,
                             cv=gkf, scoring='r2')
    return float(np.mean(scores)), float(np.std(scores))

# ============================================================
# MAIN
# ============================================================
def run():
    all_results = {}

    for pe_type in PE_TYPES:
        all_results[pe_type] = {}
        print(f"\n{'='*60}")
        print(f"PE TYPE: {pe_type.upper()}")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            model = load_model(pe_type, seed)

            print(f"    Extracting residual stream activations...")
            activations, rows, cols, image_ids = extract_residual_stream(
                model, ref_loader
            )

            n_samples = activations[0].shape[0]
            n_unique_images = len(np.unique(image_ids))
            print(f"    Samples: {n_samples} ({n_unique_images} images × {N_PATCHES} patches)")
            print(f"    Feature dim: {activations[0].shape[1]}")

            layer_results = {}
            print(f"\n    {'Layer':>7}  {'R²(row)':>10}  {'R²(col)':>10}  {'R²(mean)':>10}")
            print(f"    {'-'*45}")

            peak_r2 = -np.inf
            peak_l  = -1

            for l in range(N_LAYERS):
                r2_row, _ = probe_layer(activations[l], rows, image_ids)
                r2_col, _ = probe_layer(activations[l], cols, image_ids)
                r2_mean   = (r2_row + r2_col) / 2.0

                layer_results[str(l + 1)] = {
                    'r2_row':  r2_row,
                    'r2_col':  r2_col,
                    'r2_mean': r2_mean,
                }

                if r2_mean > peak_r2:
                    peak_r2 = r2_mean
                    peak_l  = l + 1

                print(f"    Layer {l+1:>2}:  {r2_row:>10.4f}  {r2_col:>10.4f}  "
                      f"{r2_mean:>10.4f}"
                      + (" ← L4" if l == 3 else ""))

            print(f"\n    → Peak: Layer {peak_l} (R²={peak_r2:.4f})")

            all_results[pe_type][str(seed)] = {
                'layers':    layer_results,
                'peak_layer': peak_l,
                'n_samples':  n_samples,
                'feature_dim': EMBED_DIM,
                'cv': 'image-level GroupKFold',
                'n_folds': N_FOLDS,
            }

            del model
            torch.cuda.empty_cache()

        # Save after each PE type
        with open(SAVE_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved to {SAVE_PATH}")

    return all_results

# ============================================================
# ANALYSIS
# ============================================================
def analyze(results):
    seeds_str = ['42', '123', '456']
    print("\n" + "=" * 65)
    print("RESIDUAL STREAM PROBING — SUMMARY")
    print("=" * 65)

    for pe_type in PE_TYPES:
        print(f"\n{pe_type.upper()} — R²(mean) by layer (mean ± std, n=3 seeds):")
        print(f"  {'Layer':>7}  {'R²':>8}  {'±std':>8}  {'':>5}")

        layer_means = {}
        for l in range(1, N_LAYERS + 1):
            vals = [results[pe_type][s]['layers'][str(l)]['r2_mean']
                    for s in seeds_str]
            m   = np.mean(vals)
            std = np.std(vals, ddof=1)
            layer_means[l] = m

        peak_l = max(layer_means, key=layer_means.get)

        for l in range(1, N_LAYERS + 1):
            vals = [results[pe_type][s]['layers'][str(l)]['r2_mean']
                    for s in seeds_str]
            m   = np.mean(vals)
            std = np.std(vals, ddof=1)
            flags = []
            if l == peak_l: flags.append("← PEAK")
            if l == 4:      flags.append("[L4]")
            print(f"  Layer {l:>2}:  {m:>8.4f}  {std:>8.4f}  {'  '.join(flags)}")

        # Per-seed peaks
        peaks = [results[pe_type][s]['peak_layer'] for s in seeds_str]
        l4_r2 = [results[pe_type][s]['layers']['4']['r2_mean'] for s in seeds_str]
        print(f"  Peak per seed: {peaks}")
        print(f"  Layer 4 R²: {np.mean(l4_r2):.4f} ± {np.std(l4_r2, ddof=1):.4f}")
        print(f"  Layer 4 is peak in {sum(p == 4 for p in peaks)}/3 seeds")

    # Cross-PE summary
    print(f"\n{'─'*65}")
    print("CROSS-PE PEAK LAYER SUMMARY:")
    print(f"{'PE':12} {'S42':>5} {'S123':>5} {'S456':>5} {'Mean':>7}")
    for pe_type in PE_TYPES:
        peaks = [results[pe_type][s]['peak_layer'] for s in seeds_str]
        print(f"{pe_type:12} {peaks[0]:>5} {peaks[1]:>5} {peaks[2]:>5} "
              f"{np.mean(peaks):>7.1f}")


if __name__ == '__main__':
    print("Layer-wise Positional Probing — Residual Stream")
    print(f"PE types: {PE_TYPES}, Seeds: {SEEDS}")
    print(f"Feature: residual stream patch activations, dim={EMBED_DIM}")
    print(f"CV: image-level GroupKFold, {N_FOLDS} folds")
    print(f"Samples: {len(ref_indices)} images × {N_PATCHES} patches = "
          f"{len(ref_indices) * N_PATCHES}")
    print(f"Estimated time: ~45-60 min on A100")
    print()

    results = run()
    analyze(results)
    print("\n✅ Residual stream probing complete!")
