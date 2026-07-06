"""Attention Divergence Score (ADS) Main Experiment — ImageNet-100
================================================================
Measures KL divergence between clean and PE-perturbed attention
distributions across all 12 transformer layers for each positional
encoding (PE) type and perturbation budget.

Default n=6 TFS/ADS protocol:
    4 PE types × 6 seeds × 11 epsilon values
    PE types: learned, sinusoidal, rope, alibi
    Seeds: 42, 123, 456, 789, 1011, 1213
    Checkpoint used: <models_dir>/<pe_type>_seed<seed>/best_model.pth

The script uses a PE-only PGD attack (T=20 steps, alpha=0.1*epsilon) so
that parameters and buffers are perturbed consistently across PE
implementations, including Sinusoidal buffers, RoPE cached buffers, and
ALiBi slopes.

Reference-set convention:
    ADS is computed on a fixed 256-image ImageNet-100 reference subset.
    The dataset-specific file ads_ref_indices_imagenet100.json is written
    next to the output JSON and reused on later runs.

Usage in Colab:
    1. Mount Drive.
    2. Ensure ImageNet-100 validation data is available in ImageFolder
       layout, e.g. /content/imagenet100_resized/val.
    3. Copy full_scale_experiment.py to /content/ or otherwise make
       VisionTransformer importable.
    4. Run, for example:

       !python /content/ads_experiment_n6.py \
           --models_dir "/content/drive/MyDrive/pe_experiment/results" \
           --val_dir "/content/imagenet100_resized/val" \
           --output_path "/content/drive/MyDrive/ads_tfs_n6/data/ads_results.json"

    Optional partial run:
       add --pe_types learned rope --seeds 42 1011

Outputs:
    - ads_results.json
    - ads_ref_indices_imagenet100.json
    - figures/ads_vs_epsilon.png
    - figures/ads_early_warning_dual.png
    - figures/ads_per_layer_heatmap.png
"""

import argparse
import os, sys, json, copy
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="ADS main experiment (ImageNet-100). See module docstring for details.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        required=True,
        help='Directory containing trained model checkpoints, organized as '
             '<models_dir>/<pe_type>_seed<seed>/best_model.pth',
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        required=True,
        help='Path to ImageNet-100 val directory in ImageFolder format',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for the result JSON',
    )
    parser.add_argument(
        '--pe_types',
        type=str,
        nargs='+',
        default=None,
        choices=['learned', 'sinusoidal', 'rope', 'alibi'],
        help='Optional. PE types to evaluate. If omitted, uses the default n=6 paper '
             'configuration: learned sinusoidal rope alibi.',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        help='Optional. Random seeds to evaluate. If omitted, uses the default n=6 paper '
             'configuration: 42 123 456 789 1011 1213.',
    )
    return parser.parse_args()


# ============================================================
# CONFIG
# ============================================================
args = parse_args()

RESULTS_DIR = args.models_dir
DATA_DIR    = args.val_dir
SAVE_PATH   = args.output_path
FIG_DIR     = os.path.join(os.path.dirname(SAVE_PATH), 'figures')

os.makedirs(os.path.dirname(SAVE_PATH) or '.', exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS    = [42, 123, 456, 789, 1011, 1213]

# Apply CLI overrides for PE_TYPES and SEEDS if user provided them
if args.pe_types is not None:
    PE_TYPES = args.pe_types
    print(f"[CLI override] PE_TYPES = {PE_TYPES}")
if args.seeds is not None:
    SEEDS = args.seeds
    print(f"[CLI override] SEEDS = {SEEDS}")

EPSILONS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

# PGD config — same as main adversarial experiment
PGD_STEPS       = 20
PGD_ALPHA_RATIO = 0.1   # alpha = epsilon * ratio

N_REF_IMAGES   = 256
LAYERS_FOR_ADS = list(range(12))

# ============================================================
# DATA
# ============================================================
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(DATA_DIR, val_transform)

# Fixed reference subset — saved for reproducibility
REF_INDICES_PATH = os.path.join(os.path.dirname(SAVE_PATH), 'ads_ref_indices_imagenet100.json')

if os.path.exists(REF_INDICES_PATH):
    with open(REF_INDICES_PATH) as f:
        ref_indices = json.load(f)
    print(f"Loaded reference indices from {REF_INDICES_PATH}")
else:
    torch.manual_seed(42)
    ref_indices = torch.randperm(len(val_dataset))[:N_REF_IMAGES].tolist()
    with open(REF_INDICES_PATH, 'w') as f:
        json.dump(ref_indices, f)
    print(f"Generated and saved {N_REF_IMAGES} reference indices to {REF_INDICES_PATH}")
    print("Upload ads_ref_indices_imagenet100.json with the n=6 result JSON for reproducibility!")

ref_dataset = Subset(val_dataset, ref_indices)
ref_loader  = DataLoader(ref_dataset, batch_size=32, shuffle=False,
                         num_workers=4, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

print(f"Reference images: {len(ref_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Protocol cells: {len(PE_TYPES)} PE types × {len(SEEDS)} seeds × {len(EPSILONS)} epsilons = {len(PE_TYPES) * len(SEEDS) * len(EPSILONS)} evaluations")

# ============================================================
# MODEL LOADING
# ============================================================
def load_model(pe_type, seed):
    model = VisionTransformer(
        img_size=224, patch_size=16, num_classes=100,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
    )
    ckpt_path = os.path.join(RESULTS_DIR, f'{pe_type}_seed{seed}', 'best_model.pth')
    state = torch.load(ckpt_path, map_location='cpu')
    clean_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state)
    model.eval().to(device)
    return model

# ============================================================
# ATTENTION EXTRACTION
# ============================================================
@torch.no_grad()
def get_attention_distributions(model, loader):
    """
    Extract mean attention distributions across all 12 layers.
    Returns list of 12 tensors, each (num_heads, N, N) averaged over dataset.
    """
    all_attentions = [[] for _ in range(12)]

    for images, _ in loader:
        images = images.to(device)
        _, attentions = model.forward_with_attention(images)
        for layer_idx, attn in enumerate(attentions):
            all_attentions[layer_idx].append(attn.mean(0).cpu())

    return [torch.stack(all_attentions[i]).mean(0) for i in range(12)]

# ============================================================
# ADS COMPUTATION
# ============================================================
def compute_ads(clean_attentions, perturbed_attentions):
    """
    Compute Attention Divergence Score (ADS) as KL divergence
    between clean and perturbed attention distributions per layer.
    """
    per_layer_ads = []
    eps_kl = 1e-10

    for layer_idx in LAYERS_FOR_ADS:
        P = clean_attentions[layer_idx].float() + eps_kl
        Q = perturbed_attentions[layer_idx].float() + eps_kl
        P = P / P.sum(dim=-1, keepdim=True)
        Q = Q / Q.sum(dim=-1, keepdim=True)
        kl = (P * torch.log(P / Q)).sum(dim=-1).mean().item()
        per_layer_ads.append(kl)

    return {
        'mean_ads':      float(np.mean(per_layer_ads)),
        'mid_layer_ads': float(np.mean(per_layer_ads[4:9])),
        'layer4_ads':    float(per_layer_ads[3]),
        'per_layer_ads': [float(x) for x in per_layer_ads],
    }

# ============================================================
# ACCURACY MEASUREMENT
# ============================================================
@torch.no_grad()
def measure_accuracy(model, loader):
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, pred = model(images).max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

# ============================================================
# PGD PE PERTURBATION
# ============================================================
def perturb_pe_alibi(model, epsilon):
    """
    ALiBi perturbation: convert slopes buffers to Parameters
    so gradients flow, then run PGD on slopes directly.
    """
    if epsilon == 0.0:
        return copy.deepcopy(model)

    perturbed_model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    alpha = epsilon * PGD_ALPHA_RATIO

    images, labels = next(iter(ref_loader))
    images, labels = images.to(device), labels.to(device)

    # Convert slopes buffers to Parameters in all ALiBi modules
    alibi_params = []
    for name, module in perturbed_model.named_modules():
        if type(module).__name__ == "ALiBi":
            slopes_data = module.slopes.clone().to(device)
            del module.slopes
            module.slopes = nn.Parameter(slopes_data, requires_grad=True)
            alibi_params.append(module.slopes)

    if not alibi_params:
        print(f"  WARNING: No ALiBi modules found")
        return perturbed_model

    # Initialize deltas
    deltas = [torch.zeros_like(p) for p in alibi_params]

    # PGD iterations
    perturbed_model.train()
    for step in range(PGD_STEPS):
        with torch.no_grad():
            for param, delta in zip(alibi_params, deltas):
                param.data = param.data - (delta if step > 0 else 0) + delta

        for param in alibi_params:
            if param.grad is not None:
                param.grad.zero_()

        outputs = perturbed_model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        with torch.no_grad():
            new_deltas = []
            for param, delta in zip(alibi_params, deltas):
                if param.grad is not None:
                    new_delta = delta + alpha * param.grad.sign()
                    new_delta = new_delta.clamp(-epsilon, epsilon)
                    new_deltas.append(new_delta)
                else:
                    new_deltas.append(delta)
            deltas = new_deltas

    # Apply final delta
    with torch.no_grad():
        for param, delta in zip(alibi_params, deltas):
            param.data += delta

    perturbed_model.eval()
    return perturbed_model


def perturb_pe_pgd(model, pe_type, epsilon):
    """
    PGD attack on PE parameters and buffers.
    For ALiBi, delegates to perturb_pe_alibi().
    Converts buffers to leaf tensors so gradients flow through
    Sinusoidal.pe, RoPE.cos_cached, RoPE.sin_cached etc.
    """
    if pe_type == "alibi":
        return perturb_pe_alibi(model, epsilon)

    if epsilon == 0.0:
        return copy.deepcopy(model)

    perturbed_model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    alpha = epsilon * PGD_ALPHA_RATIO

    # Get reference batch
    images, labels = next(iter(ref_loader))
    images, labels = images.to(device), labels.to(device)

    # Collect PE tensors — parameters + buffers
    # Strategy: replace buffers with leaf Parameters temporarily
    pe_param_names = []
    original_buffers = {}  # name -> original buffer value

    # First: handle named parameters
    for name, param in list(perturbed_model.named_parameters()):
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['pos_embed', 'slopes', 'inv_freq']):
            param.requires_grad_(True)
            pe_param_names.append(name)

    # Second: convert PE buffers to parameters
    for name, buf in list(perturbed_model.named_buffers()):
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['.pe', 'cos_cached', 'sin_cached']):
            if 'rel_dist' not in clean:
                original_buffers[name] = buf.clone()
                # Replace buffer with a parameter
                parts = clean.split('.')
                obj = perturbed_model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                # Remove buffer and add as parameter
                delattr(obj, parts[-1])
                new_param = nn.Parameter(buf.clone(), requires_grad=True)
                setattr(obj, parts[-1], new_param)
                pe_param_names.append(name)

    # Collect all PE parameters after conversion
    pe_params = []
    for name, param in perturbed_model.named_parameters():
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['pos_embed', 'slopes', 'inv_freq',
                                     '.pe', 'cos_cached', 'sin_cached']):
            pe_params.append(param)

    if not pe_params:
        print(f"  WARNING: No PE params found for {pe_type}")
        return perturbed_model

    # Initialize delta
    deltas = [torch.zeros_like(p) for p in pe_params]

    # PGD iterations
    perturbed_model.train()
    for step in range(PGD_STEPS):
        # Apply delta
        with torch.no_grad():
            for param, delta in zip(pe_params, deltas):
                param.data = param.data - (delta if step > 0 else 0) + delta

        # Zero grads
        for param in pe_params:
            if param.grad is not None:
                param.grad.zero_()

        # Forward + backward
        outputs = perturbed_model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Update deltas
        with torch.no_grad():
            new_deltas = []
            for param, delta in zip(pe_params, deltas):
                if param.grad is not None:
                    new_delta = delta + alpha * param.grad.sign()
                    new_delta = new_delta.clamp(-epsilon, epsilon)
                    new_deltas.append(new_delta)
                else:
                    new_deltas.append(delta)
            deltas = new_deltas

    # Apply final delta to PE parameters
    with torch.no_grad():
        for param, delta in zip(pe_params, deltas):
            param.data += delta

    perturbed_model.eval()
    return perturbed_model

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_ads_experiment():
    results = {}

    for pe_type in PE_TYPES:
        results[pe_type] = {}
        print(f"\n{'='*60}")
        print(f"PE TYPE: {pe_type.upper()}")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            results[pe_type][str(seed)] = {
                'clean_acc': None, 'epsilons': [],
                'accuracies': [], 'ads_mean': [],
                'ads_mid_layer': [], 'ads_layer4': [],
                'ads_per_layer': [],
            }

            model = load_model(pe_type, seed)
            print(f"    Model loaded")

            print(f"    Computing clean attention baseline ({N_REF_IMAGES} images)...")
            clean_attentions = get_attention_distributions(model, ref_loader)

            clean_acc = measure_accuracy(model, val_loader)
            results[pe_type][str(seed)]['clean_acc'] = clean_acc
            print(f"    Clean accuracy: {clean_acc:.2f}%")

            for eps in EPSILONS:
                print(f"    ε={eps:.3f}: ", end='', flush=True)

                if eps == 0.0:
                    ads_scores = {
                        'mean_ads': 0.0, 'mid_layer_ads': 0.0,
                        'layer4_ads': 0.0, 'per_layer_ads': [0.0] * 12,
                    }
                    acc = clean_acc
                else:
                    perturbed_model = perturb_pe_pgd(model, pe_type, eps)
                    perturbed_attentions = get_attention_distributions(
                        perturbed_model, ref_loader)
                    ads_scores = compute_ads(clean_attentions, perturbed_attentions)
                    acc = measure_accuracy(perturbed_model, val_loader)
                    del perturbed_model
                    torch.cuda.empty_cache()

                results[pe_type][str(seed)]['epsilons'].append(eps)
                results[pe_type][str(seed)]['accuracies'].append(acc)
                results[pe_type][str(seed)]['ads_mean'].append(ads_scores['mean_ads'])
                results[pe_type][str(seed)]['ads_mid_layer'].append(ads_scores['mid_layer_ads'])
                results[pe_type][str(seed)]['ads_layer4'].append(ads_scores['layer4_ads'])
                results[pe_type][str(seed)]['ads_per_layer'].append(ads_scores['per_layer_ads'])

                print(f"acc={acc:.1f}%, ADS(mean)={ads_scores['mean_ads']:.4f}, "
                      f"ADS(mid)={ads_scores['mid_layer_ads']:.4f}, "
                      f"ADS(L4)={ads_scores['layer4_ads']:.4f}")

            del model
            torch.cuda.empty_cache()

        with open(SAVE_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {SAVE_PATH}")

    return results

# ============================================================
# VISUALIZATION
# ============================================================
def plot_ads_results(results):
    import matplotlib.pyplot as plt

    PE_COLORS = {
        'learned': '#7B68EE', 'sinusoidal': '#00CED1',
        'rope': '#FF6347', 'alibi': '#32CD32'
    }
    seeds_str = [str(s) for s in SEEDS]

    def get_available_seeds(pe):
        """Return list of seed strings that have results for given PE type."""
        if pe not in results:
            return []
        return [s for s in seeds_str if s in results[pe]]

    # Plot 1: ADS(L4) vs Epsilon
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Attention Divergence Score (L4) vs ε', fontsize=15, fontweight='bold')
    for idx, pe in enumerate(PE_TYPES):
        ax = axes[idx // 2, idx % 2]
        available = get_available_seeds(pe)
        if not available:
            continue
        epsilons = results[pe][available[0]]['epsilons']
        ads_vals = np.array([results[pe][s]['ads_layer4'] for s in available])
        mean_ads = ads_vals.mean(0)
        std_ads  = ads_vals.std(0, ddof=1) if len(available) > 1 else np.zeros_like(mean_ads)
        ax.plot(epsilons, mean_ads, 'o-', color=PE_COLORS[pe], linewidth=2, markersize=7)
        ax.fill_between(epsilons, mean_ads - std_ads, mean_ads + std_ads,
                        alpha=0.2, color=PE_COLORS[pe])
        ax.set_xlabel('ε')
        ax.set_ylabel('ADS (KL divergence)')
        ax.set_title(pe.capitalize(), fontweight='bold', color=PE_COLORS[pe])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'ads_vs_epsilon.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ ads_vs_epsilon.png")

    # Plot 2: ADS + Accuracy dual axis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ADS Early Warning: Divergence vs Accuracy', fontsize=15, fontweight='bold')
    for idx, pe in enumerate(PE_TYPES):
        ax1 = axes[idx // 2, idx % 2]
        ax2 = ax1.twinx()
        available = get_available_seeds(pe)
        if not available:
            continue
        epsilons  = results[pe][available[0]]['epsilons']
        mean_ads  = np.mean([results[pe][s]['ads_layer4'] for s in available], axis=0)
        mean_accs = np.mean([results[pe][s]['accuracies'] for s in available], axis=0)
        l1, = ax1.plot(epsilons, mean_ads, 'o-', color=PE_COLORS[pe], linewidth=2, markersize=7, label='ADS (L4)')
        l2, = ax2.plot(epsilons, mean_accs, 's--', color='gray', linewidth=2, markersize=7, label='Accuracy')
        ax1.set_xlabel('ε')
        ax1.set_ylabel('ADS', color=PE_COLORS[pe])
        ax2.set_ylabel('Accuracy (%)', color='gray')
        ax1.set_title(pe.capitalize(), fontweight='bold', color=PE_COLORS[pe])
        ax1.grid(True, alpha=0.3)
        ax1.legend(handles=[l1, l2], loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'ads_early_warning_dual.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ ads_early_warning_dual.png")

    # Plot 3: Per-layer heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Layer ADS Heatmap (mean over seeds)', fontsize=15, fontweight='bold')
    for idx, pe in enumerate(PE_TYPES):
        ax = axes[idx // 2, idx % 2]
        available = get_available_seeds(pe)
        if not available:
            continue
        epsilons = results[pe][available[0]]['epsilons']
        matrix = np.array([
            np.mean([results[pe][s]['ads_per_layer'][i] for s in available], axis=0)
            for i in range(len(epsilons))
        ])
        im = ax.imshow(matrix.T, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([str(e) for e in epsilons], rotation=45, fontsize=8)
        ax.set_yticks(range(12))
        ax.set_yticklabels([f'L{i+1}' for i in range(12)])
        ax.set_xlabel('ε')
        ax.set_ylabel('Layer')
        ax.set_title(pe.capitalize(), fontweight='bold', color=PE_COLORS[pe])
        plt.colorbar(im, ax=ax, label='KL div')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'ads_per_layer_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ ads_per_layer_heatmap.png")


# ============================================================
# ANALYSIS
# ============================================================
def analyze_results(results):
    seeds_str = [str(s) for s in SEEDS]
    print("\n" + "=" * 70)
    print("KEY FINDINGS: ADS EARLY WARNING ANALYSIS")
    print("=" * 70)

    for pe in PE_TYPES:
        if pe not in results:
            continue
        # Use only seeds that actually have results (defensive against partial runs)
        available_seeds = [s for s in seeds_str if s in results[pe]]
        if not available_seeds:
            print(f"\n{pe.upper()}: no results")
            continue

        epsilons  = results[pe][available_seeds[0]]['epsilons']
        mean_acc  = np.mean([results[pe][s]['accuracies'] for s in available_seeds], axis=0)
        mean_ads  = np.mean([results[pe][s]['ads_layer4'] for s in available_seeds], axis=0)
        clean_acc = mean_acc[0]

        collapse_idx = next((i for i, acc in enumerate(mean_acc)
                             if acc < 0.5 * clean_acc), None)

        print(f"\n{pe.upper()} (n={len(available_seeds)} seeds: {available_seeds}):")
        if collapse_idx is None:
            print(f"  No collapse — max ADS(L4) at ε=1.0: {mean_ads[-1]:.4f}")
        else:
            pre = max(0, collapse_idx - 1)
            print(f"  Collapse: {mean_acc[pre]:.1f}% → {mean_acc[collapse_idx]:.1f}% "
                  f"at ε={epsilons[collapse_idx]}")
            print(f"  ADS(L4) before collapse: {mean_ads[pre]:.4f}")
            baseline = mean_ads[1] if mean_ads[1] > 1e-8 else 1e-6
            for i in range(1, collapse_idx):
                if mean_ads[i] > 10 * baseline:
                    print(f"  ADS>10×baseline at ε={epsilons[i]}: "
                          f"{collapse_idx - i} steps before collapse ← EARLY WARNING")
                    break


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    print("Attention Divergence Score (ADS) Experiment — ImageNet-100")
    print(f"Attack: PGD (T={PGD_STEPS} steps)")
    print(f"PE types: {PE_TYPES}")
    print(f"Seeds: {SEEDS}")
    print(f"Epsilons: {EPSILONS}")
    print(f"Reference images: {N_REF_IMAGES}")
    print(f"Results: {SAVE_PATH}")
    print()

    results = run_ads_experiment()
    analyze_results(results)

    try:
        plot_ads_results(results)
        print("\n✅ Complete!")
    except Exception as e:
        print(f"\n⚠️ Plotting failed: {e}")
