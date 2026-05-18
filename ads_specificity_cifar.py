"""
ADS Specificity Experiment
==========================
Tests whether ADS is specific to PE perturbations or reacts equally to
ANY parameter perturbation (which would undermine the "PE integrity" claim).

Three control conditions compared against PE perturbation:
  (a) QKV-only: perturb only Q/K/V projection weights in all attention blocks
  (b) MLP-only: perturb only MLP (FFN) block weights
  (c) All-weights: uniform small perturbation of ALL model parameters (same L-inf budget)

If ADS(PE) >> ADS(QKV) ≈ ADS(MLP) ≈ ADS(all): ADS is PE-specific -> strong result
If ADS(PE) ≈ ADS(QKV) ≈ ADS(MLP) ≈ ADS(all): ADS detects any perturbation -> weaker claim

Usage in Colab:
    1. Mount Drive
    2. Copy full_scale_experiment.py to /content/
    3. Run: !python /content/ads_specificity.py

Output:
    - results/ADS/ads_specificity_cifar.json
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
        description="ADS experiment script. See module docstring for details.",
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
        default='/tmp/cifar100', help='Optional. Path to torchvision CIFAR-100 cache directory (default: /tmp/cifar100). ',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for the result JSON',
    )
    parser.add_argument(
        '--ref_indices_path',
        type=str,
        default=None,
        help='Path to ads_ref_indices.json. If not specified, defaults to '
             'a path alongside --output_path. If the file does not exist, '
             'it will be generated using a fixed seed.',
    )
    parser.add_argument(
        '--pe_types',
        type=str,
        nargs='+',
        default=None,
        choices=['learned', 'sinusoidal', 'rope', 'alibi'],
        help='Optional. PE types to evaluate. If omitted, uses the '
             'hardcoded PE_TYPES list defined in the script (paper configuration).',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        help='Optional. Random seeds to evaluate. If omitted, uses the '
             'hardcoded SEEDS list defined in the script (paper configuration).',
    )
    return parser.parse_args()


# ============================================================
# CONFIG
# ============================================================
# Parse CLI arguments first; CONFIG constants are derived from them.
args = parse_args()

RESULTS_DIR      = args.models_dir
DATA_DIR         = args.val_dir
SAVE_PATH        = args.output_path

# REF_INDICES_PATH: if user provided, use it; else default alongside output
if args.ref_indices_path:
    REF_INDICES_PATH = args.ref_indices_path
else:
    REF_INDICES_PATH = os.path.join(
        os.path.dirname(args.output_path), 'ads_ref_indices.json'
    )

os.makedirs(os.path.dirname(SAVE_PATH) or '.', exist_ok=True)

# Use only Learned PE (most informative, fastest to perturb)
# and one additional PE type for cross-check
PE_TYPES = ['learned', 'rope', 'sinusoidal', 'alibi']
SEEDS    = [42, 123, 456]


# Apply CLI overrides for PE_TYPES and SEEDS if user provided them
if args.pe_types is not None:
    PE_TYPES = args.pe_types
    print(f"[CLI override] PE_TYPES = {PE_TYPES}")
if args.seeds is not None:
    SEEDS = args.seeds
    print(f"[CLI override] SEEDS = {SEEDS}")

# Test these epsilon values — same as main experiment
EPSILONS = [0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

# Perturbation targets
PERTURBATION_TYPES = ['pe_only', 'qkv_only', 'mlp_only', 'all_weights']

PGD_STEPS       = 20
PGD_ALPHA_RATIO = 0.1
N_REF_IMAGES    = 256

# ============================================================
# DATA
# ============================================================
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
])

val_dataset = datasets.CIFAR100(root=DATA_DIR, train=False,
                                 download=True, transform=val_transform)

# Load or create reference indices (reuse from main ADS experiment)
if os.path.exists(REF_INDICES_PATH):
    with open(REF_INDICES_PATH) as f:
        ref_indices = json.load(f)
    print(f"Loaded reference indices from {REF_INDICES_PATH}")
else:
    torch.manual_seed(42)
    ref_indices = torch.randperm(len(val_dataset))[:N_REF_IMAGES].tolist()
    with open(REF_INDICES_PATH, 'w') as f:
        json.dump(ref_indices, f)
    print(f"Generated and saved {N_REF_IMAGES} reference indices")

ref_dataset = Subset(val_dataset, ref_indices)
ref_loader  = DataLoader(ref_dataset, batch_size=32, shuffle=False,
                         num_workers=4, pin_memory=True)
val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)

print(f"Reference: {len(ref_dataset)} images, Val: {len(val_dataset)} images")

# ============================================================
# MODEL LOADING
# ============================================================
def load_model(pe_type, seed):
    model = VisionTransformer(
        img_size=32, patch_size=4, num_classes=100,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
    )
    ckpt = os.path.join(RESULTS_DIR, f'{pe_type}_seed{seed}', 'best_model.pth')
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
    model.eval().to(device)
    return model

# ============================================================
# ATTENTION EXTRACTION
# ============================================================
@torch.no_grad()
def get_attention_distributions(model, loader):
    all_attentions = [[] for _ in range(12)]
    for images, _ in loader:
        images = images.to(device)
        _, attentions = model.forward_with_attention(images)
        for layer_idx, attn in enumerate(attentions):
            all_attentions[layer_idx].append(attn.mean(0).cpu())
    return [torch.stack(all_attentions[l]).mean(0) for l in range(12)]

# ============================================================
# ADS COMPUTATION
# ============================================================
def compute_ads(clean_attn, perturbed_attn):
    eps_kl = 1e-10
    per_layer = []
    for l in range(12):
        P = clean_attn[l].float() + eps_kl
        Q = perturbed_attn[l].float() + eps_kl
        P = P / P.sum(-1, keepdim=True)
        Q = Q / Q.sum(-1, keepdim=True)
        kl = (P * torch.log(P / Q)).sum(-1).mean().item()
        per_layer.append(kl)
    return {
        'mean_ads':      float(np.mean(per_layer)),
        'mid_layer_ads': float(np.mean(per_layer[4:9])),
        'layer4_ads':    float(per_layer[3]),
        'per_layer_ads': [float(x) for x in per_layer],
    }

# ============================================================
# ACCURACY
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
# PERTURBATION FUNCTIONS
# ============================================================
def get_param_groups(model, pe_type):
    """
    Returns dict of parameter groups for selective perturbation.
    """
    groups = {
        'pe_only':   [],
        'qkv_only':  [],
        'mlp_only':  [],
        'all_weights': [],
    }

    for name, param in model.named_parameters():
        clean = name.replace('_orig_mod.', '')

        # PE parameters
        if any(k in clean for k in ['pos_embed', 'inv_freq']):
            groups['pe_only'].append(param)

        # QKV projections
        elif 'attn.qkv' in clean:
            groups['qkv_only'].append(param)

        # MLP blocks
        elif 'mlp' in clean and ('weight' in clean or 'bias' in clean):
            groups['mlp_only'].append(param)

        # All weights (includes everything above)
        groups['all_weights'].append(param)

    # Handle PE buffers (sinusoidal, rope, alibi)
    return groups


def convert_pe_buffers(model, pe_type):
    """Convert PE buffers to Parameters for gradient flow."""
    pe_params = []
    for name, buf in list(model.named_buffers()):
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['.pe', 'cos_cached', 'sin_cached']):
            if 'rel_dist' not in clean:
                parts = clean.split('.')
                obj = model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                delattr(obj, parts[-1])
                new_param = nn.Parameter(buf.clone(), requires_grad=True)
                setattr(obj, parts[-1], new_param)
                pe_params.append(new_param)

    # ALiBi slopes
    for name, module in model.named_modules():
        if type(module).__name__ == 'ALiBi':
            slopes_data = module.slopes.clone().to(device)
            del module.slopes
            module.slopes = nn.Parameter(slopes_data, requires_grad=True)
            pe_params.append(module.slopes)

    return pe_params


def pgd_perturb(model, param_list, epsilon, images, labels):
    """Generic PGD perturbation on a given list of parameters."""
    if epsilon == 0.0 or not param_list:
        return

    criterion = nn.CrossEntropyLoss()
    alpha = epsilon * PGD_ALPHA_RATIO

    for p in param_list:
        p.requires_grad_(True)

    deltas = [torch.zeros_like(p) for p in param_list]

    model.train()
    for step in range(PGD_STEPS):
        with torch.no_grad():
            for p, d in zip(param_list, deltas):
                p.data = p.data - (d if step > 0 else 0) + d

        for p in param_list:
            if p.grad is not None:
                p.grad.zero_()

        out = model(images)
        loss = criterion(out, labels)
        loss.backward()

        with torch.no_grad():
            new_deltas = []
            for p, d in zip(param_list, deltas):
                if p.grad is not None:
                    nd = d + alpha * p.grad.sign()
                    nd = nd.clamp(-epsilon, epsilon)
                    new_deltas.append(nd)
                else:
                    new_deltas.append(d)
            deltas = new_deltas

    # Apply final delta
    with torch.no_grad():
        for p, d in zip(param_list, deltas):
            p.data += d

    model.eval()


def perturb_model(model, pe_type, perturb_type, epsilon):
    """
    Creates a perturbed copy of the model according to perturb_type.
    """
    if epsilon == 0.0:
        return copy.deepcopy(model)

    pm = copy.deepcopy(model).to(device)

    # Get reference batch
    images, labels = next(iter(ref_loader))
    images, labels = images.to(device), labels.to(device)

    if perturb_type == 'pe_only':
        # Convert PE buffers to Parameters first
        extra_pe = convert_pe_buffers(pm, pe_type)
        groups = get_param_groups(pm, pe_type)
        pe_params = groups['pe_only'] + extra_pe
        if not pe_params:
            print(f"  WARNING: No PE params found for {pe_type}")
            return pm
        pgd_perturb(pm, pe_params, epsilon, images, labels)

    elif perturb_type == 'qkv_only':
        groups = get_param_groups(pm, pe_type)
        pgd_perturb(pm, groups['qkv_only'], epsilon, images, labels)

    elif perturb_type == 'mlp_only':
        groups = get_param_groups(pm, pe_type)
        pgd_perturb(pm, groups['mlp_only'], epsilon, images, labels)

    elif perturb_type == 'all_weights':
        # All trainable parameters with same L-inf budget
        all_params = [p for p in pm.parameters() if p.requires_grad]
        pgd_perturb(pm, all_params, epsilon, images, labels)

    return pm

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_specificity_experiment():
    results = {}

    for pe_type in PE_TYPES:
        results[pe_type] = {}
        print(f"\n{'='*60}")
        print(f"PE TYPE: {pe_type.upper()}")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            results[pe_type][str(seed)] = {}

            model = load_model(pe_type, seed)
            clean_attn = get_attention_distributions(model, ref_loader)
            clean_acc = measure_accuracy(model, val_loader)
            print(f"    Clean accuracy: {clean_acc:.2f}%")

            for perturb_type in PERTURBATION_TYPES:
                results[pe_type][str(seed)][perturb_type] = {
                    'epsilons': EPSILONS,
                    'accuracies': [],
                    'ads_mean': [],
                    'ads_mid_layer': [],
                    'ads_layer4': [],
                }

                print(f"\n    Perturbation: {perturb_type.upper()}")
                for eps in EPSILONS:
                    if eps == 0.0:
                        ads = {'mean_ads': 0.0, 'mid_layer_ads': 0.0,
                               'layer4_ads': 0.0, 'per_layer_ads': [0.0]*12}
                        acc = clean_acc
                    else:
                        pm = perturb_model(model, pe_type, perturb_type, eps)
                        perturbed_attn = get_attention_distributions(pm, ref_loader)
                        ads = compute_ads(clean_attn, perturbed_attn)
                        acc = measure_accuracy(pm, val_loader)
                        del pm
                        torch.cuda.empty_cache()

                    results[pe_type][str(seed)][perturb_type]['accuracies'].append(acc)
                    results[pe_type][str(seed)][perturb_type]['ads_mean'].append(ads['mean_ads'])
                    results[pe_type][str(seed)][perturb_type]['ads_mid_layer'].append(ads['mid_layer_ads'])
                    results[pe_type][str(seed)][perturb_type]['ads_layer4'].append(ads['layer4_ads'])

                    print(f"      ε={eps:.3f}: acc={acc:.1f}%, "
                          f"ADS(mean)={ads['mean_ads']:.4f}, "
                          f"ADS(mid)={ads['mid_layer_ads']:.4f}, "
                          f"ADS(L4)={ads['layer4_ads']:.4f}")

            del model
            torch.cuda.empty_cache()

        # Save after each PE type
        with open(SAVE_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {SAVE_PATH}")

    return results

# ============================================================
# ANALYSIS
# ============================================================
def analyze_specificity(results):
    seeds = [str(s) for s in SEEDS]
    print("\n" + "=" * 80)
    print("SPECIFICITY ANALYSIS: ADS(PE) vs ADS(non-PE)")
    print("=" * 80)

    for pe_type in PE_TYPES:
        print(f"\n{pe_type.upper()}:")
        print(f"  {'ε':>6}  {'PE_only':>10}  {'QKV_only':>10}  {'MLP_only':>10}  {'All_weights':>12}  {'Ratio PE/QKV':>13}")
        print(f"  {'-'*65}")

        epsilons = EPSILONS
        for i, eps in enumerate(epsilons):
            row = {}
            for pt in PERTURBATION_TYPES:
                vals = [results[pe_type][s][pt]['ads_layer4'][i] for s in seeds]
                row[pt] = np.mean(vals)

            ratio = row['pe_only'] / max(row['qkv_only'], 1e-8)
            print(f"  {eps:>6.3f}  {row['pe_only']:>10.4f}  {row['qkv_only']:>10.4f}  "
                  f"{row['mlp_only']:>10.4f}  {row['all_weights']:>12.4f}  {ratio:>13.2f}x")

        print(f"\n  VERDICT for {pe_type}:")
        # At eps=0.1 (typical operating point)
        idx = epsilons.index(0.1)
        pe_ads  = np.mean([results[pe_type][s]['pe_only']['ads_layer4'][idx] for s in seeds])
        qkv_ads = np.mean([results[pe_type][s]['qkv_only']['ads_layer4'][idx] for s in seeds])
        mlp_ads = np.mean([results[pe_type][s]['mlp_only']['ads_layer4'][idx] for s in seeds])
        all_ads = np.mean([results[pe_type][s]['all_weights']['ads_layer4'][idx] for s in seeds])

        print(f"    At ε=0.1: PE={pe_ads:.4f}, QKV={qkv_ads:.4f}, MLP={mlp_ads:.4f}, ALL={all_ads:.4f}")
        if pe_ads > 3 * max(qkv_ads, mlp_ads):
            print(f"    -> ADS is PE-SPECIFIC (PE/QKV ratio = {pe_ads/max(qkv_ads,1e-8):.1f}x) -- STRONG RESULT")
        elif pe_ads > max(qkv_ads, mlp_ads):
            print(f"    -> ADS shows PE PREFERENCE but not exclusive -- MODERATE RESULT")
        else:
            print(f"    -> ADS is NON-SPECIFIC -- reacts to any perturbation -- WEAK CLAIM")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    print("ADS Specificity Experiment")
    print(f"PE types: {PE_TYPES}")
    print(f"Seeds: {SEEDS}")
    print(f"Perturbation types: {PERTURBATION_TYPES}")
    print(f"Epsilons: {EPSILONS}")
    print()

    results = run_specificity_experiment()
    analyze_specificity(results)
    print("\n✅ Specificity experiment complete!")
