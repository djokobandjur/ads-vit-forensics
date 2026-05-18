"""
ADS Fine-Grid Threshold Experiment
====================================
Tests the "universality" claim of eps=0.005 with finer epsilon resolution
and log-log interpolation of the true crossing threshold.

Addresses reviewer concern: coarse grid may mask PE-type differences in
the actual ADS > 10x baseline crossing point.

Outputs:
    - ads_threshold_fine.json: interpolated thresholds per PE type/seed
    - Statistical test: are thresholds significantly different across PE types?
"""

import argparse
import os, sys, json, copy
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
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
        required=True, help='Path to ImageNet-100 val directory in ImageFolder format',
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

PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS    = [42, 123, 456]


# Apply CLI overrides for PE_TYPES and SEEDS if user provided them
if args.pe_types is not None:
    PE_TYPES = args.pe_types
    print(f"[CLI override] PE_TYPES = {PE_TYPES}")
if args.seeds is not None:
    SEEDS = args.seeds
    print(f"[CLI override] SEEDS = {SEEDS}")

# Fine grid in critical range + coarse grid beyond
EPSILONS_FINE = [0.0, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01,
                 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

PGD_STEPS       = 20
PGD_ALPHA_RATIO = 0.1
N_REF_IMAGES    = 256

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

if os.path.exists(REF_INDICES_PATH):
    with open(REF_INDICES_PATH) as f:
        ref_indices = json.load(f)
    print(f"Loaded {len(ref_indices)} reference indices")
else:
    torch.manual_seed(42)
    ref_indices = torch.randperm(len(val_dataset))[:N_REF_IMAGES].tolist()
    with open(REF_INDICES_PATH, 'w') as f:
        json.dump(ref_indices, f)
    print(f"Generated {N_REF_IMAGES} reference indices")

ref_dataset = Subset(val_dataset, ref_indices)
ref_loader  = DataLoader(ref_dataset, batch_size=32, shuffle=False,
                         num_workers=4, pin_memory=True)

# ============================================================
# MODEL LOADING
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
# ATTENTION + ADS
# ============================================================
@torch.no_grad()
def get_attention_distributions(model, loader):
    all_attentions = [[] for _ in range(12)]
    for images, _ in loader:
        images = images.to(device)
        _, attentions = model.forward_with_attention(images)
        for l, attn in enumerate(attentions):
            all_attentions[l].append(attn.mean(0).cpu())
    return [torch.stack(all_attentions[l]).mean(0) for l in range(12)]

def compute_ads_l4(clean_attn, perturbed_attn):
    eps_kl = 1e-10
    P = clean_attn[3].float() + eps_kl   # Layer 4 (0-indexed: 3)
    Q = perturbed_attn[3].float() + eps_kl
    P = P / P.sum(-1, keepdim=True)
    Q = Q / Q.sum(-1, keepdim=True)
    return float((P * torch.log(P / Q)).sum(-1).mean().item())

# ============================================================
# PERTURBATION (same as main ADS experiment)
# ============================================================
def convert_pe_buffers_to_params(model, pe_type):
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
    for name, module in model.named_modules():
        if type(module).__name__ == 'ALiBi':
            slopes_data = module.slopes.clone().to(device)
            del module.slopes
            module.slopes = nn.Parameter(slopes_data, requires_grad=True)
            pe_params.append(module.slopes)
    return pe_params

def perturb_pe(model, pe_type, epsilon):
    if epsilon == 0.0:
        return copy.deepcopy(model)

    pm = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    alpha = epsilon * PGD_ALPHA_RATIO

    images, labels = next(iter(ref_loader))
    images, labels = images.to(device), labels.to(device)

    # Get PE parameters
    extra = convert_pe_buffers_to_params(pm, pe_type)
    pe_params = extra[:]
    for name, param in pm.named_parameters():
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['pos_embed', 'inv_freq']):
            param.requires_grad_(True)
            pe_params.append(param)

    if not pe_params:
        print(f"  WARNING: No PE params for {pe_type}")
        return pm

    deltas = [torch.zeros_like(p) for p in pe_params]
    pm.train()

    for step in range(PGD_STEPS):
        with torch.no_grad():
            for p, d in zip(pe_params, deltas):
                p.data = p.data - (d if step > 0 else 0) + d
        for p in pe_params:
            if p.grad is not None:
                p.grad.zero_()
        out = pm(images)
        loss = criterion(out, labels)
        loss.backward()
        with torch.no_grad():
            new_deltas = []
            for p, d in zip(pe_params, deltas):
                if p.grad is not None:
                    nd = d + alpha * p.grad.sign()
                    nd = nd.clamp(-epsilon, epsilon)
                    new_deltas.append(nd)
                else:
                    new_deltas.append(d)
            deltas = new_deltas

    with torch.no_grad():
        for p, d in zip(pe_params, deltas):
            p.data += d

    pm.eval()
    return pm

# ============================================================
# LOG-LOG INTERPOLATION OF CROSSING THRESHOLD
# ============================================================
def interpolate_threshold(epsilons, ads_values, baseline, multiplier=10.0):
    """
    Find interpolated epsilon where ADS first crosses multiplier * baseline.
    Uses log-log interpolation between the two bracketing points.
    Returns None if threshold is never crossed or always exceeded.
    """
    threshold = multiplier * baseline
    eps_arr = np.array(epsilons)
    ads_arr = np.array(ads_values)

    # Find first crossing point
    cross_idx = None
    for i in range(1, len(ads_arr)):
        if ads_arr[i-1] < threshold <= ads_arr[i]:
            cross_idx = i
            break

    if cross_idx is None:
        return None  # Never crossed within tested range

    # Log-log interpolation between cross_idx-1 and cross_idx
    eps_lo = eps_arr[cross_idx - 1]
    eps_hi = eps_arr[cross_idx]
    ads_lo = ads_arr[cross_idx - 1]
    ads_hi = ads_arr[cross_idx]

    if eps_lo <= 0 or ads_lo <= 0:
        # Fall back to linear interpolation
        frac = (threshold - ads_lo) / (ads_hi - ads_lo)
        return float(eps_lo + frac * (eps_hi - eps_lo))

    # Log-log interpolation
    log_eps_lo = np.log(eps_lo)
    log_eps_hi = np.log(eps_hi)
    log_ads_lo = np.log(ads_lo)
    log_ads_hi = np.log(ads_hi)
    log_threshold = np.log(threshold)

    frac = (log_threshold - log_ads_lo) / (log_ads_hi - log_ads_lo)
    log_eps_cross = log_eps_lo + frac * (log_eps_hi - log_eps_lo)
    return float(np.exp(log_eps_cross))

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_fine_grid_experiment():
    results = {}

    for pe_type in PE_TYPES:
        results[pe_type] = {}
        print(f"\n{'='*60}")
        print(f"PE TYPE: {pe_type.upper()}")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            results[pe_type][str(seed)] = {
                'epsilons': EPSILONS_FINE,
                'ads_l4': [],
                'interpolated_threshold': None,
                'baseline_ads': None,
            }

            model = load_model(pe_type, seed)
            clean_attn = get_attention_distributions(model, ref_loader)

            ads_values = []
            for eps in EPSILONS_FINE:
                if eps == 0.0:
                    ads = 0.0
                else:
                    pm = perturb_pe(model, pe_type, eps)
                    perturbed_attn = get_attention_distributions(pm, ref_loader)
                    ads = compute_ads_l4(clean_attn, perturbed_attn)
                    del pm
                    torch.cuda.empty_cache()

                ads_values.append(ads)
                print(f"    ε={eps:.4f}: ADS(L4)={ads:.6f}")

            results[pe_type][str(seed)]['ads_l4'] = ads_values

            # Baseline: ADS at eps=0.001 (first non-zero point)
            baseline = ads_values[1]  # index 1 = eps=0.001
            results[pe_type][str(seed)]['baseline_ads'] = float(baseline)

            # Interpolated threshold
            thresh = interpolate_threshold(EPSILONS_FINE[1:], ads_values[1:], baseline)
            results[pe_type][str(seed)]['interpolated_threshold'] = thresh
            print(f"    Baseline ADS(L4) at ε=0.001: {baseline:.6f}")
            print(f"    Interpolated 10x threshold: ε = {thresh:.5f}" if thresh else
                  "    Threshold not crossed in tested range")

            del model
            torch.cuda.empty_cache()

        with open(SAVE_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {SAVE_PATH}")

    return results

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
def analyze_thresholds(results):
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: Are thresholds significantly different?")
    print("=" * 70)

    seeds = [str(s) for s in SEEDS]
    thresholds_by_pe = {}

    for pe_type in PE_TYPES:
        thresholds = []
        for s in seeds:
            t = results[pe_type][s]['interpolated_threshold']
            if t is not None:
                thresholds.append(t)
        thresholds_by_pe[pe_type] = thresholds

        mean_t = np.mean(thresholds) if thresholds else None
        std_t  = np.std(thresholds, ddof=1) if len(thresholds) > 1 else None
        print(f"\n{pe_type.upper()}:")
        print(f"  Thresholds per seed: {[f'{t:.5f}' for t in thresholds]}")
        if mean_t:
            print(f"  Mean ± Std: {mean_t:.5f} ± {std_t:.5f}" if std_t else
                  f"  Mean: {mean_t:.5f}")

    # Kruskal-Wallis test (non-parametric, appropriate for n=3)
    groups = [thresholds_by_pe[pe] for pe in PE_TYPES if len(thresholds_by_pe[pe]) >= 2]
    if len(groups) >= 2:
        try:
            stat, p_value = stats.kruskal(*groups)
            print(f"\nKruskal-Wallis test across PE types:")
            print(f"  H = {stat:.4f}, p = {p_value:.4f}")
            if p_value > 0.05:
                print(f"  -> p > 0.05: thresholds NOT significantly different")
                print(f"     'Universality' claim is STATISTICALLY SUPPORTED ✅")
            else:
                print(f"  -> p < 0.05: thresholds ARE significantly different")
                print(f"     'Universality' claim is NOT supported — revise claim ❌")
        except Exception as e:
            print(f"  Statistical test failed: {e}")

    # Pairwise comparison
    print("\nPairwise threshold comparison:")
    pe_list = [pe for pe in PE_TYPES if thresholds_by_pe[pe]]
    for i in range(len(pe_list)):
        for j in range(i+1, len(pe_list)):
            pe_a, pe_b = pe_list[i], pe_list[j]
            t_a = np.mean(thresholds_by_pe[pe_a])
            t_b = np.mean(thresholds_by_pe[pe_b])
            ratio = t_a / t_b if t_b > 0 else float('inf')
            print(f"  {pe_a} vs {pe_b}: {t_a:.5f} vs {t_b:.5f} (ratio: {ratio:.2f}x)")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    print("ADS Fine-Grid Threshold Experiment")
    print(f"PE types: {PE_TYPES}")
    print(f"Seeds: {SEEDS}")
    print(f"Fine-grid epsilons: {EPSILONS_FINE}")
    print()

    results = run_fine_grid_experiment()
    analyze_thresholds(results)
    print("\n✅ Fine-grid experiment complete!")
