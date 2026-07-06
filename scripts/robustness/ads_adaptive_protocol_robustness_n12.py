# NOTE: Protocol-robustness n=12 canonical variant.
# Conditions: alibi, alibi_2d, alibi_2d_matched.
# Seeds: 1, 5, 7, 11, 13, 21, 42, 99, 123, 456, 2024, 31337.
# Checkpoint folders remain <condition>_seed<seed>/best_model.pth.
# For alibi_2d_matched, the checkpoint/result label is kept separate but
# the model is instantiated with pe_type='alibi_2d'.

"""
Adaptive Attacker Experiment — ImageNet-100, n=12 canonical protocol-robustness protocol
===============================================================
Tests whether an adversary who knows that ADS is being monitored can
craft PE perturbations that degrade accuracy while keeping ADS below a
detection threshold.

Default protocol-robustness configuration:
    - Dataset: ImageNet-100 validation set
    - PE types: learned, rope
    - Seeds: 1, 5, 7, 11, 13, 21, 42, 99, 123, 456, 2024, 31337
    - Checkpoint: <models_dir>/<pe_type>_seed<seed>/best_model.pth
    - Reference set: 256 fixed ImageNet-100 validation images

Attack loss:
    L_adaptive = -L_CE(logits, y) + lambda * ADS(delta)

where lambda controls the trade-off between accuracy damage and ADS evasion.

Lambda sweep:
    {0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0}

For each epsilon/lambda/seed, the script reports:
    - attacked accuracy and clean-accuracy drop
    - ADS(L4)
    - ADS ratio relative to the benign/noise-floor baseline
    - whether the attack remains below the PE-specific detection threshold

Note:
    This script is ImageNet-100 only. It defaults to learned and RoPE because
    the detection-threshold and noise-floor constants below are defined for
    those PE types. Add thresholds/noise floors before extending to other PE
    families.

Output:
    ads_adaptive.json
"""

import argparse
import os, sys, json, copy
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '/content')
from full_scale_experiment_v2 import VisionTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive ADS attacker experiment (ImageNet-100, n=6 default). See module docstring for details.",
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
        help='Path to ads_ref_indices_imagenet100.json. If not specified, defaults to '
             'a path alongside --output_path. If the file does not exist, '
             'it will be generated using a fixed seed.',
    )
    parser.add_argument(
        '--pe_types',
        type=str,
        nargs='+',
        default=None,
        help='Optional. Condition labels to evaluate. For the canonical protocol, use: alibi alibi_2d alibi_2d_matched. If omitted, uses the script default list.',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        help='Optional. Random seeds to evaluate. If omitted, uses the '
             'hardcoded SEEDS list defined in the script (n=6 paper configuration).',
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
        os.path.dirname(args.output_path), 'ads_ref_indices_imagenet100.json'
    )

os.makedirs(os.path.dirname(SAVE_PATH) or '.', exist_ok=True)

# Focus on Learned PE (most vulnerable, most interesting stealth scenario)
# and RoPE (robust baseline)
PE_TYPES = ['alibi', 'alibi_2d', 'alibi_2d_matched']
SEEDS = [1, 5, 7, 11, 13, 21, 42, 99, 123, 456, 2024, 31337]



# Protocol-robustness canonical CIFAR-100 checkpoint labels.
# These are result/checkpoint-folder labels. The model architecture used
# to instantiate a checkpoint can differ from the reporting label. In
# particular, matched 2D-ALiBi checkpoints use the same architecture as
# the canonical 2D-ALiBi checkpoints but remain a separate reporting
# condition in the output JSON.
MODEL_PE_ALIAS = {
    'alibi_2d_matched': 'alibi_2d',
}

def get_model_pe_type(condition_label):
    return MODEL_PE_ALIAS.get(condition_label, condition_label)

# Apply CLI overrides for PE_TYPES and SEEDS if user provided them
if args.pe_types is not None:
    PE_TYPES = args.pe_types
    print(f"[CLI override] PE_TYPES = {PE_TYPES}")
if args.seeds is not None:
    SEEDS = args.seeds
    print(f"[CLI override] SEEDS = {SEEDS}")

# Epsilon values to test (fixed budget)
EPSILONS = [0.05, 0.1, 0.2]

# Lambda sweep: 0 = standard PGD, higher = more ADS evasion
LAMBDAS  = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

PGD_STEPS       = 20
PGD_ALPHA_RATIO = 0.1
N_REF_IMAGES    = 256

# Detection thresholds from ROC analysis (worst-case benign = blur sigma=3)
DETECTION_THRESHOLD = {
    'learned': 2.65,  # x baseline
    'rope':    5.01,  # x baseline
}

# Clean benign/noise-floor ADS(L4) baselines used for ADS-ratio reporting.
NOISE_FLOOR = {
    'learned': 0.015498,
    'rope':    0.013538,
}

# Canonical ALiBi robustness defaults: use a 10x ratio criterion.
# For a publication-quality adaptive/evasion analysis, replace these
# fallback noise floors with calibrated benign-shift values.
for _canonical_pe in ['alibi', 'alibi_2d', 'alibi_2d_matched']:
    DETECTION_THRESHOLD.setdefault(_canonical_pe, 10.0)
    NOISE_FLOOR.setdefault(_canonical_pe, 1e-6)

unsupported_pe = [pe for pe in PE_TYPES if pe not in DETECTION_THRESHOLD or pe not in NOISE_FLOOR]
if unsupported_pe:
    raise ValueError(
        f"Adaptive attacker thresholds/noise floors are defined only for "
        f"{sorted(DETECTION_THRESHOLD)}; unsupported PE types: {unsupported_pe}"
    )

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
    print(f"Loaded {len(ref_indices)} reference indices from {REF_INDICES_PATH}")
else:
    torch.manual_seed(42)
    ref_indices = torch.randperm(len(val_dataset))[:N_REF_IMAGES].tolist()
    with open(REF_INDICES_PATH, 'w') as f:
        json.dump(ref_indices, f)
    print(f"Generated and saved {N_REF_IMAGES} reference indices to {REF_INDICES_PATH}")

if max(ref_indices) >= len(val_dataset):
    raise ValueError(
        f"Reference index file {REF_INDICES_PATH} is incompatible with this dataset: "
        f"max index {max(ref_indices)} >= dataset size {len(val_dataset)}"
    )

ref_loader = DataLoader(
    Subset(val_dataset, ref_indices),
    batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

print(f"Reference: {len(ref_indices)} images, Val: {len(val_dataset)} images")



def convert_any_slope_buffers_to_params(model):
    """Convert slope-like buffers in ALiBi/2D-ALiBi modules to Parameters."""
    slope_params = []
    for module in model.modules():
        for attr_name, buf in list(module._buffers.items()):
            if buf is None:
                continue
            lname = attr_name.lower()
            if 'slope' in lname and 'rel_dist' not in lname:
                del module._buffers[attr_name]
                new_param = nn.Parameter(buf.clone().to(device), requires_grad=True)
                setattr(module, attr_name, new_param)
                slope_params.append(new_param)
    return slope_params

# ============================================================
# MODEL
# ============================================================
def load_model(pe_type, seed):
    model = VisionTransformer(
        img_size=224, patch_size=16, num_classes=100,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, pe_type=get_model_pe_type(pe_type)
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
def get_attn_l4(model, loader):
    layers = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, attns = model.forward_with_attention(imgs)
        layers.append(attns[3].mean(0).cpu())
    return torch.stack(layers).mean(0)

def compute_ads_l4_differentiable(clean_attn, perturbed_model, loader):
    """
    Compute ADS(L4) in a differentiable way for use in adaptive loss.
    Returns a scalar tensor with gradients.
    """
    eps_kl = 1e-10
    kl_total = torch.tensor(0.0, device=device)
    n_batches = 0

    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, attns = perturbed_model.forward_with_attention(imgs)
        attn_perturbed = attns[3]  # (B, H, N, N)

        # Clean attention (no grad needed)
        P = clean_attn.to(device).float() + eps_kl  # (H, N, N)
        P = P / P.sum(-1, keepdim=True)

        # Perturbed attention (differentiable)
        Q = attn_perturbed.mean(0).float() + eps_kl  # (H, N, N)
        Q = Q / Q.sum(-1, keepdim=True)

        kl = (P * torch.log(P / Q)).sum(-1).mean()
        kl_total = kl_total + kl
        n_batches += 1

    return kl_total / n_batches

@torch.no_grad()
def compute_ads_l4_scalar(clean_attn, perturbed_attn):
    eps = 1e-10
    P = clean_attn.float() + eps
    Q = perturbed_attn.float() + eps
    P = P / P.sum(-1, keepdim=True)
    Q = Q / Q.sum(-1, keepdim=True)
    return float((P * torch.log(P / Q)).sum(-1).mean().item())

# ============================================================
# ACCURACY
# ============================================================
@torch.no_grad()
def measure_accuracy(model, loader):
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, pred = model(imgs).max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

# ============================================================
# PE PARAMETER SETUP
# ============================================================
def get_pe_params(model, pe_type):
    """Convert PE buffers to params and return list of PE params."""
    pe_params = []
    pe_params.extend(convert_any_slope_buffers_to_params(model))
    for name, buf in list(model.named_buffers()):
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['.pe', 'cos_cached', 'sin_cached', 'slope']):
            if 'rel_dist' not in clean:
                parts = clean.split('.')
                obj = model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                delattr(obj, parts[-1])
                new_p = nn.Parameter(buf.clone(), requires_grad=True)
                setattr(obj, parts[-1], new_p)
                pe_params.append(new_p)
    for name, mod in model.named_modules():
        if type(mod).__name__ == 'ALiBi' and 'slopes' in getattr(mod, '_buffers', {}):
            sd = mod.slopes.clone().to(device)
            del mod.slopes
            mod.slopes = nn.Parameter(sd, requires_grad=True)
            pe_params.append(mod.slopes)
    for name, param in model.named_parameters():
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['pos_embed', 'inv_freq', 'slope']):
            param.requires_grad_(True)
            pe_params.append(param)
    return pe_params

# ============================================================
# ADAPTIVE PGD ATTACK
# ============================================================
def adaptive_pgd(model, clean_attn, pe_type, epsilon, lam, n_steps=PGD_STEPS):
    """
    Adaptive PGD attack with ADS regularization.
    
    Loss: -L_CE + lambda * ADS(delta)
    
    Maximizes accuracy damage while minimizing ADS signal.
    lambda=0: standard PGD
    lambda->inf: pure ADS minimization
    """
    pm = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    alpha = epsilon * PGD_ALPHA_RATIO

    # Get attack batch
    images, labels = next(iter(ref_loader))
    images, labels = images.to(device), labels.to(device)

    pe_params = get_pe_params(pm, pe_type)
    if not pe_params:
        print(f"  WARNING: No PE params for {pe_type}")
        return pm

    deltas = [torch.zeros_like(p) for p in pe_params]
    pm.train()

    for step in range(n_steps):
        # Apply current delta
        with torch.no_grad():
            for p, d in zip(pe_params, deltas):
                p.data = p.data - (d if step > 0 else 0) + d

        for p in pe_params:
            if p.grad is not None:
                p.grad.zero_()

        # Forward pass
        out = pm(images)
        ce_loss = criterion(out, labels)

        if lam > 0:
            # Compute ADS in differentiable way (on ref batch only for speed)
            eps_kl = 1e-10
            _, attns = pm.forward_with_attention(images)
            attn_pert = attns[3].mean(0).float() + eps_kl  # (H, N, N)
            attn_pert = attn_pert / attn_pert.sum(-1, keepdim=True)

            attn_pert = attn_pert.to(device)
            P = clean_attn.detach().to(device).float() + eps_kl
            P = P / P.sum(-1, keepdim=True)

            ads_loss = (P * torch.log(P / attn_pert)).sum(-1).mean()
            total_loss = -ce_loss + lam * ads_loss
        else:
            total_loss = -ce_loss  # Standard PGD

        total_loss.backward()

        with torch.no_grad():
            new_d = []
            for p, d in zip(pe_params, deltas):
                if p.grad is not None:
                    nd = (d + alpha * p.grad.sign()).clamp(-epsilon, epsilon)
                    new_d.append(nd)
                else:
                    new_d.append(d)
            deltas = new_d

    # Apply final delta
    with torch.no_grad():
        for p, d in zip(pe_params, deltas):
            p.data += d

    pm.eval()
    return pm

# ============================================================
# MAIN
# ============================================================
def run():
    results = {}

    for pe_type in PE_TYPES:
        results[pe_type] = {}
        det_threshold = DETECTION_THRESHOLD[pe_type]
        print(f"\n{'='*60}")
        print(f"PE TYPE: {pe_type.upper()}")
        print(f"Detection threshold: {det_threshold}x baseline")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            results[pe_type][str(seed)] = {}

            model = load_model(pe_type, seed)
            clean_attn = get_attn_l4(model, ref_loader)
            clean_acc = measure_accuracy(model, val_loader)
            print(f"    Clean accuracy: {clean_acc:.2f}%")

            # Get clean noise floor (from roc_v2 results - use fixed value)
            # Learned: 0.015498, RoPE: 0.013538
            noise_floor = NOISE_FLOOR[pe_type]

            for eps in EPSILONS:
                results[pe_type][str(seed)][str(eps)] = {}
                print(f"\n    ε={eps}:")

                for lam in LAMBDAS:
                    pm = adaptive_pgd(model, clean_attn, get_model_pe_type(pe_type), eps, lam)

                    # Measure ADS
                    pm_attn = get_attn_l4(pm, ref_loader)
                    ads = compute_ads_l4_scalar(clean_attn, pm_attn)
                    ads_ratio = ads / noise_floor

                    # Measure accuracy
                    acc = measure_accuracy(pm, val_loader)
                    acc_drop = clean_acc - acc

                    evades = ads_ratio < det_threshold
                    print(f"      λ={lam:5.1f}: acc_drop={acc_drop:6.2f}pp, "
                          f"ADS={ads:.6f} ({ads_ratio:.2f}x), "
                          f"{'EVADES ✅' if evades else 'DETECTED ❌'}")

                    results[pe_type][str(seed)][str(eps)][str(lam)] = {
                        'lambda': lam,
                        'acc_drop': float(acc_drop),
                        'accuracy': float(acc),
                        'ads_l4': float(ads),
                        'ads_ratio': float(ads_ratio),
                        'evades_detection': bool(evades),
                    }

                    del pm
                    torch.cuda.empty_cache()

            del model
            torch.cuda.empty_cache()

        with open(SAVE_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {SAVE_PATH}")

    return results

# ============================================================
# ANALYSIS
# ============================================================
def analyze(results):
    seeds = [str(s) for s in SEEDS]
    det_thresholds = DETECTION_THRESHOLD

    print("\n" + "=" * 75)
    print("ADAPTIVE ATTACKER ANALYSIS")
    print("=" * 75)

    for pe_type in PE_TYPES:
        if pe_type not in results:
            continue
        available_seeds = [s for s in seeds if s in results[pe_type]]
        if not available_seeds:
            print(f"\n{pe_type.upper()}: no available seeds")
            continue

        tau = det_thresholds[pe_type]
        print(f"\n{pe_type.upper()} (n={len(available_seeds)} seeds, detection threshold: {tau}x baseline):")

        for eps in EPSILONS:
            eps_key = str(eps)
            print(f"\n  ε={eps}:")
            print(f"  {'λ':>6}  {'Acc drop':>10}  {'ADS ratio':>10}  {'Evades?':>10}  {'Trade-off':>20}")
            print(f"  {'-'*65}")

            for lam in LAMBDAS:
                lam_key = str(lam)
                drops, ratios, evades = [], [], []
                for s in available_seeds:
                    if eps_key not in results[pe_type][s] or lam_key not in results[pe_type][s][eps_key]:
                        continue
                    r = results[pe_type][s][eps_key][lam_key]
                    drops.append(r['acc_drop'])
                    ratios.append(r['ads_ratio'])
                    evades.append(r['evades_detection'])

                if not drops:
                    continue

                mean_drop = np.mean(drops)
                mean_ratio = np.mean(ratios)
                evades_all = all(evades)

                tradeoff = f"{mean_drop:.1f}pp @ {mean_ratio:.1f}x"
                verdict = "BYPASSES ✅" if evades_all else "detected"

                print(f"  {lam:>6.1f}  {mean_drop:>10.2f}  {mean_ratio:>10.2f}  "
                      f"{verdict:>10}  {tradeoff:>20}")

        print(f"\n  KEY FINDING for {pe_type}:")
        for eps in EPSILONS:
            eps_key = str(eps)
            best_drop_evading = 0.0
            best_lam = None
            for lam in LAMBDAS:
                lam_key = str(lam)
                drops, evades = [], []
                for s in available_seeds:
                    if eps_key not in results[pe_type][s] or lam_key not in results[pe_type][s][eps_key]:
                        continue
                    drops.append(results[pe_type][s][eps_key][lam_key]['acc_drop'])
                    evades.append(results[pe_type][s][eps_key][lam_key]['evades_detection'])
                if drops and all(evades) and np.mean(drops) > best_drop_evading:
                    best_drop_evading = float(np.mean(drops))
                    best_lam = lam

            std_drops = [
                results[pe_type][s][eps_key]['0.0']['acc_drop']
                for s in available_seeds
                if eps_key in results[pe_type][s] and '0.0' in results[pe_type][s][eps_key]
            ]
            std_drop = float(np.mean(std_drops)) if std_drops else float('nan')
            if best_lam is not None:
                print(f"    ε={eps}: Adaptive attacker (λ={best_lam}) evades with "
                      f"{best_drop_evading:.1f}pp drop "
                      f"(vs standard PGD: {std_drop:.1f}pp)")
            else:
                print(f"    ε={eps}: No lambda achieves evasion + damage "
                      f"(standard PGD: {std_drop:.1f}pp)")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    print("Adaptive Attacker Experiment")
    print(f"PE types: {PE_TYPES}, Seeds: {SEEDS}")
    print(f"Epsilons: {EPSILONS}, Lambdas: {LAMBDAS}")
    print(f"PGD steps: {PGD_STEPS}")
    print(f"Detection thresholds: {DETECTION_THRESHOLD}")
    print()

    results = run()
    analyze(results)
    print("\n✅ Adaptive attacker experiment complete!")
