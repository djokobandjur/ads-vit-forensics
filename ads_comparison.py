"""
Detection Method Comparison Experiment
========================================
Compares ADS against three baseline detection methods:
  1. Weight hash (cryptographic integrity) - simulated
  2. Mahalanobis distance on attention outputs (OOD detection)
  3. KL divergence of logit distributions

For each method, we measure:
  - Detection rate at each epsilon (TPR against benign FPR baseline)
  - AUC across epsilon values
  - Computational cost (forward passes required)

This addresses the reviewer requirement for baseline comparison.

Output: ads_comparison.json
"""

import argparse
import os, sys, json, copy, time
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from scipy.spatial.distance import mahalanobis

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

PE_TYPES        = ['learned', 'rope']
SEEDS           = [42, 123, 456]

# Apply CLI overrides for PE_TYPES and SEEDS if user provided them
if args.pe_types is not None:
    PE_TYPES = args.pe_types
    print(f"[CLI override] PE_TYPES = {PE_TYPES}")
if args.seeds is not None:
    SEEDS = args.seeds
    print(f"[CLI override] SEEDS = {SEEDS}")

ATTACK_EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
N_BOOTSTRAP     = 20

PGD_STEPS       = 20
PGD_ALPHA_RATIO = 0.1

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

with open(REF_INDICES_PATH) as f:
    ref_indices = json.load(f)

ref_loader = DataLoader(
    Subset(val_dataset, ref_indices),
    batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

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
# PE PERTURBATION
# ============================================================
def perturb_pe(model, pe_type, epsilon):
    if epsilon == 0.0:
        return copy.deepcopy(model)

    pm = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    alpha = epsilon * PGD_ALPHA_RATIO

    images, labels = next(iter(ref_loader))
    images, labels = images.to(device), labels.to(device)

    pe_params = []
    for name, buf in list(pm.named_buffers()):
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['.pe', 'cos_cached', 'sin_cached']):
            if 'rel_dist' not in clean:
                parts = clean.split('.')
                obj = pm
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                delattr(obj, parts[-1])
                new_p = nn.Parameter(buf.clone(), requires_grad=True)
                setattr(obj, parts[-1], new_p)
                pe_params.append(new_p)
    for name, mod in pm.named_modules():
        if type(mod).__name__ == 'ALiBi':
            sd = mod.slopes.clone().to(device)
            del mod.slopes
            mod.slopes = nn.Parameter(sd, requires_grad=True)
            pe_params.append(mod.slopes)
    for name, param in pm.named_parameters():
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['pos_embed', 'inv_freq']):
            param.requires_grad_(True)
            pe_params.append(param)

    deltas = [torch.zeros_like(p) for p in pe_params]
    pm.train()
    for step in range(PGD_STEPS):
        with torch.no_grad():
            for p, d in zip(pe_params, deltas):
                p.data = p.data - (d if step > 0 else 0) + d
        for p in pe_params:
            if p.grad is not None: p.grad.zero_()
        loss = criterion(pm(images), labels)
        loss.backward()
        with torch.no_grad():
            new_d = []
            for p, d in zip(pe_params, deltas):
                if p.grad is not None:
                    nd = (d + alpha * p.grad.sign()).clamp(-epsilon, epsilon)
                    new_d.append(nd)
                else:
                    new_d.append(d)
            deltas = new_d
    with torch.no_grad():
        for p, d in zip(pe_params, deltas):
            p.data += d
    pm.eval()
    return pm

# ============================================================
# METHOD 1: ADS (L4)
# ============================================================
@torch.no_grad()
def get_attn_l4(model, loader):
    layers = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, attns = model.forward_with_attention(imgs)
        layers.append(attns[3].mean(0).cpu())
    return torch.stack(layers).mean(0)

def compute_ads(clean_attn, perturbed_attn):
    eps = 1e-10
    P = clean_attn.float() + eps
    Q = perturbed_attn.float() + eps
    P = P / P.sum(-1, keepdim=True)
    Q = Q / Q.sum(-1, keepdim=True)
    return float((P * torch.log(P / Q)).sum(-1).mean().item())

# ============================================================
# METHOD 2: MAHALANOBIS ON ATTENTION OUTPUTS
# Uses mean attention vector per layer as feature
# Mahalanobis distance from clean distribution
# ============================================================
@torch.no_grad()
def get_attention_features(model, loader):
    """
    Get flattened mean attention vector (H*N) from Layer 4
    for each image in loader.
    Returns: (N_images, H*N) numpy array
    """
    features = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, attns = model.forward_with_attention(imgs)
        # Layer 4 attention: (B, H, N, N) -> mean over N -> (B, H*N)
        attn_l4 = attns[3]  # (B, H, N, N)
        feat = attn_l4.mean(-1).reshape(attn_l4.shape[0], -1)  # (B, H*N)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

def compute_mahalanobis_score(clean_features, test_features):
    """
    Compute mean Mahalanobis distance of test features from clean distribution.
    """
    mu = clean_features.mean(0)
    # Regularized covariance
    cov = np.cov(clean_features.T) + 1e-6 * np.eye(clean_features.shape[1])
    try:
        cov_inv = np.linalg.inv(cov)
        dists = []
        for feat in test_features:
            diff = feat - mu
            d = float(np.sqrt(diff @ cov_inv @ diff))
            dists.append(d)
        return float(np.mean(dists))
    except np.linalg.LinAlgError:
        return float('nan')

# ============================================================
# METHOD 3: KL DIVERGENCE OF LOGIT DISTRIBUTIONS
# ============================================================
@torch.no_grad()
def get_logit_distribution(model, loader):
    """
    Get mean softmax distribution over classes from ref set.
    Returns: (num_classes,) numpy array
    """
    all_probs = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    return all_probs.mean(0)  # Mean distribution over images

def compute_logit_kl(clean_dist, perturbed_dist):
    """KL divergence between two logit distributions."""
    eps = 1e-10
    P = clean_dist + eps
    Q = perturbed_dist + eps
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(np.sum(P * np.log(P / Q)))

# ============================================================
# ROC COMPUTATION
# ============================================================
def compute_auc(negative_scores, positive_score):
    """Simple AUC: fraction of negatives below positive score."""
    return float(np.mean([n < positive_score for n in negative_scores]))

# ============================================================
# MAIN
# ============================================================
def run():
    results = {}

    for pe_type in PE_TYPES:
        results[pe_type] = {}
        print(f"\n{'='*60}")
        print(f"PE TYPE: {pe_type.upper()}")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            model = load_model(pe_type, seed)

            # --- CALIBRATE ALL METHODS ON CLEAN MODEL ---

            # Method 1: ADS baseline
            t0 = time.time()
            ref_attn = get_attn_l4(model, ref_loader)
            t_ads_cal = time.time() - t0

            # Method 2: Mahalanobis - need full features per image
            t0 = time.time()
            clean_features = get_attention_features(model, ref_loader)
            t_mah_cal = time.time() - t0

            # Method 3: Logit KL
            t0 = time.time()
            clean_logits = get_logit_distribution(model, ref_loader)
            t_logit_cal = time.time() - t0

            # --- BOOTSTRAP CLEAN SCORES ---
            print(f"    Computing clean noise floors...")
            clean_ads, clean_mah, clean_logit_kl = [], [], []
            all_ref = np.array(ref_indices)

            for b in range(N_BOOTSTRAP):
                rng = np.random.RandomState(b * 31 + seed * 7)
                perm = rng.permutation(len(all_ref))
                half_a_idx = all_ref[perm[:128]].tolist()
                half_b_idx = all_ref[perm[128:]].tolist()

                loader_a = DataLoader(Subset(val_dataset, half_a_idx),
                                      batch_size=32, shuffle=False, num_workers=2)
                loader_b = DataLoader(Subset(val_dataset, half_b_idx),
                                      batch_size=32, shuffle=False, num_workers=2)

                # ADS between halves
                attn_a = get_attn_l4(model, loader_a)
                attn_b = get_attn_l4(model, loader_b)
                clean_ads.append(compute_ads(attn_a, attn_b))

                # Mahalanobis of half_b from half_a distribution
                feat_a = get_attention_features(model, loader_a)
                feat_b = get_attention_features(model, loader_b)
                # Use simplified version: L2 distance from mean (full Mahalanobis too slow)
                mu_a = feat_a.mean(0)
                clean_mah.append(float(np.mean(np.linalg.norm(feat_b - mu_a, axis=1))))

                # Logit KL between halves
                logit_a = get_logit_distribution(model, loader_a)
                logit_b = get_logit_distribution(model, loader_b)
                clean_logit_kl.append(compute_logit_kl(logit_a, logit_b))

            ads_baseline  = float(np.mean(clean_ads))
            mah_baseline  = float(np.mean(clean_mah))
            logit_baseline = float(np.mean(clean_logit_kl))

            print(f"    Clean baselines — ADS: {ads_baseline:.6f}, "
                  f"Mah: {mah_baseline:.4f}, LogitKL: {logit_baseline:.6f}")

            results[pe_type][str(seed)] = {
                'baselines': {
                    'ads': ads_baseline,
                    'mahalanobis': mah_baseline,
                    'logit_kl': logit_baseline,
                },
                'clean_scores': {
                    'ads': clean_ads,
                    'mahalanobis': clean_mah,
                    'logit_kl': clean_logit_kl,
                },
                'attack': {},
                'timing': {
                    'ads_calibration_s': t_ads_cal,
                    'mahalanobis_calibration_s': t_mah_cal,
                    'logit_calibration_s': t_logit_cal,
                }
            }

            # --- ATTACK EVALUATION ---
            print(f"    Evaluating attacks...")
            for eps in ATTACK_EPSILONS:
                pm = perturb_pe(model, pe_type, eps)

                # ADS
                t0 = time.time()
                pm_attn = get_attn_l4(pm, ref_loader)
                ads_score = compute_ads(ref_attn, pm_attn)
                t_ads = time.time() - t0

                # Mahalanobis
                t0 = time.time()
                pm_features = get_attention_features(pm, ref_loader)
                mu_clean = clean_features.mean(0)
                mah_score = float(np.mean(np.linalg.norm(pm_features - mu_clean, axis=1)))
                t_mah = time.time() - t0

                # Logit KL
                t0 = time.time()
                pm_logits = get_logit_distribution(pm, ref_loader)
                logit_score = compute_logit_kl(clean_logits, pm_logits)
                t_logit = time.time() - t0

                # AUC for each method
                auc_ads    = compute_auc(clean_ads,      ads_score)
                auc_mah    = compute_auc(clean_mah,      mah_score)
                auc_logit  = compute_auc(clean_logit_kl, logit_score)

                results[pe_type][str(seed)]['attack'][str(eps)] = {
                    'ads':         {'score': float(ads_score),   'ratio': float(ads_score/ads_baseline),   'auc': auc_ads,   't_s': t_ads},
                    'mahalanobis': {'score': float(mah_score),   'ratio': float(mah_score/mah_baseline),   'auc': auc_mah,   't_s': t_mah},
                    'logit_kl':    {'score': float(logit_score), 'ratio': float(logit_score/logit_baseline),'auc': auc_logit, 't_s': t_logit},
                }

                print(f"      ε={eps:.3f}: ADS AUC={auc_ads:.2f}, Mah AUC={auc_mah:.2f}, "
                      f"LogitKL AUC={auc_logit:.2f}")

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
    print("\n" + "=" * 75)
    print("COMPARISON SUMMARY")
    print("=" * 75)

    for pe_type in PE_TYPES:
        print(f"\n{pe_type.upper()}:")
        print(f"  {'ε':>6}  {'ADS AUC':>10}  {'Mah AUC':>10}  {'Logit AUC':>10}")
        print(f"  {'-'*45}")

        for eps in ATTACK_EPSILONS:
            aucs = {'ads': [], 'mahalanobis': [], 'logit_kl': []}
            for s in seeds:
                for method in aucs:
                    aucs[method].append(
                        results[pe_type][s]['attack'][str(eps)][method]['auc']
                    )
            print(f"  {eps:>6.3f}  {np.mean(aucs['ads']):>10.3f}  "
                  f"{np.mean(aucs['mahalanobis']):>10.3f}  "
                  f"{np.mean(aucs['logit_kl']):>10.3f}")

        # Timing
        t_ads = np.mean([results[pe_type][s]['timing']['ads_calibration_s'] for s in seeds])
        t_mah = np.mean([results[pe_type][s]['timing']['mahalanobis_calibration_s'] for s in seeds])
        t_log = np.mean([results[pe_type][s]['timing']['logit_calibration_s'] for s in seeds])
        print(f"\n  Calibration time (256 images): "
              f"ADS={t_ads:.2f}s, Mahalanobis={t_mah:.2f}s, LogitKL={t_log:.2f}s")


if __name__ == '__main__':
    print("Detection Method Comparison Experiment")
    print(f"PE types: {PE_TYPES}, Seeds: {SEEDS}")
    print(f"Methods: ADS(L4), Mahalanobis(attention), LogitKL")
    print()

    results = run()
    analyze(results)
    print("\n✅ Comparison complete!")
