"""
ADS ROC Analysis — ImageNet-100, n=6 protocol
==============================================

Evaluates the ADS detector against benign image transformations and
PE-parameter attacks using the corrected ROC design.

Dataset/protocol:
  - ImageNet-100 validation set in ImageFolder format.
  - Default PE types: Learned and RoPE.
  - Default seeds: 42, 123, 456, 789, 1011, 1213 (n=6).
  - Primary checkpoint per run: <models_dir>/<pe_type>_seed<seed>/best_model.pth.
  - Shared reference set: 256 ImageNet-100 validation images, stored in
    ads_ref_indices_imagenet100.json for reproducibility.

Correct ADS design:
  ADS measures KL(attn_clean_model(X) || attn_perturbed_model(X)) on the
  same fixed reference image set X. This isolates parameter-level PE
  perturbations from image-set variation.

Noise/negative-class design:
  - Clean noise floor: ADS between clean-model attention estimates from
    different random 128-image subsets of the fixed 256-image reference set.
    This estimates measurement variance under no parameter perturbation.
  - Benign shifts: JPEG, blur, and noise applied to the same reference images.
  - Positive class: PE-perturbed model evaluated on the same fixed reference set.

Outputs:
  - ads_roc_v2.json
    containing clean noise floors, benign ADS scores, attack ADS scores,
    ROC tables by epsilon, and overall ROC summaries.

This script is ImageNet-100 only. CIFAR-100 is intentionally not included
because the benign-shift calibration and reference-index file are
dataset-specific.
"""

import argparse
import os, sys, json, copy
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageFilter
import io

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="ADS ROC analysis on ImageNet-100 (benign shifts vs PE attacks, n=6 default). See module docstring for details.",
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
        help='Path to the ImageNet-100 ADS reference-index JSON. If not specified, defaults to '
             'ads_ref_indices_imagenet100.json alongside --output_path. If the file does not exist, '
             'it will be generated using a fixed seed.',
    )
    parser.add_argument(
        '--pe_types',
        type=str,
        nargs='+',
        default=None,
        choices=['learned', 'sinusoidal', 'rope', 'alibi'],
        help='Optional. PE types to evaluate. If omitted, uses the default ROC subset '
             'defined in the script: learned rope.',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        help='Optional. Random seeds to evaluate. If omitted, uses the n=6 TFS protocol '
             'defined in the script: 42 123 456 789 1011 1213.',
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

PE_TYPES        = ['learned', 'rope']
SEEDS           = [42, 123, 456, 789, 1011, 1213]

# Apply CLI overrides for PE_TYPES and SEEDS if user provided them
if args.pe_types is not None:
    PE_TYPES = args.pe_types
    print(f"[CLI override] PE_TYPES = {PE_TYPES}")
if args.seeds is not None:
    SEEDS = args.seeds
    print(f"[CLI override] SEEDS = {SEEDS}")

N_BOOTSTRAP     = 20   # Bootstrap iterations for clean noise floor
ATTACK_EPSILONS = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
TAU_MULTIPLIERS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]

PGD_STEPS       = 20
PGD_ALPHA_RATIO = 0.1

# ============================================================
# DATA
# ============================================================
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(DATA_DIR, base_transform)

N_REF_IMAGES = 256
if os.path.exists(REF_INDICES_PATH):
    with open(REF_INDICES_PATH) as f:
        ref_indices = json.load(f)
    print(f"Loaded reference indices from {REF_INDICES_PATH}")
else:
    torch.manual_seed(42)
    ref_indices = torch.randperm(len(val_dataset))[:N_REF_IMAGES].tolist()
    with open(REF_INDICES_PATH, 'w') as f:
        json.dump(ref_indices, f)
    print(f"Generated and saved {N_REF_IMAGES} ImageNet-100 reference indices to {REF_INDICES_PATH}")

# Full reference set loader (fixed 256 images)
ref_loader = DataLoader(
    Subset(val_dataset, ref_indices),
    batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

print(f"Reference set: {len(ref_indices)} images")

# ============================================================
# BENIGN TRANSFORMS (applied to reference images, not model)
# ============================================================
def apply_jpeg(tensor, quality):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = (tensor * std + mean).clamp(0, 1)
    pil = TF.to_pil_image(img)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    img2 = TF.to_tensor(Image.open(buf))
    return (img2 - mean) / std

def apply_blur(tensor, sigma):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = (tensor * std + mean).clamp(0, 1)
    pil = TF.to_pil_image(img)
    img2 = TF.to_tensor(pil.filter(ImageFilter.GaussianBlur(radius=sigma)))
    return (img2 - mean) / std

def apply_noise(tensor, sigma):
    return tensor + torch.randn_like(tensor) * sigma

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
# ATTENTION EXTRACTION
# ============================================================
@torch.no_grad()
def get_attn_l4(model, loader):
    """Get mean Layer 4 attention for given loader."""
    layers = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, attns = model.forward_with_attention(imgs)
        layers.append(attns[3].mean(0).cpu())
    return torch.stack(layers).mean(0)

@torch.no_grad()
def get_attn_l4_transformed(model, loader, transform_fn):
    """Get Layer 4 attention after applying transform to each image."""
    layers = []
    for imgs, _ in loader:
        transformed = torch.stack([transform_fn(img) for img in imgs])
        transformed = transformed.to(device)
        _, attns = model.forward_with_attention(transformed)
        layers.append(attns[3].mean(0).cpu())
    return torch.stack(layers).mean(0)

def kl_l4(P_attn, Q_attn):
    eps = 1e-10
    P = P_attn.float() + eps
    Q = Q_attn.float() + eps
    P = P / P.sum(-1, keepdim=True)
    Q = Q / Q.sum(-1, keepdim=True)
    return float((P * torch.log(P / Q)).sum(-1).mean().item())

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
# ROC
# ============================================================
def compute_roc(negative_scores, positive_scores, tau_multipliers, baseline):
    roc = []
    for mult in tau_multipliers:
        tau = mult * baseline
        tpr = float(np.mean([s > tau for s in positive_scores]))
        fpr = float(np.mean([s > tau for s in negative_scores]))
        roc.append({'tau_mult': mult, 'tau': float(tau), 'tpr': tpr, 'fpr': fpr})

    fprs = sorted(set([r['fpr'] for r in roc] + [0.0, 1.0]))
    tprs = []
    for fpr in fprs:
        # Find max TPR at or below this FPR
        matching = [r['tpr'] for r in roc if r['fpr'] <= fpr]
        tprs.append(max(matching) if matching else 0.0)

    auc = float(abs(np.trapz(tprs, fprs)))
    return roc, auc

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

            # Step 1: Reference attention (clean model, full ref set)
            ref_attn = get_attn_l4(model, ref_loader)
            print(f"    Reference attention computed ({len(ref_indices)} images)")

            # Step 2: Clean noise floor via bootstrap
            # Split ref_indices into two halves repeatedly
            # ADS(first_half_attn vs second_half_attn) using CLEAN model
            print(f"    Computing clean noise floor ({N_BOOTSTRAP} bootstrap splits)...")
            clean_scores = []
            all_ref_idx = np.array(ref_indices)
            for b in range(N_BOOTSTRAP):
                rng = np.random.RandomState(b * 31 + seed * 7)
                perm = rng.permutation(len(all_ref_idx))
                half_a = Subset(val_dataset, all_ref_idx[perm[:128]].tolist())
                half_b = Subset(val_dataset, all_ref_idx[perm[128:]].tolist())
                loader_a = DataLoader(half_a, batch_size=32, shuffle=False, num_workers=2)
                loader_b = DataLoader(half_b, batch_size=32, shuffle=False, num_workers=2)
                attn_a = get_attn_l4(model, loader_a)
                attn_b = get_attn_l4(model, loader_b)
                # ADS between two halves of clean data under clean model
                score = kl_l4(attn_a, attn_b)
                clean_scores.append(score)

            clean_mean = float(np.mean(clean_scores))
            clean_std  = float(np.std(clean_scores, ddof=1))
            clean_cv   = clean_std / clean_mean
            print(f"    Clean noise: mean={clean_mean:.6f}, std={clean_std:.6f}, CV={clean_cv*100:.1f}%")

            # Step 3: Benign distribution shifts
            # ADS(clean_model, ref_images) vs ADS(clean_model, transformed_ref_images)
            print(f"    Computing benign distribution shifts...")
            benign_configs = [
                ('jpeg_q50',  lambda t: apply_jpeg(t, 50)),
                ('jpeg_q30',  lambda t: apply_jpeg(t, 30)),
                ('jpeg_q10',  lambda t: apply_jpeg(t, 10)),
                ('blur_s1',   lambda t: apply_blur(t, 1)),
                ('blur_s2',   lambda t: apply_blur(t, 2)),
                ('blur_s3',   lambda t: apply_blur(t, 3)),
                ('noise_005', lambda t: apply_noise(t, 0.05)),
                ('noise_010', lambda t: apply_noise(t, 0.10)),
                ('noise_020', lambda t: apply_noise(t, 0.20)),
            ]

            benign_ads = {}
            all_negative_scores = list(clean_scores)  # start with clean bootstrap scores

            for name, tfn in benign_configs:
                try:
                    t_attn = get_attn_l4_transformed(model, ref_loader, tfn)
                    score = kl_l4(ref_attn, t_attn)
                    benign_ads[name] = float(score)
                    all_negative_scores.append(score)
                    print(f"      {name:12s}: {score:.6f} ({score/clean_mean:.1f}x baseline)")
                except Exception as e:
                    print(f"      {name:12s}: FAILED ({e})")
                    benign_ads[name] = None

            # Step 4: PE attack scores (true positives)
            # ADS(clean_model, ref) vs ADS(perturbed_model, ref) — same ref images
            print(f"    Computing PE attack scores...")
            attack_ads = {}
            for eps in ATTACK_EPSILONS:
                pm = perturb_pe(model, pe_type, eps)
                pm_attn = get_attn_l4(pm, ref_loader)
                score = kl_l4(ref_attn, pm_attn)
                attack_ads[str(eps)] = float(score)
                print(f"      ε={eps:.3f}: {score:.6f} ({score/clean_mean:.1f}x baseline)")
                del pm
                torch.cuda.empty_cache()

            # Step 5: ROC analysis
            print(f"    Computing ROC curves...")
            roc_by_eps = {}
            all_attack_scores = list(attack_ads.values())

            # ROC for each individual epsilon
            for eps in ATTACK_EPSILONS:
                pos_scores = [attack_ads[str(eps)]]
                roc, auc = compute_roc(all_negative_scores, pos_scores,
                                       TAU_MULTIPLIERS, clean_mean)
                roc_by_eps[str(eps)] = {'roc': roc, 'auc': float(auc)}

            # ROC for all epsilons combined (overall detector performance)
            roc_all, auc_all = compute_roc(all_negative_scores, all_attack_scores,
                                           TAU_MULTIPLIERS, clean_mean)

            all_results[pe_type][str(seed)] = {
                'clean_noise_floor': {
                    'scores': clean_scores,
                    'mean': clean_mean,
                    'std': clean_std,
                    'cv': clean_cv,
                },
                'benign_ads': benign_ads,
                'all_negative_scores': all_negative_scores,
                'attack_ads': attack_ads,
                'roc_by_eps': roc_by_eps,
                'roc_all': {'roc': roc_all, 'auc': float(auc_all)},
            }

            del model
            torch.cuda.empty_cache()

        with open(SAVE_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved to {SAVE_PATH}")

    return all_results

# ============================================================
# ANALYSIS
# ============================================================
def analyze(results):
    seeds_str = [str(s) for s in SEEDS]
    print("\n" + "=" * 75)
    print("CORRECTED ROC ANALYSIS SUMMARY")
    print("=" * 75)

    for pe_type in PE_TYPES:
        if pe_type not in results:
            continue

        available = [s for s in seeds_str if s in results[pe_type]]
        if not available:
            print(f"\n{pe_type.upper()}: no results")
            continue

        n = len(available)
        ddof = 1 if n > 1 else 0
        print(f"\n{pe_type.upper()} (n={n} seeds: {available}):")

        # Clean noise floor
        cvs = [results[pe_type][s]['clean_noise_floor']['cv'] for s in available]
        means = [results[pe_type][s]['clean_noise_floor']['mean'] for s in available]
        print(f"  Clean noise CV: {np.mean(cvs)*100:.1f}% ± {np.std(cvs, ddof=ddof)*100:.1f}%")
        print(f"  Clean noise mean: {np.mean(means):.6f} ± {np.std(means, ddof=ddof):.6f}")

        # Max benign ratio
        max_ratios = []
        for s in available:
            baseline = results[pe_type][s]['clean_noise_floor']['mean']
            vals = [v for v in results[pe_type][s]['benign_ads'].values()
                    if v is not None]
            max_ratios.append(max(vals) / baseline if vals else 0.0)
        print(f"  Max benign/baseline: {np.mean(max_ratios):.1f}x ± {np.std(max_ratios, ddof=ddof):.1f}x")

        # TPR/FPR table
        print(f"\n  TPR/FPR (all epsilons combined):")
        print(f"  {'tau':>8}  {'TPR':>6}  {'FPR':>6}")
        for mult in [5.0, 10.0, 20.0, 50.0]:
            tprs, fprs = [], []
            for s in available:
                for r in results[pe_type][s]['roc_all']['roc']:
                    if abs(r['tau_mult'] - mult) < 0.01:
                        tprs.append(r['tpr'])
                        fprs.append(r['fpr'])
            if tprs:
                print(f"  {mult:>7.0f}x  {np.mean(tprs):>6.3f}  {np.mean(fprs):>6.3f}")

        # AUC
        aucs = [results[pe_type][s]['roc_all']['auc'] for s in available]
        print(f"\n  Overall AUC: {np.mean(aucs):.3f} ± {np.std(aucs, ddof=ddof):.3f}")
        print(f"  Minimum seed AUC: {np.min(aucs):.3f}")

        # AUC by epsilon
        first_seed = available[0]
        print(f"\n  AUC by epsilon:")
        for eps in ['0.003', '0.005', '0.01', '0.05', '0.1', '0.2']:
            if eps in results[pe_type][first_seed]['roc_by_eps']:
                aucs_eps = [results[pe_type][s]['roc_by_eps'][eps]['auc']
                            for s in available
                            if eps in results[pe_type][s]['roc_by_eps']]
                if aucs_eps:
                    print(f"    ε={eps}: AUC={np.mean(aucs_eps):.3f} ± "
                          f"{np.std(aucs_eps, ddof=ddof):.3f}, "
                          f"min={np.min(aucs_eps):.3f}")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    print("ADS ROC Analysis v2 — ImageNet-100 n=6 corrected design")
    print(f"PE types: {PE_TYPES}, Seeds: {SEEDS}")
    print(f"N_BOOTSTRAP: {N_BOOTSTRAP} splits for clean noise floor")
    print()

    results = run()
    analyze(results)
    print("\n✅ Done!")
