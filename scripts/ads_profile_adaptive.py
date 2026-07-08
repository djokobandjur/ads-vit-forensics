#!/usr/bin/env python3
"""
ads_profile_adaptive.py — Profile-aware adaptive ADS attacker experiment
(ImageNet-100, n=6).

Purpose
-------
This script is a direct stress test for the revised ADS/TIFS narrative.  The
patched adaptive/ref-evasion scripts optimize

    CE(logits_ref, y_ref) - lambda * ADS_L4(delta; D_ref),

which is a white-box attack against a known Layer-4 sentinel.  This script
instead attacks a full-profile ADS objective:

    CE(logits_ref, y_ref) - lambda * S(ADS_L1, ..., ADS_L12),

where S is one of:

    max   : max_l ADS_l
    mean  : mean_l ADS_l
    lse   : smooth max, temperature-controlled log-sum-exp

The goal is not to replace the existing L4 adaptive/ref-evasion experiments,
but to answer the reviewer-critical question: if the forensic diagnostic is the
full ADS profile, what happens when the attacker regularizes the full profile
rather than only Layer 4?

Important inherited fixes from the patched July-2026 scripts
-----------------------------------------------------------
* PGD is projected ASCENT on CE - lambda * profile_ADS.
* lambda=0 is standard CE-ascent PGD.
* The objective uses the full fixed 256-image reference set by default.
* The model is kept in eval() during PGD; dropout is disabled.
* The clean attention p is detached; the perturbed attention q is kept on the
  autograd graph, so the ADS term has a real gradient.
* Use full_scale_experiment.py with VisionTransformer.forward_with_attention_grad.

Outputs
-------
A JSON file with per-seed/per-epsilon/per-lambda results containing:
  * accuracy/drop on full ImageNet-100 validation set
  * per-layer ADS and ratios on reference set
  * per-layer ADS and ratios on unseen holdout set
  * L4 evasion and full-profile max evasion indicators
  * output-collapse diagnostics on the reference set

Recommended smoke test
----------------------
python ads_profile_adaptive.py \
  --models_dir /content/drive/MyDrive/ads_tfs_n6/results_n6 \
  --val_dir /content/drive/MyDrive/imagenet100/val \
  --output_path /content/drive/MyDrive/ads_tfs_n6/data_n6/ads_profile_adaptive_smoke.json \
  --pe_types learned --seeds 42 --epsilons 0.2 --lambdas 2 \
  --profile_score lse --attack_ref_batches 1

Recommended targeted run after smoke test
-----------------------------------------
python ads_profile_adaptive.py \
  --models_dir /content/drive/MyDrive/ads_tfs_n6/results_n6 \
  --val_dir /content/drive/MyDrive/imagenet100/val \
  --output_path /content/drive/MyDrive/ads_tfs_n6/data_n6/ads_profile_adaptive_lse.json \
  --pe_types learned rope \
  --seeds 42 123 456 789 1011 1213 \
  --epsilons 0.1 0.2 0.5 \
  --lambdas 0 1 2 5 10 50 \
  --profile_score lse
"""

import argparse
import copy
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Colab compatibility: many notebooks copy full_scale_experiment.py to /content.
# Also allow colocating this script next to full_scale_experiment.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/content")
from full_scale_experiment import VisionTransformer  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

DEFAULT_PE_TYPES = ["learned", "rope"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 1213]
# Includes the ref-evasion grid and the main adaptive epsilon values.
DEFAULT_EPSILONS = [0.1, 0.2, 0.5]
DEFAULT_LAMBDAS = [0.0, 1.0, 2.0, 5.0, 10.0, 50.0]

PGD_STEPS = 20
PGD_ALPHA_RATIO = 0.1
N_REF_IMAGES = 256
N_HOLDOUT_IMAGES = 256

# These are the already used benign-calibrated L4 thresholds/floors.  They are
# intentionally not layer-specific; profile results should therefore be reported
# as a stress diagnostic unless layer-specific benign thresholds are later added.
DETECTION_THRESHOLD = {
    "learned": 3.0,
    "rope": 6.2,
}
NOISE_FLOOR = {
    "learned": 0.017010231611008446,
    "rope": 0.013948560304318862,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile-aware adaptive ADS attacker experiment (ImageNet-100, n=6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models_dir", required=True,
                   help="Checkpoint root, containing <pe>_seed<seed>/best_model.pth")
    p.add_argument("--val_dir", required=True,
                   help="ImageNet-100 validation directory in ImageFolder format")
    p.add_argument("--output_path", required=True,
                   help="Output JSON path")
    p.add_argument("--ref_indices_path", default=None,
                   help="Path to ads_ref_indices_imagenet100.json; defaults next to output")
    p.add_argument("--pe_types", nargs="+", default=None,
                   choices=["learned", "sinusoidal", "rope", "alibi"],
                   help="PE types to evaluate")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds to evaluate")
    p.add_argument("--epsilons", nargs="+", type=float, default=None,
                   help="Attack budgets")
    p.add_argument("--lambdas", nargs="+", type=float, default=None,
                   help="Profile-ADS regularization strengths")
    p.add_argument("--profile_score", choices=["max", "mean", "lse"], default="lse",
                   help="Full-profile ADS score optimized by the attacker")
    p.add_argument("--profile_layers", nargs="+", type=int, default=None,
                   help="0-indexed layers included in profile score; default all 12 layers")
    p.add_argument("--lse_temperature", type=float, default=0.01,
                   help="Temperature for smooth max when --profile_score lse")
    p.add_argument("--attack_ref_batches", type=int, default=None,
                   help=("Optional smoke/debug mode. If set, uses only the first N "
                         "reference batches for the attack objective. Default None "
                         "uses all 256 reference images."))
    p.add_argument("--ref_batch_size", type=int, default=32,
                   help="Reference/holdout batch size for attention measurement and attack objective")
    p.add_argument("--val_batch_size", type=int, default=64,
                   help="Validation batch size for accuracy evaluation")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers")
    p.add_argument("--pgd_steps", type=int, default=PGD_STEPS,
                   help="PGD steps")
    p.add_argument("--alpha_ratio", type=float, default=PGD_ALPHA_RATIO,
                   help="PGD step size as epsilon multiplier")
    p.add_argument("--objective_scope", choices=["full", "batchmean"], default="full",
                   help=("full = exact full-reference ADS objective by accumulating "
                         "differentiable attention sums over the reference set. "
                         "batchmean = memory fallback that averages per-batch profile "
                         "scores; use only as a clearly labelled ablation/smoke mode."))
    p.add_argument("--save_each", action="store_true",
                   help="Write JSON after every evaluated lambda to protect long runs")
    return p.parse_args()


ARGS = parse_args()
RESULTS_DIR = ARGS.models_dir
DATA_DIR = ARGS.val_dir
SAVE_PATH = ARGS.output_path
REF_INDICES_PATH = (
    ARGS.ref_indices_path
    if ARGS.ref_indices_path
    else os.path.join(os.path.dirname(SAVE_PATH), "ads_ref_indices_imagenet100.json")
)
os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)

PE_TYPES = ARGS.pe_types if ARGS.pe_types is not None else DEFAULT_PE_TYPES
SEEDS = ARGS.seeds if ARGS.seeds is not None else DEFAULT_SEEDS
EPSILONS = ARGS.epsilons if ARGS.epsilons is not None else DEFAULT_EPSILONS
LAMBDAS = ARGS.lambdas if ARGS.lambdas is not None else DEFAULT_LAMBDAS
PROFILE_LAYERS = ARGS.profile_layers if ARGS.profile_layers is not None else list(range(12))
PROFILE_LAYERS = sorted(set(PROFILE_LAYERS))

unsupported = [pe for pe in PE_TYPES if pe not in DETECTION_THRESHOLD or pe not in NOISE_FLOOR]
if unsupported:
    raise ValueError(
        f"Thresholds/noise floors are defined only for {sorted(DETECTION_THRESHOLD)}; "
        f"unsupported PE types: {unsupported}"
    )
if any(L < 0 or L > 11 for L in PROFILE_LAYERS):
    raise ValueError(f"profile_layers must be 0..11, got {PROFILE_LAYERS}")
if ARGS.lse_temperature <= 0:
    raise ValueError("--lse_temperature must be positive")

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_dataset = datasets.ImageFolder(DATA_DIR, VAL_TRANSFORM)

if os.path.exists(REF_INDICES_PATH):
    with open(REF_INDICES_PATH) as f:
        ref_indices = json.load(f)
    print(f"Loaded {len(ref_indices)} reference indices from {REF_INDICES_PATH}")
else:
    torch.manual_seed(42)
    ref_indices = torch.randperm(len(val_dataset))[:N_REF_IMAGES].tolist()
    with open(REF_INDICES_PATH, "w") as f:
        json.dump(ref_indices, f)
    print(f"Generated and saved {N_REF_IMAGES} reference indices to {REF_INDICES_PATH}")

if len(ref_indices) != N_REF_IMAGES:
    print(f"WARNING: reference index file has {len(ref_indices)} indices, expected {N_REF_IMAGES}")
if max(ref_indices) >= len(val_dataset):
    raise ValueError(
        f"Reference index file {REF_INDICES_PATH} is incompatible with this dataset: "
        f"max index {max(ref_indices)} >= dataset size {len(val_dataset)}"
    )

ref_set = set(ref_indices)
non_ref = [i for i in range(len(val_dataset)) if i not in ref_set]
torch.manual_seed(999)
holdout_perm = torch.randperm(len(non_ref))[:N_HOLDOUT_IMAGES].tolist()
holdout_indices = [non_ref[i] for i in holdout_perm]

ref_loader = DataLoader(
    Subset(val_dataset, ref_indices), batch_size=ARGS.ref_batch_size,
    shuffle=False, num_workers=ARGS.num_workers, pin_memory=True,
)
holdout_loader = DataLoader(
    Subset(val_dataset, holdout_indices), batch_size=ARGS.ref_batch_size,
    shuffle=False, num_workers=ARGS.num_workers, pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=ARGS.val_batch_size,
    shuffle=False, num_workers=ARGS.num_workers, pin_memory=True,
)

print(f"Reference set: {len(ref_indices)} images (known to attacker)")
print(f"Holdout set: {len(holdout_indices)} images (unseen by attacker)")
if ARGS.attack_ref_batches is None:
    print("Attack objective: full 256-image reference set")
else:
    print(f"Attack objective: first {ARGS.attack_ref_batches} reference batch(es) only [debug/smoke]")
print(f"Profile score: {ARGS.profile_score}; layers: {[L + 1 for L in PROFILE_LAYERS]}; scope: {ARGS.objective_scope}")

# Sum-reduction CE so averaging is exact for any batch sizes.
_ce_sum_reduction = nn.CrossEntropyLoss(reduction="sum")


# -----------------------------------------------------------------------------
# Model and measurements
# -----------------------------------------------------------------------------
def load_model(pe_type: str, seed: int) -> VisionTransformer:
    model = VisionTransformer(
        img_size=224, patch_size=16, num_classes=100, embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1, pe_type=pe_type,
    )
    ckpt = os.path.join(RESULTS_DIR, f"{pe_type}_seed{seed}", "best_model.pth")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def get_attn_layers(model: nn.Module, loader: DataLoader) -> List[torch.Tensor]:
    """Image-averaged attention tensor for all layers, each shape (H, N, N)."""
    totals = None
    total_images = 0
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        _, attns = model.forward_with_attention(imgs)
        batch_sums = [a.sum(dim=0).detach().cpu() for a in attns]
        if totals is None:
            totals = batch_sums
        else:
            totals = [t + s for t, s in zip(totals, batch_sums)]
        total_images += imgs.size(0)
    if total_images == 0:
        raise RuntimeError("Empty loader passed to get_attn_layers")
    return [t / total_images for t in totals]


def compute_ads_scalar(clean_attn: torch.Tensor, perturbed_attn: torch.Tensor) -> float:
    eps = 1e-10
    p = clean_attn.float() + eps
    q = perturbed_attn.float() + eps
    p = p / p.sum(-1, keepdim=True)
    q = q / q.sum(-1, keepdim=True)
    return float((p * torch.log(p / q)).sum(-1).mean().item())


def compute_per_layer_ads(clean_layers: Sequence[torch.Tensor], pert_layers: Sequence[torch.Tensor]) -> List[float]:
    return [compute_ads_scalar(c, p) for c, p in zip(clean_layers, pert_layers)]


@torch.no_grad()
def measure_accuracy(model: nn.Module, loader: DataLoader) -> float:
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        _, pred = model(imgs).max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def output_collapse_stats(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    """Entropy in nats of mean softmax over the reference set, and top-1 share."""
    prob_sum = None
    total_images = 0
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        probs = torch.softmax(model(imgs), dim=1).sum(dim=0).detach().cpu()
        prob_sum = probs if prob_sum is None else prob_sum + probs
        total_images += imgs.size(0)
    mean_p = (prob_sum / total_images).clamp_min(1e-12)
    mean_p = mean_p / mean_p.sum()
    entropy = float(-(mean_p * mean_p.log()).sum().item())
    top1_share = float(mean_p.max().item())
    return entropy, top1_share


def get_pe_params(model: nn.Module, pe_type: str) -> List[nn.Parameter]:
    """Convert PE buffers to trainable parameters and return the PE attack surface."""
    pe_params: List[nn.Parameter] = []

    for name, buf in list(model.named_buffers()):
        clean = name.replace("_orig_mod.", "")
        if any(k in clean for k in [".pe", "cos_cached", "sin_cached"]):
            if "rel_dist" in clean:
                continue
            parts = clean.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            delattr(obj, parts[-1])
            new_p = nn.Parameter(buf.clone().to(DEVICE), requires_grad=True)
            setattr(obj, parts[-1], new_p)
            pe_params.append(new_p)

    for _, mod in model.named_modules():
        if type(mod).__name__ == "ALiBi":
            slopes = mod.slopes.clone().to(DEVICE)
            del mod.slopes
            mod.slopes = nn.Parameter(slopes, requires_grad=True)
            pe_params.append(mod.slopes)

    for name, param in model.named_parameters():
        clean = name.replace("_orig_mod.", "")
        if any(k in clean for k in ["pos_embed", "inv_freq"]):
            if not param.requires_grad:
                param.requires_grad_(True)
            if not any(param is p for p in pe_params):
                pe_params.append(param)

    return pe_params


def iter_attack_ref_batches() -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for i, (imgs, labels) in enumerate(ref_loader):
        if ARGS.attack_ref_batches is not None and i >= ARGS.attack_ref_batches:
            break
        yield imgs.to(DEVICE), labels.to(DEVICE)


def profile_score_from_ads(ads_values: Sequence[torch.Tensor], score: str) -> torch.Tensor:
    vals = torch.stack(list(ads_values))
    if score == "max":
        return vals.max()
    if score == "mean":
        return vals.mean()
    if score == "lse":
        t = ARGS.lse_temperature
        return t * torch.logsumexp(vals / t, dim=0)
    raise ValueError(score)


def differentiable_ads(clean_attn: torch.Tensor, attn_sum: torch.Tensor, total_images: int) -> torch.Tensor:
    """KL(clean || perturbed) for one layer, keeping perturbed q differentiable."""
    eps = 1e-10
    q = (attn_sum / total_images).float() + eps
    q = q / q.sum(-1, keepdim=True)
    p = clean_attn.detach().to(q.device).float() + eps
    p = p / p.sum(-1, keepdim=True)
    return (p * torch.log(p / q)).sum(-1).mean()


def compute_full_profile_objective(
    model: nn.Module,
    clean_ref_layers: Sequence[torch.Tensor],
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Return CE, profile ADS score, objective, and ADS per selected layer."""
    if ARGS.objective_scope == "batchmean":
        return compute_batchmean_profile_objective(model, clean_ref_layers, lam)

    ce_sum = None
    attn_sums: Dict[int, torch.Tensor] = {}
    total_images = 0

    for imgs, labels in iter_attack_ref_batches():
        if lam > 0:
            # Critical: forward_with_attention_grad keeps q on the autograd graph.
            logits, attns = model.forward_with_attention_grad(imgs, layers=tuple(PROFILE_LAYERS))
            for L in PROFILE_LAYERS:
                batch_attn_sum = attns[L].sum(dim=0).float()
                attn_sums[L] = batch_attn_sum if L not in attn_sums else attn_sums[L] + batch_attn_sum
        else:
            logits = model(imgs)

        ce = _ce_sum_reduction(logits, labels)
        ce_sum = ce if ce_sum is None else ce_sum + ce
        total_images += imgs.size(0)

    if total_images == 0:
        raise RuntimeError("No reference images used for attack objective")

    ce_loss = ce_sum / total_images

    if lam > 0:
        ads_per_layer = [differentiable_ads(clean_ref_layers[L], attn_sums[L], total_images)
                         for L in PROFILE_LAYERS]
        profile_ads = profile_score_from_ads(ads_per_layer, ARGS.profile_score)
    else:
        ads_per_layer = []
        profile_ads = torch.zeros((), device=ce_loss.device, dtype=ce_loss.dtype)

    objective = ce_loss - lam * profile_ads
    return ce_loss, profile_ads, objective, ads_per_layer


def compute_batchmean_profile_objective(
    model: nn.Module,
    clean_ref_layers: Sequence[torch.Tensor],
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Memory fallback: average CE and per-batch profile ADS scores.

    This is not identical to the exact full-reference mean-attention objective.
    Use only for smoke tests or clearly labelled ablations.
    """
    ce_sum = None
    profile_sum = None
    ads_layer_sums = [None for _ in PROFILE_LAYERS]
    total_images = 0

    for imgs, labels in iter_attack_ref_batches():
        if lam > 0:
            logits, attns = model.forward_with_attention_grad(imgs, layers=tuple(PROFILE_LAYERS))
            batch_ads = []
            for j, L in enumerate(PROFILE_LAYERS):
                # Batch-mean perturbed attention.  Clean p remains the full-reference clean mean.
                ads_j = differentiable_ads(clean_ref_layers[L], attns[L].sum(dim=0).float(), imgs.size(0))
                batch_ads.append(ads_j)
                ads_layer_sums[j] = ads_j if ads_layer_sums[j] is None else ads_layer_sums[j] + ads_j
            batch_profile = profile_score_from_ads(batch_ads, ARGS.profile_score)
            profile_sum = batch_profile if profile_sum is None else profile_sum + batch_profile
        else:
            logits = model(imgs)
        ce = _ce_sum_reduction(logits, labels)
        ce_sum = ce if ce_sum is None else ce_sum + ce
        total_images += imgs.size(0)

    if total_images == 0:
        raise RuntimeError("No reference images used for attack objective")
    ce_loss = ce_sum / total_images
    if lam > 0:
        # Number of processed batches, not images.  Batch sizes are normally equal here.
        n_batches = sum(1 for _ in range(math.ceil(total_images / ARGS.ref_batch_size)))
        ads_per_layer = [x / n_batches for x in ads_layer_sums]
        profile_ads = profile_sum / n_batches
    else:
        ads_per_layer = []
        profile_ads = torch.zeros((), device=ce_loss.device, dtype=ce_loss.dtype)
    objective = ce_loss - lam * profile_ads
    return ce_loss, profile_ads, objective, ads_per_layer


# -----------------------------------------------------------------------------
# Profile-aware PGD
# -----------------------------------------------------------------------------
def profile_adaptive_pgd(
    model: nn.Module,
    clean_ref_layers: Sequence[torch.Tensor],
    pe_type: str,
    epsilon: float,
    lam: float,
) -> nn.Module:
    """Projected ascent on CE - lambda*profile_ADS over the PE state."""
    pm = copy.deepcopy(model).to(DEVICE)
    alpha = epsilon * ARGS.alpha_ratio

    pe_params = get_pe_params(pm, pe_type)
    if not pe_params:
        print(f"WARNING: no PE params for {pe_type}")
        return pm

    base_params = [p.detach().clone() for p in pe_params]
    deltas = [torch.zeros_like(p) for p in pe_params]

    # Critical inherited fix: attack in eval() mode, not train().
    pm.eval()
    for step in range(ARGS.pgd_steps):
        with torch.no_grad():
            for p, base, d in zip(pe_params, base_params, deltas):
                p.copy_(base + d)

        for p in pe_params:
            if p.grad is not None:
                p.grad.zero_()

        ce_loss, profile_ads, objective, _ = compute_full_profile_objective(pm, clean_ref_layers, lam)
        objective.backward()

        with torch.no_grad():
            for j, p in enumerate(pe_params):
                if p.grad is not None:
                    deltas[j] = (deltas[j] + alpha * p.grad.sign()).clamp(-epsilon, epsilon)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    with torch.no_grad():
        for p, base, d in zip(pe_params, base_params, deltas):
            p.copy_(base + d)
    pm.eval()
    return pm


def summarize_profile(per_layer_ads: Sequence[float], nf: float, tau: float) -> Dict[str, object]:
    ratios = [float(x / nf) for x in per_layer_ads]
    max_ratio = max(ratios)
    max_layer = int(max(range(len(ratios)), key=lambda i: ratios[i])) + 1
    mean_ratio = float(sum(ratios) / len(ratios))
    l4_ratio = ratios[3]
    caught_layers = [i + 1 for i, r in enumerate(ratios) if r >= tau]
    return {
        "per_layer_ads": [float(x) for x in per_layer_ads],
        "per_layer_ratios": ratios,
        "ads_l4": float(per_layer_ads[3]),
        "ratio_l4": float(l4_ratio),
        "max_ratio": float(max_ratio),
        "max_layer": max_layer,
        "mean_ratio": mean_ratio,
        "evades_l4": bool(l4_ratio < tau),
        "evades_profile_max": bool(max_ratio < tau),
        "caught_layers": caught_layers,
    }


def maybe_save(results: dict) -> None:
    if ARGS.save_each:
        with open(SAVE_PATH, "w") as f:
            json.dump(results, f, indent=2)


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------
def run() -> dict:
    results = {
        "metadata": {
            "script": "ads_profile_adaptive.py",
            "objective": "full_profile_ce_minus_lambda_profile_ads",
            "profile_score": ARGS.profile_score,
            "profile_layers_0_indexed": PROFILE_LAYERS,
            "profile_layers_1_indexed": [L + 1 for L in PROFILE_LAYERS],
            "lse_temperature": ARGS.lse_temperature,
            "objective_scope": ARGS.objective_scope,
            "attack_ref_images": N_REF_IMAGES if ARGS.attack_ref_batches is None else min(
                N_REF_IMAGES, ARGS.attack_ref_batches * ARGS.ref_batch_size),
            "full_reference_default": ARGS.attack_ref_batches is None,
            "attack_mode": "eval",
            "pgd_steps": ARGS.pgd_steps,
            "alpha_ratio": ARGS.alpha_ratio,
            "noise_floors": NOISE_FLOOR,
            "detection_thresholds": DETECTION_THRESHOLD,
            "note": (
                "Profile evasion uses the same PE-specific L4 noise floor and threshold "
                "for reporting all-layer ratios. Treat as a stress diagnostic unless "
                "layer-specific benign thresholds are added."
            ),
        },
        "results": {},
    }

    for pe_type in PE_TYPES:
        nf = NOISE_FLOOR[pe_type]
        tau = DETECTION_THRESHOLD[pe_type]
        results["results"][pe_type] = {}

        print("\n" + "=" * 72)
        print(f"PE TYPE: {pe_type.upper()} | tau={tau}x | nf={nf:.6f}")
        print("=" * 72)

        for seed in SEEDS:
            print(f"\n Seed {seed}:")
            model = load_model(pe_type, seed)
            clean_ref_layers = get_attn_layers(model, ref_loader)
            clean_hold_layers = get_attn_layers(model, holdout_loader)
            clean_acc = measure_accuracy(model, val_loader)
            print(f" Clean accuracy: {clean_acc:.2f}%")

            results["results"][pe_type][str(seed)] = {"clean_acc": clean_acc}
            for eps in EPSILONS:
                results["results"][pe_type][str(seed)][str(eps)] = {}
                print(f"\n ε={eps}:")
                print("      λ   drop   L4(ref)  max(ref) Lmax  L4(hold) max(hold)  profile-evade(ref/hold)")

                for lam in LAMBDAS:
                    pm = profile_adaptive_pgd(model, clean_ref_layers, pe_type, eps, lam)
                    acc = measure_accuracy(pm, val_loader)
                    drop = clean_acc - acc

                    pert_ref_layers = get_attn_layers(pm, ref_loader)
                    pert_hold_layers = get_attn_layers(pm, holdout_loader)
                    ref_ads = compute_per_layer_ads(clean_ref_layers, pert_ref_layers)
                    hold_ads = compute_per_layer_ads(clean_hold_layers, pert_hold_layers)
                    ref_summary = summarize_profile(ref_ads, nf, tau)
                    hold_summary = summarize_profile(hold_ads, nf, tau)
                    entropy, top1_share = output_collapse_stats(pm, ref_loader)

                    row = {
                        "lambda": float(lam),
                        "epsilon": float(eps),
                        "accuracy": float(acc),
                        "acc_drop": float(drop),
                        "profile_score": ARGS.profile_score,
                        "profile_layers": PROFILE_LAYERS,
                        "ref": ref_summary,
                        "holdout": hold_summary,
                        "evades_l4_ref_and_hold": bool(ref_summary["evades_l4"] and hold_summary["evades_l4"]),
                        "evades_profile_ref_and_hold": bool(
                            ref_summary["evades_profile_max"] and hold_summary["evades_profile_max"]),
                        "out_entropy_ref": float(entropy),
                        "top1_share_ref": float(top1_share),
                        "attack_objective": "full_reference_ce_minus_lambda_profile_ads",
                        "attack_ref_images": results["metadata"]["attack_ref_images"],
                        "attack_mode": "eval",
                    }
                    results["results"][pe_type][str(seed)][str(eps)][str(lam)] = row

                    print(
                        f" {lam:6.1f} {drop:6.2f}pp "
                        f"{ref_summary['ratio_l4']:7.2f}x {ref_summary['max_ratio']:8.2f}x "
                        f"L{ref_summary['max_layer']:<2d} "
                        f"{hold_summary['ratio_l4']:8.2f}x {hold_summary['max_ratio']:8.2f}x  "
                        f"{'YES' if ref_summary['evades_profile_max'] else 'NO '}/"
                        f"{'YES' if hold_summary['evades_profile_max'] else 'NO '}  "
                        f"ent={entropy:.2f} top1={top1_share:.2f}"
                    )
                    maybe_save(results)

    return results


if __name__ == "__main__":
    final_results = run()
    with open(SAVE_PATH, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved profile-aware adaptive results to {SAVE_PATH}")
