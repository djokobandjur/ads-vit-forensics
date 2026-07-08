#!/usr/bin/env python3
"""
ads_ref_evasion.py — Reference-set ADS evasion experiment (ImageNet-100, n=6).

This patched version fixes the evasion objective and uses the full 256-image
ADS reference set for the attack objective by default.

Key conventions
---------------
* PGD is implemented as projected *ascent* on

      J_evasion = CE(logits_ref, y_ref) - lambda * ADS_L4(delta; D_ref)

  so lambda=0 is standard CE-ascent PGD, and lambda>0 trades accuracy damage
  against ADS minimization on the known monitored reference set.
* The CE term and differentiable ADS term are computed over the full fixed
  256-image reference set by default, not only over the first 32-image batch.
* Accuracy is evaluated on the full ImageNet-100 validation set.
* ADS is reported both on the known reference set and on an unseen 256-image
  holdout set.

Output: ads_ref_evasion.json
"""

import argparse
import copy
import json
import os
import sys
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Colab compatibility: many notebooks copy full_scale_experiment.py to /content.
sys.path.insert(0, "/content")
from full_scale_experiment import VisionTransformer  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

DEFAULT_PE_TYPES = ["learned", "rope"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 1213]
DEFAULT_EPSILONS = [0.1, 0.2, 0.5]
DEFAULT_LAMBDAS = [0.0, 1.0, 5.0, 10.0, 50.0]

PGD_STEPS = 20
PGD_ALPHA_RATIO = 0.1
N_REF_IMAGES = 256
N_HOLDOUT_IMAGES = 256

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
        description="ADS reference-set evasion experiment (ImageNet-100, n=6).",
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
                   help="Evasion regularization strengths")
    p.add_argument("--attack_ref_batches", type=int, default=None,
                   help=("Optional smoke/debug mode. If set, uses only the first N "
                         "reference batches for the attack objective. Default None "
                         "uses all 256 reference images."))
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

unsupported = [pe for pe in PE_TYPES if pe not in DETECTION_THRESHOLD or pe not in NOISE_FLOOR]
if unsupported:
    raise ValueError(
        f"Thresholds/noise floors are defined only for {sorted(DETECTION_THRESHOLD)}; "
        f"unsupported PE types: {unsupported}"
    )

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

ref_loader = DataLoader(Subset(val_dataset, ref_indices), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
holdout_loader = DataLoader(Subset(val_dataset, holdout_indices), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

print(f"Reference set: {len(ref_indices)} images (known to attacker)")
print(f"Holdout set: {len(holdout_indices)} images (unseen by attacker)")
if ARGS.attack_ref_batches is None:
    print("Attack objective: full 256-image reference set")
else:
    print(f"Attack objective: first {ARGS.attack_ref_batches} reference batch(es) only [debug/smoke]")

# -----------------------------------------------------------------------------
# Model and metrics
# -----------------------------------------------------------------------------

def load_model(pe_type: str, seed: int) -> VisionTransformer:
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=100,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        pe_type=pe_type,
    )
    ckpt = os.path.join(RESULTS_DIR, f"{pe_type}_seed{seed}", "best_model.pth")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def get_attn_l4(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    total_attn = None
    total_images = 0
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        _, attns = model.forward_with_attention(imgs)
        batch_sum = attns[3].sum(dim=0).detach().cpu()
        total_attn = batch_sum if total_attn is None else total_attn + batch_sum
        total_images += imgs.size(0)
    if total_images == 0:
        raise RuntimeError("Empty loader passed to get_attn_l4")
    return total_attn / total_images


def compute_ads(clean_attn: torch.Tensor, perturbed_attn: torch.Tensor) -> float:
    eps = 1e-10
    p = clean_attn.float() + eps
    q = perturbed_attn.float() + eps
    p = p / p.sum(-1, keepdim=True)
    q = q / q.sum(-1, keepdim=True)
    return float((p * torch.log(p / q)).sum(-1).mean().item())


@torch.no_grad()
def measure_accuracy(model: nn.Module, loader: DataLoader) -> float:
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        _, pred = model(imgs).max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def get_pe_params(model: nn.Module, pe_type: str) -> List[nn.Parameter]:
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


# Sum-reduction CE so that dividing by total_images gives the exact mean
# regardless of how images split into batches (guards the last partial batch).
_ce_sum_reduction = nn.CrossEntropyLoss(reduction="sum")


def compute_full_ref_objective(
    model: nn.Module,
    clean_ref_attn: torch.Tensor,
    criterion: nn.Module,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CE, ADS, and ascent objective CE - lambda*ADS on the reference set."""
    ce_sum = None
    attn_sum = None
    total_images = 0

    for imgs, labels in iter_attack_ref_batches():
        if lam > 0:
            # forward_with_attention_grad keeps L4 attention on the autograd graph
            # so the ADS term backpropagates into the PE delta. The old
            # forward_with_attention detaches attention -> zero ADS gradient.
            logits, attns = model.forward_with_attention_grad(imgs, layers=(3,))
            batch_attn_sum = attns[3].sum(dim=0).float()
            attn_sum = batch_attn_sum if attn_sum is None else attn_sum + batch_attn_sum
        else:
            logits = model(imgs)

        # reduction='sum' so multi-batch averaging is exact for any batch
        # sizes (previous mean*batch_size only matched when all batches were equal).
        ce = _ce_sum_reduction(logits, labels)
        ce_sum = ce if ce_sum is None else ce_sum + ce
        total_images += imgs.size(0)

    if total_images == 0:
        raise RuntimeError("No reference images used for attack objective")

    ce_loss = ce_sum / total_images

    if lam > 0:
        eps_kl = 1e-10
        obj_device = ce_loss.device
        # Keep q differentiable: detaching it would remove the ADS gradient
        # and make all lambda values reduce to the lambda=0 CE attack.
        q = (attn_sum / total_images).to(obj_device).float() + eps_kl
        q = q / q.sum(-1, keepdim=True)
        p = clean_ref_attn.detach().to(obj_device).float() + eps_kl
        p = p / p.sum(-1, keepdim=True)
        ads_loss = (p * torch.log(p / q)).sum(-1).mean()
    else:
        ads_loss = torch.zeros((), device=DEVICE, dtype=ce_loss.dtype)

    objective = ce_loss - lam * ads_loss
    return ce_loss, ads_loss, objective


# -----------------------------------------------------------------------------
# Evasion PGD
# -----------------------------------------------------------------------------

def evasion_pgd(model: nn.Module, clean_ref_attn: torch.Tensor, pe_type: str, epsilon: float, lam: float) -> nn.Module:
    """Projected ascent on CE - lambda*ADS over the PE state."""
    pm = copy.deepcopy(model).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    alpha = epsilon * PGD_ALPHA_RATIO

    pe_params = get_pe_params(pm, pe_type)
    if not pe_params:
        print(f"WARNING: no PE params for {pe_type}")
        return pm

    base_params = [p.detach().clone() for p in pe_params]
    deltas = [torch.zeros_like(p) for p in pe_params]

    # Attack in eval() mode: dropout during PGD is stochastic and was shown to
    # create spurious high-damage/low-ADS "evasions" (seeds 789/1213) that vanish
    # once dropout is disabled. eval() gives a deterministic, stronger attack.
    pm.eval()
    for _ in range(PGD_STEPS):
        with torch.no_grad():
            for p, base, d in zip(pe_params, base_params, deltas):
                p.copy_(base + d)

        for p in pe_params:
            if p.grad is not None:
                p.grad.zero_()

        _, _, objective = compute_full_ref_objective(pm, clean_ref_attn, criterion, lam)
        objective.backward()

        with torch.no_grad():
            for j, p in enumerate(pe_params):
                if p.grad is not None:
                    deltas[j] = (deltas[j] + alpha * p.grad.sign()).clamp(-epsilon, epsilon)

    with torch.no_grad():
        for p, base, d in zip(pe_params, base_params, deltas):
            p.copy_(base + d)
    pm.eval()
    return pm


# -----------------------------------------------------------------------------
# Main run and analysis
# -----------------------------------------------------------------------------

def run() -> dict:
    results = {}
    for pe_type in PE_TYPES:
        results[pe_type] = {}
        nf = NOISE_FLOOR[pe_type]
        tau = DETECTION_THRESHOLD[pe_type]

        print("\n" + "=" * 60)
        print(f"PE TYPE: {pe_type.upper()} | detection threshold: {tau}x")
        print("=" * 60)

        for seed in SEEDS:
            print(f"\n Seed {seed}:")
            model = load_model(pe_type, seed)
            clean_ref_attn = get_attn_l4(model, ref_loader)
            clean_hold_attn = get_attn_l4(model, holdout_loader)
            clean_acc = measure_accuracy(model, val_loader)
            print(f" Clean accuracy: {clean_acc:.2f}%")

            results[pe_type][str(seed)] = {}
            for eps in EPSILONS:
                results[pe_type][str(seed)][str(eps)] = {}
                print(f"\n ε={eps}:")
                print(f" {'λ':>6} {'Acc drop':>10} {'ADS(ref)':>10} {'ADS(hold)':>10} {'Evasion?':>10} {'Generalized?':>13}")

                for lam in LAMBDAS:
                    pm = evasion_pgd(model, clean_ref_attn, pe_type, eps, lam)

                    pm_ref_attn = get_attn_l4(pm, ref_loader)
                    ads_ref = compute_ads(clean_ref_attn, pm_ref_attn)

                    pm_hold_attn = get_attn_l4(pm, holdout_loader)
                    ads_hold = compute_ads(clean_hold_attn, pm_hold_attn)

                    acc = measure_accuracy(pm, val_loader)
                    acc_drop = clean_acc - acc

                    ref_ratio = ads_ref / nf
                    hold_ratio = ads_hold / nf
                    evades_ref = ref_ratio < tau
                    evades_hold = hold_ratio < tau

                    results[pe_type][str(seed)][str(eps)][str(lam)] = {
                        "lambda": float(lam),
                        "acc_drop": float(acc_drop),
                        "accuracy": float(acc),
                        "ads_ref": float(ads_ref),
                        "ads_hold": float(ads_hold),
                        "ads_ref_ratio": float(ref_ratio),
                        "ads_hold_ratio": float(hold_ratio),
                        "evades_ref": bool(evades_ref),
                        "evades_hold": bool(evades_hold),
                        "attack_objective": "full_reference_ce_minus_lambda_ads",
                        "attack_ref_images": int(len(ref_indices) if ARGS.attack_ref_batches is None else min(len(ref_indices), 32 * ARGS.attack_ref_batches)),
                    }

                    print(
                        f" {lam:>6.1f} {acc_drop:>10.2f}pp "
                        f"{ref_ratio:>8.2f}x {hold_ratio:>9.2f}x "
                        f"{'YES' if evades_ref else 'NO':>10} "
                        f"{'YES' if evades_hold else 'NO':>13}"
                    )
                    del pm
                    torch.cuda.empty_cache()

            del model
            torch.cuda.empty_cache()

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {SAVE_PATH}")
    return results


def analyze(results: dict) -> None:
    print("\n" + "=" * 75)
    print("REFERENCE SET EVASION ANALYSIS")
    print("=" * 75)
    print("Question: if attacker evades the reference set, does evasion transfer to holdout?")

    for pe_type in PE_TYPES:
        if pe_type not in results:
            continue
        available = [str(s) for s in SEEDS if str(s) in results[pe_type]]
        print(f"\n{pe_type.upper()} (n={len(available)} seeds, τ={DETECTION_THRESHOLD[pe_type]}x):")
        for eps in EPSILONS:
            eps_key = str(eps)
            print(f"\n ε={eps}:")
            print(f" {'λ':>7} {'drop mean':>10} {'drop sd':>9} {'ref ratio':>10} {'hold ratio':>11} {'ref':>8} {'hold':>8} {'both':>8}")
            print(" " + "-" * 83)
            for lam in LAMBDAS:
                lam_key = str(lam)
                drops, ref_ratios, hold_ratios, ev_ref, ev_hold = [], [], [], [], []
                for s in available:
                    if eps_key in results[pe_type][s] and lam_key in results[pe_type][s][eps_key]:
                        r = results[pe_type][s][eps_key][lam_key]
                        drops.append(r["acc_drop"])
                        ref_ratios.append(r["ads_ref_ratio"])
                        hold_ratios.append(r["ads_hold_ratio"])
                        ev_ref.append(r["evades_ref"])
                        ev_hold.append(r["evades_hold"])
                if not drops:
                    continue
                both = [a and b for a, b in zip(ev_ref, ev_hold)]
                print(
                    f" {lam:7.1f} {np.mean(drops):10.2f} {np.std(drops, ddof=1) if len(drops)>1 else 0:9.2f} "
                    f"{np.mean(ref_ratios):10.2f} {np.mean(hold_ratios):11.2f} "
                    f"{sum(ev_ref)}/{len(ev_ref):<4} {sum(ev_hold)}/{len(ev_hold):<4} {sum(both)}/{len(both):<4}"
                )


if __name__ == "__main__":
    print("Reference Set Evasion Experiment — patched full-reference CE - lambda*ADS")
    print(f"PE types: {PE_TYPES}, Seeds: {SEEDS}")
    print(f"Epsilons: {EPSILONS}, Lambdas: {LAMBDAS}")
    print(f"Reference: {N_REF_IMAGES} known images, Holdout: {N_HOLDOUT_IMAGES} unseen images")
    print(f"PGD steps: {PGD_STEPS}, alpha ratio: {PGD_ALPHA_RATIO}")
    print(f"Noise floors: {NOISE_FLOOR}")
    print(f"Detection thresholds: {DETECTION_THRESHOLD}")
    print()
    res = run()
    analyze(res)
    print("\nDone.")
