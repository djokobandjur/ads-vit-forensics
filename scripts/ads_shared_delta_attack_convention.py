#!/usr/bin/env python3
"""
ads_shared_delta_attack_convention.py
=====================================

Generate the missing shared-delta / tied-buffer PGD-PE control results for the
attack-convention comparison table in the ADS TIFS manuscript.

This script is intentionally modeled on `experiment3_perturbation_norm_v4.py`:
it uses the same shared-delta attack pattern, but instead of logging only
perturbation norms it evaluates clean and attacked validation accuracy.

Shared-delta convention
-----------------------
For PE objects repeated across transformer blocks, gradients are aggregated
across the 12 blocks and a single delta tensor is applied to every block for a
given buffer name:

  RoPE  : one shared delta for cos_cached, one for sin_cached, one for inv_freq
  ALiBi : one shared delta for slopes

For Learned and Sinusoidal PE there is only one positional object, so the shared
vs per-buffer distinction is effectively degenerate, but the same code path is
kept for completeness.

Output JSON structure
---------------------
{
  "metadata": {...},
  "results": {
    "learned": {
      "42": {
        "status": "ok",
        "checkpoint": "...",
        "clean_acc": 82.34,
        "epsilons": [0.05, 0.1, 0.2, 0.5, 1.0],
        "accuracies": [75.7, 64.6, ...],
        "attacks": {
          "0.05": {"accuracy": 75.7, "correct": 3785, "total": 5000, ...}
        }
      }
    }
  },
  "summary": {
    "mean_accuracy_by_pe": {
      "learned": [75.7, 64.6, 2.2, 1.3, 1.0]
    }
  }
}

Typical Colab usage
-------------------
ImageNet-100:
  !python ads_shared_delta_attack_convention.py \
      --models_dir "/content/drive/MyDrive/Trained models_ImageNet100" \
      --val_dir "/content/imagenet100/val" \
      --dataset imagenet \
      --output_path "/content/drive/MyDrive/ads_tfs_n6/data_n6/ads_shared_delta_imagenet100.json" \
      --batch_size 128

CIFAR-100:
  !python ads_shared_delta_attack_convention.py \
      --models_dir "/content/drive/MyDrive/Trained models_CIFAR100" \
      --dataset cifar \
      --val_dir "/content/cifar100_data" \
      --output_path "/content/drive/MyDrive/ads_tfs_n6/data_n6/ads_shared_delta_cifar100.json" \
      --batch_size 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Try to import VisionTransformer from the working directory, script directory,
# parent directory, and /content, matching the existing ADS Colab scripts.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/content")
from full_scale_experiment import VisionTransformer  # noqa: E402


PE_TYPES = ["learned", "sinusoidal", "rope", "alibi"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 1213]
DEFAULT_EPSILONS = [0.05, 0.10, 0.20, 0.50, 1.00]

MODEL_KWARGS_BASE = dict(
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.0,
)

DATASET_CONFIG = {
    "imagenet": {
        "display_name": "ImageNet-100",
        "num_classes": 100,
        "img_size": 224,
        "patch_size": 16,
        "grid_size": 14,
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    },
    "cifar": {
        "display_name": "CIFAR-100",
        "num_classes": 100,
        "img_size": 32,
        "patch_size": 4,
        "grid_size": 8,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                 [0.2675, 0.2565, 0.2761]),
        ]),
    },
}


# -----------------------------------------------------------------------------
# Buffer access and shared-delta PGD
# -----------------------------------------------------------------------------

def get_buffers(model: nn.Module, pe_type: str) -> Dict[str, List[torch.Tensor]]:
    """Return PE tensors as lists of per-block objects where applicable."""
    if pe_type == "rope":
        return {
            "cos_cached": [b.attn.rope.cos_cached for b in model.blocks],
            "sin_cached": [b.attn.rope.sin_cached for b in model.blocks],
            "inv_freq":   [b.attn.rope.inv_freq for b in model.blocks],
        }
    if pe_type == "alibi":
        return {"slopes": [b.attn.alibi.slopes for b in model.blocks]}
    if pe_type == "learned":
        return {"pos_embed": [model.pos_encoding.pos_embed]}
    if pe_type == "sinusoidal":
        return {"pe": [model.pos_encoding.pe]}
    raise ValueError(f"Unknown PE type: {pe_type}")


def snapshot_buffers(buffer_dict: Mapping[str, Sequence[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
    return {name: [b.data.clone() for b in bufs] for name, bufs in buffer_dict.items()}


def restore_buffers(buffer_dict: Mapping[str, Sequence[torch.Tensor]], originals: Mapping[str, Sequence[torch.Tensor]]) -> None:
    with torch.no_grad():
        for name, bufs in buffer_dict.items():
            for b, orig in zip(bufs, originals[name]):
                b.data.copy_(orig)


def set_requires_grad_all(buffer_dict: Mapping[str, Sequence[torch.Tensor]], requires_grad: bool) -> None:
    for bufs in buffer_dict.values():
        for b in bufs:
            b.requires_grad_(requires_grad)


def zero_all_buffer_grads(buffer_dict: Mapping[str, Sequence[torch.Tensor]]) -> None:
    for bufs in buffer_dict.values():
        for b in bufs:
            if b.grad is not None:
                b.grad.zero_()


def aggregate_grads_across_blocks(buffer_dict: Mapping[str, Sequence[torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
    """Aggregate gradients across blocks for each buffer name.

    This is the tied/shared-delta step. For RoPE, for example, all 12
    cos_cached gradients are summed to update one cos_cached delta tensor,
    which is then applied back to all 12 blocks.
    """
    aggregated: Dict[str, Optional[torch.Tensor]] = {}
    for name, bufs in buffer_dict.items():
        grads = [b.grad for b in bufs if b.grad is not None]
        if not grads:
            aggregated[name] = None
        elif len(grads) == 1:
            aggregated[name] = grads[0].detach().clone()
        else:
            aggregated[name] = torch.stack([g.detach() for g in grads]).sum(dim=0)
    return aggregated


def apply_shared_delta(
    buffer_dict: Mapping[str, Sequence[torch.Tensor]],
    deltas: Mapping[str, Optional[torch.Tensor]],
    originals: Mapping[str, Sequence[torch.Tensor]],
) -> None:
    """Apply one delta per buffer name to all corresponding blocks."""
    with torch.no_grad():
        for name, bufs in buffer_dict.items():
            delta = deltas.get(name)
            if delta is None:
                continue
            for b, orig in zip(bufs, originals[name]):
                b.data.copy_(orig + delta)


def pgd_pe_shared_delta(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    pe_type: str,
    epsilon: float,
    seed: int,
    steps: int,
    alpha_ratio: float,
    num_grad_batches: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
    """Compute shared-delta PGD perturbations and restore model to clean state."""
    model.eval()
    bufs = get_buffers(model, pe_type)
    originals = snapshot_buffers(bufs)

    if epsilon == 0:
        deltas = {name: torch.zeros_like(originals[name][0]) for name in bufs}
        return deltas, originals

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    deltas: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(originals[name][0]).uniform_(-epsilon, epsilon)
        for name in bufs
    }

    alpha = epsilon * alpha_ratio
    criterion = nn.CrossEntropyLoss()
    n_batches = min(num_grad_batches, len(loader))

    set_requires_grad_all(bufs, True)

    for _step in range(steps):
        apply_shared_delta(bufs, deltas, originals)
        model.zero_grad(set_to_none=True)
        zero_all_buffer_grads(bufs)

        for i, (images, labels) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

        agg_grads = aggregate_grads_across_blocks(bufs)
        with torch.no_grad():
            for name in deltas:
                grad = agg_grads[name]
                if grad is not None:
                    deltas[name] = deltas[name] + alpha * grad.sign()
                    deltas[name] = torch.clamp(deltas[name], -epsilon, epsilon)

    set_requires_grad_all(bufs, False)
    restore_buffers(bufs, originals)
    return deltas, originals


# -----------------------------------------------------------------------------
# Evaluation and loading
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, int, int]:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    acc = 100.0 * correct / total if total else float("nan")
    return acc, correct, total


def build_loader(args: argparse.Namespace) -> Tuple[DataLoader, int]:
    cfg = DATASET_CONFIG[args.dataset]
    if args.dataset == "imagenet":
        if not args.val_dir:
            raise ValueError("--val_dir is required for ImageNet-100")
        dataset = datasets.ImageFolder(args.val_dir, cfg["transform"])
    else:
        cache_dir = args.val_dir or "/content/cifar100_data"
        os.makedirs(cache_dir, exist_ok=True)
        dataset = datasets.CIFAR100(
            root=cache_dir,
            train=False,
            download=True,
            transform=cfg["transform"],
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader, len(dataset)


def load_clean_model(checkpoint_path: str, pe_type: str, dataset_cfg: Mapping[str, object], device: str) -> nn.Module:
    model_kwargs = {
        **MODEL_KWARGS_BASE,
        "img_size": dataset_cfg["img_size"],
        "patch_size": dataset_cfg["patch_size"],
        "num_classes": dataset_cfg["num_classes"],
        "pe_type": pe_type,
    }
    model = VisionTransformer(**model_kwargs).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def load_existing(path: str) -> Optional[dict]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path: str, obj: Mapping[str, object]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def normalize_eps_key(eps: float) -> str:
    # Keep keys compact and stable: 0.1, not 0.10; 1.0, not 1
    return str(float(eps))


def already_done(results: Mapping[str, object], pe_type: str, seed: int, epsilons: Sequence[float]) -> bool:
    try:
        run = results["results"][pe_type][str(seed)]  # type: ignore[index]
        if run.get("status") != "ok":
            return False
        attacks = run.get("attacks", {})
        for eps in epsilons:
            rec = attacks.get(normalize_eps_key(eps))
            if not isinstance(rec, dict) or "accuracy" not in rec:
                return False
        return "clean_acc" in run
    except Exception:
        return False


def summarize(results: Mapping[str, object], pe_types: Sequence[str], seeds: Sequence[int], epsilons: Sequence[float]) -> dict:
    out = {
        "epsilons": [float(e) for e in epsilons],
        "mean_accuracy_by_pe": {},
        "std_accuracy_by_pe": {},
        "n_by_pe": {},
    }
    for pe in pe_types:
        means = []
        stds = []
        ns = []
        for eps in epsilons:
            vals = []
            for seed in seeds:
                try:
                    rec = results["results"][pe][str(seed)]["attacks"][normalize_eps_key(eps)]  # type: ignore[index]
                    vals.append(float(rec["accuracy"]))
                except Exception:
                    pass
            ns.append(len(vals))
            if vals:
                t = torch.tensor(vals, dtype=torch.float64)
                means.append(float(t.mean().item()))
                stds.append(float(t.std(unbiased=True).item()) if len(vals) > 1 else 0.0)
            else:
                means.append(float("nan"))
                stds.append(float("nan"))
        out["mean_accuracy_by_pe"][pe] = means
        out["std_accuracy_by_pe"][pe] = stds
        out["n_by_pe"][pe] = ns
    return out


def print_summary(summary: Mapping[str, object], pe_types: Sequence[str]) -> None:
    epsilons = summary["epsilons"]  # type: ignore[index]
    print("\nShared-delta mean accuracy table")
    print("eps:", "  ".join(f"{e:g}" for e in epsilons))
    print("-" * 78)
    for pe in pe_types:
        vals = summary["mean_accuracy_by_pe"][pe]  # type: ignore[index]
        print(f"{pe:<11}", "  ".join(f"{v:6.1f}" for v in vals))


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    pe_types = list(args.pe_types)
    seeds = list(args.seeds)
    epsilons = [float(e) for e in args.epsilons]
    cfg = DATASET_CONFIG[args.dataset]

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} ({cfg['display_name']})")
    print(f"PE types: {pe_types}")
    print(f"Seeds: {seeds} (n={len(seeds)})")
    print(f"Epsilons: {epsilons}")
    print(f"Shared-delta PGD: steps={args.pgd_steps}, alpha_ratio={args.alpha_ratio}, grad_batches={args.num_grad_batches}")
    print(f"Output: {args.output_path}")

    print("\nLoading validation data...")
    val_loader, n_val = build_loader(args)
    print(f"Validation images: {n_val}; grid={cfg['grid_size']}x{cfg['grid_size']}")

    existing = load_existing(args.output_path)
    if existing is not None:
        print("\nFound existing output; resuming.")
        results = existing
        results.setdefault("metadata", {})["resumed_at"] = datetime.now().isoformat()
        results["metadata"]["resumed_with_seeds"] = seeds
        results["metadata"]["resumed_with_epsilons"] = epsilons
    else:
        results = {
            "metadata": {
                "experiment": "shared-delta / tied-buffer PGD-PE attack-convention accuracy control",
                "script": "ads_shared_delta_attack_convention.py",
                "timestamp": datetime.now().isoformat(),
                "dataset": args.dataset,
                "dataset_display": cfg["display_name"],
                "n_val_images": n_val,
                "device": str(device),
                "batch_size": args.batch_size,
                "pe_types": pe_types,
                "seeds": seeds,
                "epsilons": epsilons,
                "attack": {
                    "name": "pgd_pe_shared_delta",
                    "pattern": "shared_delta_all_12_blocks",
                    "description": (
                        "Gradients are aggregated across transformer blocks for each PE buffer name; "
                        "one delta tensor per buffer name is applied to all blocks."
                    ),
                    "pgd_steps": args.pgd_steps,
                    "alpha_ratio": args.alpha_ratio,
                    "num_grad_batches": args.num_grad_batches,
                    "objective": "maximize_cross_entropy",
                    "delta_init": "uniform(-epsilon, epsilon)",
                    "constraint": "L_inf per shared delta tensor",
                },
                "model_kwargs_base": MODEL_KWARGS_BASE,
            },
            "results": {},
        }

    start_all = time.time()
    total = len(pe_types) * len(seeds)
    combo = 0

    for pe in pe_types:
        results.setdefault("results", {}).setdefault(pe, {})
        for seed in seeds:
            combo += 1
            print("\n" + "=" * 88)
            print(f"[{combo}/{total}] PE={pe}, seed={seed}")
            print("=" * 88)

            if already_done(results, pe, seed, epsilons):
                print("Already complete; skipping.")
                continue

            ckpt = os.path.join(args.models_dir, f"{pe}_seed{seed}", "best_model.pth")
            if not os.path.exists(ckpt):
                print(f"MISSING CHECKPOINT: {ckpt}")
                results["results"][pe][str(seed)] = {"status": "missing_checkpoint", "checkpoint": ckpt}
                save_json(args.output_path, results)
                continue

            model_t0 = time.time()
            model = load_clean_model(ckpt, pe, cfg, device)
            print(f"Model loaded in {time.time() - model_t0:.1f}s")

            clean_acc, clean_correct, clean_total = evaluate_accuracy(model, val_loader, device)
            print(f"Clean accuracy: {clean_acc:.2f}% ({clean_correct}/{clean_total})")

            run_data = {
                "status": "ok",
                "checkpoint": ckpt,
                "clean_acc": float(clean_acc),
                "clean_correct": int(clean_correct),
                "clean_total": int(clean_total),
                "epsilons": epsilons,
                "accuracies": [],
                "attacks": {},
            }

            print(f"\n{'eps':<8} {'accuracy':<12} {'drop_pp':<12} {'correct/total':<18} {'time':<10}")
            print("-" * 70)

            for eps in epsilons:
                eps_key = normalize_eps_key(eps)
                t0 = time.time()
                try:
                    deltas, originals = pgd_pe_shared_delta(
                        model=model,
                        loader=val_loader,
                        device=device,
                        pe_type=pe,
                        epsilon=eps,
                        seed=seed,
                        steps=args.pgd_steps,
                        alpha_ratio=args.alpha_ratio,
                        num_grad_batches=args.num_grad_batches,
                    )
                    bufs = get_buffers(model, pe)
                    apply_shared_delta(bufs, deltas, originals)
                    acc, correct, total_eval = evaluate_accuracy(model, val_loader, device)
                    restore_buffers(bufs, originals)
                    elapsed = time.time() - t0

                    # Compact norm sanity record, useful for debugging but not the primary reported value.
                    norm_sanity = {}
                    for name, delta in deltas.items():
                        d = delta.detach()
                        norm_sanity[name] = {
                            "shape": list(d.shape),
                            "delta_inf": float(d.abs().max().item()),
                            "delta_inf_to_eps": float(d.abs().max().item() / eps) if eps > 0 else 0.0,
                        }

                    run_data["accuracies"].append(float(acc))
                    run_data["attacks"][eps_key] = {
                        "epsilon": float(eps),
                        "accuracy": float(acc),
                        "acc_drop_pp": float(clean_acc - acc),
                        "correct": int(correct),
                        "total": int(total_eval),
                        "elapsed_sec": float(elapsed),
                        "norm_sanity": norm_sanity,
                    }
                    print(f"{eps:<8g} {acc:<12.2f} {clean_acc - acc:<12.2f} {correct}/{total_eval:<12} {elapsed:.0f}s")
                    del deltas, originals

                except Exception as exc:
                    elapsed = time.time() - t0
                    print(f"{eps:<8g} ERROR: {type(exc).__name__}: {exc}")
                    import traceback
                    traceback.print_exc()
                    run_data["accuracies"].append(float("nan"))
                    run_data["attacks"][eps_key] = {
                        "epsilon": float(eps),
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_sec": float(elapsed),
                    }

            results["results"][pe][str(seed)] = run_data
            results["summary"] = summarize(results, pe_types, seeds, epsilons)
            save_json(args.output_path, results)
            print(f"Saved: {args.output_path}")

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    results["summary"] = summarize(results, pe_types, seeds, epsilons)
    results.setdefault("metadata", {})["completed_at"] = datetime.now().isoformat()
    results["metadata"]["total_elapsed_sec"] = float(time.time() - start_all)
    save_json(args.output_path, results)

    print("\n" + "=" * 88)
    print("COMPLETE")
    print("=" * 88)
    print_summary(results["summary"], pe_types)
    print(f"\nOutput: {args.output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate shared-delta PGD-PE attack-convention accuracy JSON.")
    p.add_argument("--models_dir", required=True, help="Dataset-specific checkpoint directory.")
    p.add_argument("--val_dir", default=None, help="ImageNet val directory or CIFAR cache directory.")
    p.add_argument("--dataset", required=True, choices=["imagenet", "cifar"])
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto.")
    p.add_argument("--pe_types", nargs="+", default=PE_TYPES, choices=PE_TYPES)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    p.add_argument("--pgd_steps", type=int, default=20)
    p.add_argument("--alpha_ratio", type=float, default=0.1)
    p.add_argument("--num_grad_batches", type=int, default=20)
    args = p.parse_args()
    if args.dataset == "imagenet" and not args.val_dir:
        p.error("--val_dir is required for --dataset imagenet")
    return args


if __name__ == "__main__":
    run(parse_args())
