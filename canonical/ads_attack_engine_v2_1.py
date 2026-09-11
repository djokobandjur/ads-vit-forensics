#!/usr/bin/env python3
"""
Corrected attack engine for ADS/TIFS relative-coordinate scale diagnostic v1.9.

Protocol lock implemented here
------------------------------
1) True iterative projected gradient ASCENT on cross-entropy.
2) At every PGD step the attacked tensors are set to base + current_delta
   BEFORE the new gradient is evaluated.
3) model.eval() during attack construction (dropout disabled).
4) Gradient objective uses the full fixed reference set by default.
5) Cross-entropy is accumulated with reduction='sum' and divided by the exact
   number of reference images, so batch partitioning cannot change the objective.
6) Zero-delta initialization unless a caller explicitly implements another lock.
7) Primary PE attack surface follows the manuscript text exactly:
     learned     -> one pos_embed tensor
     sinusoidal  -> one pe tensor
     rope        -> cos_cached + sin_cached in each of 12 blocks (NO inv_freq)
     alibi       -> slopes in each of 12 blocks
8) This module is used as the ATTACK ENGINE for the v2.1 canonical primary full-grid execution. Canonical ADS
   measurement is implemented separately as
   logit-domain float64 mean(per-image KL), with no probability floor. Legacy
   legacy measurement helpers retained below are NOT invoked by the v2.1 full-grid runner.

The helper is deliberately dataset-agnostic and imports no project model class.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import platform
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# Must be set before any CUDA context is initialized.  Keeping it at module
# import time makes the determinism lock independent of caller ordering.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader




def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _force_math_sdpa_backend() -> Dict[str, object]:
    """Force PyTorch SDPA to the math backend for cross-PE comparability.

    This is global process state.  It avoids backend selection changing because
    ALiBi supplies an attention mask while other PE families may not.
    """
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None:
        for name, value in [
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
            ("enable_cudnn_sdp", False),
        ]:
            fn = getattr(cuda_backend, name, None)
            if callable(fn):
                fn(value)
    return _sdpa_status()


def _sdpa_status() -> Dict[str, object]:
    cuda_backend = getattr(torch.backends, "cuda", None)
    def read(name):
        if cuda_backend is None:
            return None
        fn = getattr(cuda_backend, name, None)
        if callable(fn):
            try:
                return bool(fn())
            except TypeError:
                return None
        return None
    flash = read("flash_sdp_enabled")
    mem = read("mem_efficient_sdp_enabled")
    math = read("math_sdp_enabled")
    cudnn = read("cudnn_sdp_enabled")
    math_only = (math is True and flash in {False, None} and mem in {False, None} and cudnn in {False, None})
    return {
        "sdpa_backend": "math_only" if math_only else "explicit_math_preferred",
        "flash_sdp_enabled": flash,
        "mem_efficient_sdp_enabled": mem,
        "math_sdp_enabled": math,
        "cudnn_sdp_enabled": cudnn,
    }


def configure_deterministic_runtime(seed: int = 0) -> Dict[str, object]:
    """Lock deterministic CUDA/PyTorch settings used by the canonical rerun."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _force_math_sdpa_backend()
    torch.use_deterministic_algorithms(True, warn_only=False)
    return runtime_provenance(seed=seed)


def runtime_provenance(seed: Optional[int] = None) -> Dict[str, object]:
    gpu = None
    capability = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
    out = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": gpu,
        "gpu_capability": capability,
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": seed,
    }
    out.update(_sdpa_status())
    return out

def file_identity(path: Optional[str]) -> Optional[Dict[str, object]]:
    if not path:
        return None
    q = Path(path)
    if not q.exists():
        return {"path": str(q), "exists": False}
    return {"path": str(q.resolve()), "exists": True, "bytes": q.stat().st_size, "sha256": sha256_file(str(q))}


def _flatten_bases(g: "AttackGroup") -> torch.Tensor:
    return torch.cat([b.detach().reshape(-1).float().cpu() for b in g.bases])


Bound = Union[float, torch.Tensor]


def _bound_tensor_for_delta(bound: Bound, delta: torch.Tensor) -> torch.Tensor:
    if isinstance(bound, torch.Tensor):
        return bound.detach().to(delta.device, dtype=delta.dtype)
    return torch.full_like(delta, float(bound))


def _bound_summary(bound: Bound) -> Dict[str, float]:
    if isinstance(bound, torch.Tensor):
        b = bound.detach().reshape(-1).float().cpu()
        return {
            "min": float(b.min().item()) if b.numel() else 0.0,
            "max": float(b.max().item()) if b.numel() else 0.0,
            "rms": float(torch.sqrt((b*b).mean()).item()) if b.numel() else 0.0,
            "mean": float(b.mean().item()) if b.numel() else 0.0,
            "zero_fraction": float((b == 0).float().mean().item()) if b.numel() else 0.0,
        }
    x = float(bound)
    return {"min": x, "max": x, "rms": abs(x), "mean": x, "zero_fraction": 1.0 if x == 0 else 0.0}


def attack_group_geometry(g: "AttackGroup", bound: Bound) -> Dict[str, object]:
    base = _flatten_bases(g)
    d = g.delta.detach().reshape(-1).float().cpu()
    k = len(g.bases)
    d_total_l2 = float(d.norm().item() * (k ** 0.5))
    delta_mean_abs = float(d.abs().mean().item())
    bt = _bound_tensor_for_delta(bound, g.delta).detach().reshape(-1).float().cpu()
    active = bt > 0
    sat = float(((d.abs() >= bt * (1.0 - 1e-6)) & active).float().sum().item() / active.sum().item()) if int(active.sum()) else 0.0
    flips = 0
    nonzero = 0
    for b in g.bases:
        bb = b.detach().float().cpu()
        dd = g.delta.detach().float().cpu()
        nz = bb != 0
        nonzero += int(nz.sum().item())
        flips += int(((bb * (bb + dd) < 0) & nz).sum().item())
    return {
        "name": g.name,
        "n_tensors": k,
        "n_coordinates_total": int(base.numel()),
        "base_linf": float(base.abs().max().item()) if base.numel() else 0.0,
        "base_l2": float(base.norm().item()),
        "base_rms": float(torch.sqrt((base * base).mean()).item()) if base.numel() else 0.0,
        "base_mean_abs": float(base.abs().mean().item()) if base.numel() else 0.0,
        "delta_linf": float(d.abs().max().item()) if d.numel() else 0.0,
        "delta_l2_total": d_total_l2,
        "delta_mean_abs": delta_mean_abs,
        "saturated_fraction": sat,
        "sign_flip_fraction_nonzero_base": float(flips / nonzero) if nonzero else 0.0,
        "coordinate_bound_summary": _bound_summary(bound),
    }

def summarize_attack_geometry(groups: Sequence["AttackGroup"], bound: Bound) -> Dict[str, object]:
    per = [attack_group_geometry(g, bound) for g in groups]
    return {
        "per_group": per,
        "any_sign_flip": any(x["sign_flip_fraction_nonzero_base"] > 0 for x in per),
        "max_sign_flip_fraction": max((x["sign_flip_fraction_nonzero_base"] for x in per), default=0.0),
        "max_delta_linf": max((x["delta_linf"] for x in per), default=0.0),
        "total_delta_l2": float(sum(float(x["delta_l2_total"])**2 for x in per) ** 0.5),
        "total_base_l2": float(sum(float(x["base_l2"])**2 for x in per) ** 0.5),
    }


@dataclass
class AttackGroup:
    """One projected delta shared by one or more tensors."""
    name: str
    params: List[nn.Parameter]
    bases: List[torch.Tensor]
    delta: torch.Tensor


def _replace_tensor_with_parameter(module: nn.Module, attr: str, device: str) -> nn.Parameter:
    value = getattr(module, attr)
    if isinstance(value, nn.Parameter):
        value.requires_grad_(True)
        return value
    data = value.detach().clone().to(device)
    # Remove registered buffer if present before installing a Parameter.
    if attr in module._buffers:
        del module._buffers[attr]
    elif hasattr(module, attr):
        delattr(module, attr)
    param = nn.Parameter(data, requires_grad=True)
    setattr(module, attr, param)
    return param


def _single_group(name: str, p: nn.Parameter) -> AttackGroup:
    base = p.detach().clone()
    return AttackGroup(name=name, params=[p], bases=[base], delta=torch.zeros_like(base))


def collect_pe_groups(model: nn.Module, pe_type: str, device: str,
                      delta_convention: str = "per_buffer") -> List[AttackGroup]:
    """Return PE attack groups under the locked manuscript attack surface.

    delta_convention:
      per_buffer : independent delta for every replicated RoPE/ALiBi tensor.
      shared     : one delta per tensor *kind* tied across blocks.
                   RoPE => one shared cos delta + one shared sin delta.
                   ALiBi => one shared slopes delta.
    """
    if delta_convention not in {"per_buffer", "shared"}:
        raise ValueError(f"Unsupported delta_convention={delta_convention!r}")

    pe_type = pe_type.lower()
    groups: List[AttackGroup] = []

    if pe_type == "learned":
        p = model.pos_encoding.pos_embed
        p.requires_grad_(True)
        groups.append(_single_group("pos_embed", p))
        return groups

    if pe_type == "sinusoidal":
        p = _replace_tensor_with_parameter(model.pos_encoding, "pe", device)
        groups.append(_single_group("pe", p))
        return groups

    if pe_type == "rope":
        cos_params: List[nn.Parameter] = []
        sin_params: List[nn.Parameter] = []
        for i, block in enumerate(model.blocks):
            rope = block.attn.rope
            cos_p = _replace_tensor_with_parameter(rope, "cos_cached", device)
            sin_p = _replace_tensor_with_parameter(rope, "sin_cached", device)
            cos_params.append(cos_p)
            sin_params.append(sin_p)
            if delta_convention == "per_buffer":
                groups.append(_single_group(f"block{i}.cos_cached", cos_p))
                groups.append(_single_group(f"block{i}.sin_cached", sin_p))
        if delta_convention == "shared":
            cos_bases = [p.detach().clone() for p in cos_params]
            sin_bases = [p.detach().clone() for p in sin_params]
            groups.append(AttackGroup("shared.cos_cached", cos_params, cos_bases,
                                      torch.zeros_like(cos_bases[0])))
            groups.append(AttackGroup("shared.sin_cached", sin_params, sin_bases,
                                      torch.zeros_like(sin_bases[0])))
        return groups

    if pe_type.startswith("alibi"):
        # Works for standard ALiBi and grid-aware 2D variants. Select every
        # registered buffer whose attribute name contains "slope" (excluding
        # distance buffers), preserving one independent delta per replicated
        # buffer under the primary convention.
        by_kind: Dict[str, List[nn.Parameter]] = {}
        found = 0
        for module_name, module in model.named_modules():
            for attr, buf in list(module._buffers.items()):
                if buf is None:
                    continue
                low = attr.lower()
                if "slope" not in low or "rel_dist" in low:
                    continue
                p = _replace_tensor_with_parameter(module, attr, device)
                found += 1
                by_kind.setdefault(attr, []).append(p)
                if delta_convention == "per_buffer":
                    groups.append(_single_group(f"{module_name}.{attr}", p))
        if found == 0:
            # Fallback for unusual modules that expose slopes as Parameters.
            for name, p in model.named_parameters():
                if "slope" in name.lower() and "rel_dist" not in name.lower():
                    p.requires_grad_(True)
                    found += 1
                    by_kind.setdefault(name.split(".")[-1], []).append(p)
                    if delta_convention == "per_buffer":
                        groups.append(_single_group(name, p))
        if found == 0:
            raise RuntimeError(f"No slope-like PE tensors found for {pe_type}")
        if delta_convention == "shared":
            for kind, params in by_kind.items():
                bases = [p.detach().clone() for p in params]
                groups.append(AttackGroup(f"shared.{kind}", params, bases,
                                          torch.zeros_like(bases[0])))
        return groups

    raise ValueError(f"Unknown PE type: {pe_type}")


def collect_weight_groups(model: nn.Module, surface: str,
                          include_biases: bool = False) -> List[AttackGroup]:
    """Collect non-PE specificity attack variables.

    By default this follows the Methods wording literally: QKV and MLP *weight
    matrices* only.  Set include_biases=True only for a legacy sensitivity run.
    """
    surface = surface.lower()
    selected: List[Tuple[str, nn.Parameter]] = []

    for name, p in model.named_parameters():
        clean = name.replace("_orig_mod.", "")
        if surface == "qkv_only":
            if "attn.qkv.weight" in clean or (include_biases and "attn.qkv.bias" in clean):
                selected.append((clean, p))
        elif surface == "mlp_only":
            if ".mlp." in clean and clean.endswith("weight"):
                selected.append((clean, p))
            elif include_biases and ".mlp." in clean and clean.endswith("bias"):
                selected.append((clean, p))
        elif surface == "all_non_pe_weights":
            # Canonical specificity control: all NON-PE trainable model parameters.
            # This removes the representation asymmetry whereby Learned pos_embed is
            # an nn.Parameter while Sinusoidal/RoPE/ALiBi PE state is stored as buffers.
            pe_like = any(tok in clean.lower() for tok in [
                "pos_embed", ".pe", "cos_cached", "sin_cached", "inv_freq", "slope"
            ])
            if p.requires_grad and not pe_like:
                selected.append((clean, p))
        else:
            raise ValueError(f"Unknown weight attack surface: {surface}")

    if not selected:
        raise RuntimeError(f"No parameters selected for surface={surface}")

    return [_single_group(name, p) for name, p in selected]


def collect_attack_groups(model: nn.Module, pe_type: str, surface: str, device: str,
                          delta_convention: str = "per_buffer",
                          include_biases: bool = False) -> List[AttackGroup]:
    if surface == "pe_only":
        return collect_pe_groups(model, pe_type, device, delta_convention)
    return collect_weight_groups(model, surface, include_biases=include_biases)


def _apply_groups(groups: Sequence[AttackGroup]) -> None:
    with torch.no_grad():
        for g in groups:
            for p, base in zip(g.params, g.bases):
                p.copy_(base + g.delta)


def _aggregate_group_grad(g: AttackGroup) -> Optional[torch.Tensor]:
    grads = [p.grad for p in g.params if p.grad is not None]
    if not grads:
        return None
    if len(grads) == 1:
        return grads[0].detach()
    # Shared-delta control: sum gradients from the replicated tensors.
    return torch.stack([x.detach() for x in grads], dim=0).sum(dim=0)


def save_attack_delta(groups: Sequence[AttackGroup], path: str, metadata: Optional[Mapping[str, object]] = None,
                      storage_dtype: str = "float32", protocol_version: Optional[str] = None,
                      execution_implementation: Optional[str] = None) -> Dict[str, object]:
    """Persist an attack state so later measurement changes do not require a new PGD run."""
    if storage_dtype not in {"float32", "float16"}:
        raise ValueError("storage_dtype must be float32 or float16")
    dtype = torch.float32 if storage_dtype == "float32" else torch.float16
    if not protocol_version or not execution_implementation:
        raise ValueError("delta artifact protocol_version and execution_implementation must be supplied by caller")
    artifact = {
        "format_version": "ADS_DELTA_ARTIFACT_v1",
        "protocol_version": str(protocol_version),
        "execution_implementation": str(execution_implementation),
        "storage_dtype": storage_dtype,
        "metadata": dict(metadata or {}),
        "groups": [
            {
                "name": g.name,
                "n_target_tensors": len(g.params),
                "target_shapes": [list(p.shape) for p in g.params],
                "delta": g.delta.detach().cpu().to(dtype),
            }
            for g in groups
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    torch.save(artifact, tmp)
    os.replace(tmp, out)
    return file_identity(str(out))


def apply_saved_delta(clean_model: nn.Module, pe_type: str, surface: str, delta_path: str, device: str,
                      delta_convention: str = "per_buffer", include_biases: bool = False) -> Tuple[nn.Module, Dict[str, object]]:
    """Reconstruct an attacked model from a saved delta artifact."""
    artifact = torch.load(delta_path, map_location="cpu", weights_only=False)
    pm = copy.deepcopy(clean_model).to(device)
    pm.eval()
    groups = collect_attack_groups(pm, pe_type, surface, device, delta_convention, include_biases)
    saved = {str(x["name"]): x for x in artifact.get("groups", [])}
    if set(saved) != {g.name for g in groups}:
        raise RuntimeError("Saved delta group names do not match the requested attack surface")
    for g in groups:
        d = saved[g.name]["delta"].to(device=device, dtype=g.delta.dtype)
        if tuple(d.shape) != tuple(g.delta.shape):
            raise RuntimeError(f"Saved delta shape mismatch for {g.name}: {tuple(d.shape)} vs {tuple(g.delta.shape)}")
        g.delta = d
    _apply_groups(groups)
    return pm, artifact


def projected_ce_ascent(
    clean_model: nn.Module,
    pe_type: str,
    surface: str,
    epsilon: float,
    ref_loader: DataLoader,
    device: str,
    steps: int = 20,
    alpha_ratio: float = 0.1,
    delta_convention: str = "per_buffer",
    include_biases: bool = False,
    budget_mode: str = "absolute",
    delta_save_path: Optional[str] = None,
    delta_storage_dtype: str = "float32",
    progress_label: Optional[str] = None,
    artifact_protocol_version: Optional[str] = None,
    artifact_execution_implementation: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Return a perturbed model under the corrected full-reference PGD protocol."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    pm = copy.deepcopy(clean_model).to(device)
    pm.eval()

    if epsilon == 0.0:
        return pm, {
            "epsilon": 0.0,
            "budget_mode": budget_mode,
            "steps": steps,
            "alpha_ratio": alpha_ratio,
            "objective": "full_reference_mean_cross_entropy_ascent",
            "attack_mode": "eval",
            "delta_init": "zero",
            "surface": surface,
            "delta_convention": delta_convention,
            "n_ref_images": len(ref_loader.dataset),
            "group_count": 0,
        }

    groups = collect_attack_groups(
        pm, pe_type, surface, device,
        delta_convention=delta_convention,
        include_biases=include_biases,
    )
    if budget_mode not in {"absolute", "relative_rms", "relative_linf", "relative_coordinate"}:
        raise ValueError(f"Unsupported budget_mode={budget_mode!r}")
    bounds: Dict[str, Bound] = {}
    for g in groups:
        base = _flatten_bases(g)
        if budget_mode == "absolute":
            bound: Bound = float(epsilon)
        elif budget_mode == "relative_rms":
            scale = float(torch.sqrt((base * base).mean()).item()) if base.numel() else 0.0
            bound = float(epsilon) * scale
        elif budget_mode == "relative_linf":
            scale = float(base.abs().max().item()) if base.numel() else 0.0
            bound = float(epsilon) * scale
        else:
            # Multiplicative / coordinate-relative sensitivity box:
            # |delta_j| <= rho * |theta_j|.  For a shared delta tied across
            # replicated tensors, use the minimum absolute base coordinate so
            # the constraint holds for every tensor receiving that delta.
            abs_bases = torch.stack([b.detach().abs().to(g.delta.device, dtype=g.delta.dtype) for b in g.bases], dim=0)
            coord_scale = abs_bases.amin(dim=0)
            bound = float(epsilon) * coord_scale
        if isinstance(bound, float) and bound < 0:
            raise RuntimeError(f"Negative bound for group {g.name}")
        bounds[g.name] = bound
    total_images = len(ref_loader.dataset)
    if total_images <= 0:
        raise RuntimeError("Reference loader is empty")

    criterion_sum = nn.CrossEntropyLoss(reduction="sum")
    pm.eval()  # lock dropout off for every step

    for _step in range(steps):
        if progress_label:
            print(f"  [{progress_label}] PGD step {_step+1}/{steps}", flush=True)
        # Critical correction: evaluate gradient at base + CURRENT delta.
        _apply_groups(groups)
        pm.zero_grad(set_to_none=True)

        seen = 0
        for images, labels in ref_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = pm(images)
            loss_sum = criterion_sum(logits, labels)
            # Backward each batch with exact dataset normalization. This is
            # mathematically the gradient of the full-reference mean CE and
            # avoids retaining all 256-image graphs simultaneously.
            (loss_sum / total_images).backward()
            seen += labels.numel()

        if seen != total_images:
            raise RuntimeError(f"Attack objective saw {seen} images, expected {total_images}")

        with torch.no_grad():
            for g in groups:
                grad = _aggregate_group_grad(g)
                if grad is None:
                    raise RuntimeError(f"No gradient for attack group {g.name}")
                bound = bounds[g.name]
                bt = _bound_tensor_for_delta(bound, g.delta)
                alpha_g = bt * float(alpha_ratio)
                candidate = g.delta + alpha_g * grad.sign()
                g.delta = torch.maximum(torch.minimum(candidate, bt), -bt)

    _apply_groups(groups)
    pm.eval()

    per_geom = [attack_group_geometry(g, bounds[g.name]) for g in groups]
    geometry = {
        "per_group": per_geom,
        "any_sign_flip": any(x["sign_flip_fraction_nonzero_base"] > 0 for x in per_geom),
        "max_sign_flip_fraction": max((x["sign_flip_fraction_nonzero_base"] for x in per_geom), default=0.0),
        "max_delta_linf": max((x["delta_linf"] for x in per_geom), default=0.0),
        "total_delta_l2": float(sum(float(x["delta_l2_total"])**2 for x in per_geom) ** 0.5),
        "total_base_l2": float(sum(float(x["base_l2"])**2 for x in per_geom) ** 0.5),
    }
    meta = {
        "epsilon": float(epsilon),
        "budget_mode": budget_mode,
        "effective_bound_summary_by_group": {name: _bound_summary(bound) for name, bound in bounds.items()},
        "steps": int(steps),
        "alpha_ratio": float(alpha_ratio),
        "objective": "full_reference_mean_cross_entropy_ascent",
        "attack_mode": "eval",
        "delta_init": "zero",
        "constraint": "coordinatewise_relative_box" if budget_mode == "relative_coordinate" else "L_inf_per_delta_group",
        "surface": surface,
        "delta_convention": delta_convention,
        "include_biases": bool(include_biases),
        "n_ref_images": int(total_images),
        "group_count": len(groups),
        "attack_geometry": geometry,
    }
    if delta_save_path:
        delta_ident = save_attack_delta(
            groups, delta_save_path,
            metadata={
                "pe_type": pe_type, "surface": surface, "epsilon": float(epsilon),
                "budget_mode": budget_mode, "delta_convention": delta_convention,
                "steps": int(steps), "alpha_ratio": float(alpha_ratio),
            },
            storage_dtype=delta_storage_dtype,
            protocol_version=artifact_protocol_version,
            execution_implementation=artifact_execution_implementation,
        )
        meta["delta_artifact"] = delta_ident
        meta["delta_storage_dtype"] = delta_storage_dtype
    return pm, meta


@torch.no_grad()
def measure_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
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
    return 100.0 * correct / total if total else float("nan")


@torch.no_grad()
def get_mean_attention(model: nn.Module, loader: DataLoader, device: str,
                       layers: Optional[Sequence[int]] = None) -> List[torch.Tensor]:
    """Exact image-weighted mean attention; returns CPU tensors HxNxN (float32).

    This helper exists for legacy/ref-profile compatibility only. Canonical ADS
    is measured by :func:`measure_ads_pair` below.
    """
    model.eval()
    wanted = None if layers is None else set(layers)
    totals: Dict[int, torch.Tensor] = {}
    total_images = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _, attentions = model.forward_with_attention(images)
        for i, a in enumerate(attentions):
            if wanted is not None and i not in wanted:
                continue
            s = a.sum(dim=0).detach().cpu().float()
            totals[i] = s if i not in totals else totals[i] + s
        total_images += images.size(0)
    if total_images == 0:
        raise RuntimeError("Empty loader passed to get_mean_attention")
    indices = sorted(totals)
    return [totals[i] / total_images for i in indices]


def _kl_rows_typed(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10,
                   dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Row-wise KL with explicit additive floor and explicit arithmetic dtype."""
    if eps <= 0:
        raise ValueError("KL epsilon_s must be > 0")
    p = p.to(dtype=dtype) + float(eps)
    q = q.to(dtype=dtype) + float(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    return (p * torch.log(p / q)).sum(dim=-1)


def kl_rows(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Legacy helper kept for compatibility; arithmetic is float32."""
    p = p.float() + eps
    q = q.float() + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    return (p * torch.log(p / q)).sum(dim=-1)


def compute_ads_from_means(clean_attn: Sequence[torch.Tensor],
                           test_attn: Sequence[torch.Tensor]) -> Dict[str, object]:
    """Legacy implemented operator = KL(mean clean attention || mean test attention).

    Retained only for archived-number compatibility. It is NOT canonical ADS in v1.3.
    """
    if len(clean_attn) != len(test_attn):
        raise ValueError("clean/test attention layer count mismatch")
    per_layer = [float(kl_rows(p, q).mean().item()) for p, q in zip(clean_attn, test_attn)]
    return _ads_summary(per_layer)


def _ads_summary(per_layer: Sequence[float]) -> Dict[str, object]:
    vals = [float(x) for x in per_layer]
    if len(vals) == 1:
        return {"layer4_ads": vals[0], "per_layer_ads": vals}
    return {
        "mean_ads": float(np.mean(vals)),
        "mid_layer_ads": float(np.mean(vals[4:9])),
        "layer4_ads": float(vals[3]),
        "per_layer_ads": vals,
    }


def ads_operator_spec(kl_epsilon: float = 1e-10) -> Dict[str, object]:
    return {
        "name": "ADS_Definition_1",
        "sample_aggregation": "mean_of_per_image_divergences",
        "per_image_reduction": "mean_over_heads_and_query_rows",
        "row_divergence": "KL(clean_attention_row || test_attention_row)",
        "probability_floor_epsilon_s": float(kl_epsilon),
        "probability_floor_application": "add_to_each_probability_then_renormalize_rows",
        "divergence_arithmetic": "float64",
        "attention_forward_dtype": "model_native",
        "legacy_compat_operator": "KL(mean_attention_clean || mean_attention_test) in float32",
    }


def operator_spec_hash(kl_epsilon: float = 1e-10) -> str:
    raw = json.dumps(ads_operator_spec(kl_epsilon), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _save_per_image_ads(path: str, matrix: np.ndarray, layer_indices: Sequence[int],
                        kl_epsilon: float) -> Dict[str, object]:
    """Persist N_images x N_layers canonical per-image ADS scalars."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp.npz")
    np.savez_compressed(
        tmp,
        per_image_ads=np.asarray(matrix, dtype=np.float32),
        layer_indices=np.asarray(layer_indices, dtype=np.int16),
        kl_epsilon=np.asarray([float(kl_epsilon)], dtype=np.float64),
        operator_spec_hash=np.asarray([operator_spec_hash(kl_epsilon)]),
    )
    os.replace(tmp, out)
    return file_identity(str(out))


@torch.no_grad()
def measure_ads_pair(clean_model: nn.Module, test_model: nn.Module,
                     loader: DataLoader, device: str,
                     kl_epsilon: float = 1e-10,
                     layers: Optional[Sequence[int]] = None,
                     per_image_save_path: Optional[str] = None,
                     include_legacy_refprofile: bool = True) -> Dict[str, object]:
    """Measure canonical Definition-1 ADS and optional legacy compatibility ADS.

    Canonical result:
        mean_x mean_{h,i} KL(A_clean^{h,i}(x) || A_test^{h,i}(x))
    KL arithmetic is float64. Per-image scalars can be persisted cheaply.
    """
    clean_model.eval(); test_model.eval()
    wanted = None if layers is None else set(int(x) for x in layers)
    per_layer_values: Dict[int, List[np.ndarray]] = {}
    clean_sums: Dict[int, torch.Tensor] = {}
    test_sums: Dict[int, torch.Tensor] = {}
    total_images = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _, ac = clean_model.forward_with_attention(images)
        _, at = test_model.forward_with_attention(images)
        if len(ac) != len(at):
            raise RuntimeError("Clean/test model returned different attention layer counts")
        for i, (p, q) in enumerate(zip(ac, at)):
            if wanted is not None and i not in wanted:
                continue
            # B,H,N,N -> row KL B,H,N -> per-image scalar B
            v = _kl_rows_typed(p, q, eps=kl_epsilon, dtype=torch.float64).mean(dim=(1, 2))
            per_layer_values.setdefault(i, []).append(v.detach().cpu().numpy().astype(np.float64, copy=False))
            if include_legacy_refprofile:
                ps = p.sum(dim=0).detach().cpu().float()
                qs = q.sum(dim=0).detach().cpu().float()
                clean_sums[i] = ps if i not in clean_sums else clean_sums[i] + ps
                test_sums[i] = qs if i not in test_sums else test_sums[i] + qs
        total_images += int(images.size(0))

    if total_images == 0:
        raise RuntimeError("Empty loader passed to measure_ads_pair")
    indices = sorted(per_layer_values)
    arrays = [np.concatenate(per_layer_values[i], axis=0) for i in indices]
    if any(len(a) != total_images for a in arrays):
        raise RuntimeError("Per-image ADS count mismatch")
    matrix = np.stack(arrays, axis=1)  # N x L
    canonical = _ads_summary(matrix.mean(axis=0, dtype=np.float64).tolist())
    canonical.update({
        "operator": "mean(per-image KL)",
        "kl_epsilon": float(kl_epsilon),
        "divergence_arithmetic": "float64",
        "operator_spec_hash": operator_spec_hash(kl_epsilon),
        "n_images": int(total_images),
        "per_image_distribution": {
            "median_per_layer": np.median(matrix, axis=0).astype(float).tolist(),
            "q25_per_layer": np.quantile(matrix, 0.25, axis=0).astype(float).tolist(),
            "q75_per_layer": np.quantile(matrix, 0.75, axis=0).astype(float).tolist(),
            "max_per_layer": matrix.max(axis=0).astype(float).tolist(),
        },
    })
    if per_image_save_path:
        canonical["per_image_artifact"] = _save_per_image_ads(per_image_save_path, matrix, indices, kl_epsilon)

    out: Dict[str, object] = {"canonical": canonical}
    if include_legacy_refprofile:
        cm = [clean_sums[i] / total_images for i in indices]
        tm = [test_sums[i] / total_images for i in indices]
        legacy = compute_ads_from_means(cm, tm)
        legacy.update({"operator": "KL(mean||mean)", "arithmetic": "float32", "n_images": int(total_images)})
        out["legacy_refprofile"] = legacy
    return out


@torch.no_grad()
def per_image_ads_layer(clean_model: nn.Module, test_model: nn.Module,
                        loader: DataLoader, device: str, layer: int = 3,
                        kl_epsilon: float = 1e-10) -> np.ndarray:
    """Return one canonical per-image ADS scalar for a selected layer."""
    clean_model.eval(); test_model.eval(); chunks = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _, ac = clean_model.forward_with_attention(images)
        _, at = test_model.forward_with_attention(images)
        p, q = ac[layer], at[layer]
        v = _kl_rows_typed(p, q, eps=kl_epsilon, dtype=torch.float64).mean(dim=(1, 2))
        chunks.append(v.detach().cpu().numpy().astype(np.float64, copy=False))
    if not chunks:
        raise RuntimeError("Empty loader passed to per_image_ads_layer")
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def compute_mean_per_image_kl(clean_model: nn.Module, test_model: nn.Module,
                              loader: DataLoader, device: str,
                              layers: Optional[Sequence[int]] = None,
                              kl_epsilon: float = 1e-10) -> Dict[str, object]:
    """Compatibility wrapper returning only canonical Definition-1 ADS."""
    return measure_ads_pair(clean_model, test_model, loader, device,
                            kl_epsilon=kl_epsilon, layers=layers,
                            include_legacy_refprofile=False)["canonical"]

def atomic_json_dump(path: str, obj: object) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, out)




def checkpoint_identity_path(output_path: str) -> str:
    return output_path + ".checkpoints.json"


def record_checkpoint_identity(output_path: str, pe_type: str, seed: int, checkpoint_path: str) -> Dict[str, object]:
    side = checkpoint_identity_path(output_path)
    if os.path.exists(side):
        with open(side, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = {"protocol_version": "ADS_TIFS_CANONICAL_PRIMARY_FULL_GRID_v2_1_20260909", "checkpoints": {}}
    ident = file_identity(checkpoint_path)
    obj.setdefault("checkpoints", {})[f"{pe_type}/seed{seed}"] = ident
    atomic_json_dump(side, obj)
    return ident


def protocol_sidecar_path(output_path: str) -> str:
    return output_path + ".protocol.json"


def write_protocol_sidecar(output_path: str, payload: Mapping[str, object]) -> None:
    base = {
        "protocol_version": "ADS_TIFS_CANONICAL_PRIMARY_FULL_GRID_v2_1_20260909",
        "pgd_update": "base_plus_current_delta_before_each_gradient",
        "attack_mode": "eval",
        "attack_objective": "full_reference_mean_cross_entropy_ascent",
        "delta_init": "zero",
        "ads_primary_operator": "mean(per-image KL) — manuscript Definition 1",
        "ads_divergence_arithmetic": "float64",
        "ads_probability_floor_epsilon_s": payload.get("kl_epsilon", 1e-10) if isinstance(payload, Mapping) else 1e-10,
        "ads_operator_spec_hash": operator_spec_hash(float(payload.get("kl_epsilon", 1e-10))) if isinstance(payload, Mapping) else operator_spec_hash(1e-10),
        "legacy_compat_operator": "KL(mean_attention_clean || mean_attention_test) retained as compatibility field",
        "rope_primary_attack_surface": ["cos_cached_per_block", "sin_cached_per_block"],
        "rope_inv_freq_attacked": False,
        "runtime": runtime_provenance(),
    }
    base.update(dict(payload))
    refp = base.get("reference_indices_path")
    if refp:
        base["reference_indices_identity"] = file_identity(str(refp))
    atomic_json_dump(protocol_sidecar_path(output_path), base)
