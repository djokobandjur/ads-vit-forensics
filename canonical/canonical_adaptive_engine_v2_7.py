#!/usr/bin/env python3
"""Differentiable canonical ADS attack engine for LOCK-003 pilot.

Scientific ADS semantics:
- native PRE-softmax attention logits
- float64 log_softmax
- no probability epsilon floor
- no renormalization
- per-image KL first, then mean over reference images

The ADS path deliberately reproduces the MANUAL attention measurement path used
by forward_with_attention_logits(), while the CE term uses model.forward() and
therefore the standard SDPA value path. This keeps:
  CE == canonical CE attack semantics
  ADS == canonical ADS measurement semantics
without silently changing either path.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn.functional as F

from ads_attack_engine_v2_1 import (
    collect_pe_groups, _apply_groups, _aggregate_group_grad,
    _bound_tensor_for_delta, attack_group_geometry, save_attack_delta
)

CANONICAL_OPERATOR_HASH = "093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362"

def canonical_row_kl(clean_logits: torch.Tensor, test_logits: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(clean_logits.to(torch.float64), dim=-1)
    lq = F.log_softmax(test_logits.to(torch.float64), dim=-1)
    p = lp.exp()
    row = (p * (lp - lq)).sum(dim=-1)
    mn = float(row.detach().min().item())
    if mn < -1e-11:
        raise RuntimeError(f"Canonical KL negative below tolerance: {mn}")
    # Identical numerical safeguard to canonical measurement implementation.
    return row.clamp_min(0.0)

def canonical_per_image(clean_logits: torch.Tensor, test_logits: torch.Tensor) -> torch.Tensor:
    return canonical_row_kl(clean_logits, test_logits).mean(dim=(1,2))

def _manual_attention_block_with_logits(block, x):
    """Exactly reproduce TransformerBlock(..., return_attention_logits=True).

    The function is external so the canonical v1.6 model class remains byte-for-byte
    unchanged for the lambda=0 regression check.
    """
    a = block.attn
    y = block.norm1(x)
    B,N,C = y.shape
    qkv = a.qkv(y).reshape(B,N,3,a.num_heads,a.head_dim).permute(2,0,3,1,4)
    q,k,v = qkv[0],qkv[1],qkv[2]
    if a.pe_type == "rope":
        q,k = a.rope(q,k,N)
    bias = a.alibi.get_bias(N) if a.pe_type == "alibi" else None
    z = (q @ k.transpose(-2,-1)) * a.scale
    if bias is not None:
        z = z + bias
    w = z.softmax(dim=-1)
    out = (w @ v).transpose(1,2).reshape(B,N,C)
    out = a.proj(out)
    x = x + out
    x = x + block.mlp(block.norm2(x))
    return x,z

def canonical_attention_logits_grad(model, images, layers_0based: Sequence[int]) -> Dict[int,torch.Tensor]:
    """Differentiable native pre-softmax logits on the canonical measurement path."""
    wanted=sorted(set(int(x) for x in layers_0based))
    if not wanted or wanted[0] < 0 or wanted[-1] >= len(model.blocks):
        raise ValueError("invalid layer selection")
    B=images.shape[0]
    x=model.patch_embed(images)
    cls=model.cls_token.expand(B,-1,-1)
    x=torch.cat([cls,x],dim=1)
    x=model.pos_encoding(x)
    out={}
    max_layer=wanted[-1]
    want=set(wanted)
    for i,block in enumerate(model.blocks):
        if i>max_layer:
            break
        x,z=_manual_attention_block_with_logits(block,x)
        if i in want:
            out[i]=z
    if set(out)!=want:
        raise RuntimeError("canonical logits path failed to return requested layers")
    return out

@torch.no_grad()
def assert_grad_logits_path_exact(model, images, layers_0based: Sequence[int]) -> Dict[str,float]:
    """Fail closed unless differentiable logits equal canonical measurement logits bitwise."""
    model.eval()
    _,_,z_ref=model.forward_with_attention_logits(images)
    z_new=canonical_attention_logits_grad(model,images,layers_0based)
    diffs={}
    for i in layers_0based:
        if not torch.equal(z_new[int(i)],z_ref[int(i)]):
            d=float((z_new[int(i)]-z_ref[int(i)]).abs().max().item())
            raise RuntimeError(f"Grad-logits path drift at layer {int(i)+1}; max_abs={d}")
        diffs[str(int(i)+1)]=0.0
    return diffs

@torch.no_grad()
def build_clean_logits_cache(clean_model, measurement_loader, device: str) -> List[Tuple[torch.Tensor,...]]:
    """Cache exact clean canonical measurement logits for all 12 layers on GPU."""
    clean_model.eval()
    cache=[]
    total=0
    for images,_ in measurement_loader:
        images=images.to(device,non_blocking=True)
        _,_,z=clean_model.forward_with_attention_logits(images)
        if len(z)!=12:
            raise RuntimeError("expected 12 layers")
        cache.append(tuple(x.detach().clone() for x in z))
        total += int(images.shape[0])
    if total!=len(measurement_loader.dataset):
        raise RuntimeError("clean logits cache image count mismatch")
    return cache

def _profile_lse(layer_means: torch.Tensor, temperature: float) -> torch.Tensor:
    t=float(temperature)
    return t*torch.logsumexp(layer_means/t,dim=0)

@torch.no_grad()
def target_value_no_grad(model, clean_cache, measurement_loader, device: str,
                         objective_kind: str, lse_temperature: float=0.01):
    model.eval()
    layers=(3,) if objective_kind=="l4" else tuple(range(12))
    sums=torch.zeros(len(layers),device=device,dtype=torch.float64)
    total=0
    for bi,(images,_) in enumerate(measurement_loader):
        images=images.to(device,non_blocking=True)
        zt=canonical_attention_logits_grad(model,images,layers)
        for j,L in enumerate(layers):
            sums[j] += canonical_per_image(clean_cache[bi][L],zt[L]).sum()
        total += int(images.shape[0])
    if total!=len(measurement_loader.dataset):
        raise RuntimeError("target measurement count mismatch")
    means=sums/total
    if objective_kind=="l4":
        target=means[0]
    elif objective_kind=="profile_lse":
        target=_profile_lse(means,lse_temperature)
    else:
        raise ValueError(objective_kind)
    return float(target.item()), [float(x) for x in means.detach().cpu().tolist()]

def adaptive_projected_ascent(
    clean_model,
    pe_type: str,
    epsilon: float,
    lam: float,
    objective_kind: str,
    attack_ref_loader,
    measurement_ref_loader,
    clean_logits_cache,
    device: str,
    delta_save_path: str,
    protocol_version: str,
    execution_implementation: str,
    steps: int=20,
    alpha_ratio: float=0.1,
    lse_temperature: float=0.01,
):
    """Projected ascent on standard CE - lambda*canonical ADS target.

    CE and ADS gradients are accumulated from separate exact forward paths.
    """
    if lam<=0:
        raise ValueError("adaptive_projected_ascent is for positive lambda only")
    if objective_kind not in {"l4","profile_lse"}:
        raise ValueError(objective_kind)
    import copy
    pm=copy.deepcopy(clean_model).to(device)
    pm.eval()
    groups=collect_pe_groups(pm,pe_type,device,delta_convention="per_buffer")
    bounds={g.name:float(epsilon) for g in groups}
    N=len(attack_ref_loader.dataset)
    if N!=256 or len(measurement_ref_loader.dataset)!=256:
        raise RuntimeError("LOCK-003 requires exactly 256 reference images")
    ce_sum_loss=torch.nn.CrossEntropyLoss(reduction="sum")
    trajectory=[]

    for step in range(1,steps+1):
        _apply_groups(groups)
        pm.zero_grad(set_to_none=True)

        # 1) exact standard model CE gradient, same objective semantics as EXP-014.
        ce_scalar=0.0
        seen=0
        for images,labels in attack_ref_loader:
            images=images.to(device,non_blocking=True)
            labels=labels.to(device,non_blocking=True)
            logits=pm(images)
            ls=ce_sum_loss(logits,labels)
            (ls/N).backward()
            ce_scalar += float(ls.detach().item())/N
            seen += int(labels.numel())
        if seen!=N:
            raise RuntimeError(f"CE saw {seen}, expected {N}")

        # 2) canonical ADS gradient on the exact canonical measurement path.
        if objective_kind=="l4":
            ads_sum_value=0.0
            seen2=0
            for bi,(images,_) in enumerate(measurement_ref_loader):
                images=images.to(device,non_blocking=True)
                zt=canonical_attention_logits_grad(pm,images,(3,))
                v=canonical_per_image(clean_logits_cache[bi][3],zt[3])
                (-float(lam)*(v.sum()/N)).backward()
                ads_sum_value += float(v.detach().sum().item())
                seen2 += int(images.shape[0])
            if seen2!=N:
                raise RuntimeError("ADS L4 count mismatch")
            target=float(ads_sum_value/N)
            layer_means=[target]
            lse_weights=None
        else:
            # Exact memory-bounded gradient of LSE(mean ADS per layer):
            # first get current layer means, then use dLSE/dADS_l softmax weights.
            with torch.no_grad():
                sums=torch.zeros(12,device=device,dtype=torch.float64)
                seen2=0
                for bi,(images,_) in enumerate(measurement_ref_loader):
                    images=images.to(device,non_blocking=True)
                    zt=canonical_attention_logits_grad(pm,images,tuple(range(12)))
                    for L in range(12):
                        sums[L]+=canonical_per_image(clean_logits_cache[bi][L],zt[L]).sum()
                    seen2 += int(images.shape[0])
                if seen2!=N:
                    raise RuntimeError("profile first-pass count mismatch")
                means=sums/N
                weights=torch.softmax(means/float(lse_temperature),dim=0).detach()
                target_tensor=_profile_lse(means,lse_temperature)
                target=float(target_tensor.item())
                layer_means=[float(x) for x in means.detach().cpu().tolist()]
                lse_weights=[float(x) for x in weights.detach().cpu().tolist()]

            seen3=0
            for bi,(images,_) in enumerate(measurement_ref_loader):
                images=images.to(device,non_blocking=True)
                zt=canonical_attention_logits_grad(pm,images,tuple(range(12)))
                batch_term=None
                for L in range(12):
                    v=canonical_per_image(clean_logits_cache[bi][L],zt[L])
                    term=weights[L]*(v.sum()/N)
                    batch_term=term if batch_term is None else batch_term+term
                (-float(lam)*batch_term).backward()
                seen3 += int(images.shape[0])
            if seen3!=N:
                raise RuntimeError("profile gradient-pass count mismatch")

        # 3) fail-closed projected sign ascent.
        grad_linf={}
        with torch.no_grad():
            for g in groups:
                grad=_aggregate_group_grad(g)
                if grad is None:
                    raise RuntimeError(f"No gradient for attacked group {g.name} at step {step}")
                grad_linf[g.name]=float(grad.abs().max().item())
                bt=_bound_tensor_for_delta(bounds[g.name],g.delta)
                candidate=g.delta + (float(alpha_ratio)*bt)*grad.sign()
                g.delta=torch.maximum(torch.minimum(candidate,bt),-bt)

        trajectory.append({
            "step":step,
            "mean_ce":ce_scalar,
            "target_ads":target,
            "objective":ce_scalar-float(lam)*target,
            "layer_means_for_target":layer_means,
            "lse_weights":lse_weights,
            "gradient_linf_by_group":grad_linf,
        })

    _apply_groups(groups)
    pm.eval()
    per_geom=[attack_group_geometry(g,bounds[g.name]) for g in groups]
    max_delta=max((float(x["delta_linf"]) for x in per_geom),default=0.0)
    if max_delta > float(epsilon)+1e-7:
        raise RuntimeError(f"projection violation: {max_delta} > {epsilon}")
    meta={
        "objective_kind":objective_kind,
        "objective":"mean_CE_ref - lambda * canonical_ADS_target",
        "epsilon":float(epsilon),
        "lambda":float(lam),
        "steps":int(steps),
        "alpha_ratio":float(alpha_ratio),
        "model_mode":"eval",
        "n_ref_images":N,
        "delta_init":"zero",
        "constraint":"absolute_L_inf_per_buffer_group",
        "selected_state":"raw_final_step20",
        "group_count":len(groups),
        "attack_geometry":{"per_group":per_geom,"max_delta_linf":max_delta},
        "trajectory":trajectory,
    }
    ident=save_attack_delta(
        groups,delta_save_path,
        metadata={
            "pe_type":pe_type,"surface":"pe_only","epsilon":float(epsilon),
            "lambda":float(lam),"objective_kind":objective_kind,
            "budget_mode":"absolute","delta_convention":"per_buffer",
            "steps":int(steps),"alpha_ratio":float(alpha_ratio),
        },
        storage_dtype="float32",
        protocol_version=protocol_version,
        execution_implementation=execution_implementation,
    )
    meta["delta_artifact"]=ident
    return pm,meta
