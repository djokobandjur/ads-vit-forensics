#!/usr/bin/env python3
"""Canonical ADS numerical operator for the TIFS v2.1 canonical primary full-grid execution (scientific operator lock unchanged).

SCIENTIFIC LOCK
---------------
ADS = mean over reference images of per-image attention-row KL divergences.
The KL is computed directly from pre-softmax attention logits using float64
log-softmax arithmetic. There is NO additive probability epsilon/floor and NO
renormalization after a floor.

Canonical operator_spec_hash:
093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from ads_attack_engine_v2_1 import _kl_rows_typed, compute_ads_from_means, file_identity

CANONICAL_OPERATOR_HASH='093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362'


def canonical_operator_spec():
    # Must remain byte-for-byte semantically identical to the v1.4.1 lock spec
    # used to derive CANONICAL_OPERATOR_HASH.
    return {
        'name':'ADS_Definition_1_candidate_v1_4_1',
        'sample_aggregation':'mean_of_per_image_divergences',
        'per_image_reduction':'mean_over_heads_and_query_rows',
        'row_divergence':'KL(clean_attention_row || test_attention_row)',
        'input_to_divergence':'native_pre_softmax_attention_logits',
        'log_probability_computation':'torch.log_softmax(logits.float64, dim=-1)',
        'probability_computation':'exp(log_probability_float64)',
        'probability_floor':None,
        'renormalization_after_floor':False,
        'divergence_arithmetic':'float64',
        'model_attention_forward_dtype':'model_native',
        'attack_state':'reused_corrected_CE_only_delta',
        'measurement_execution':'GPU_resident_attention_logits_and_float64_KL; CPU_only_final_per_image_scalars',
    }


def operator_spec_hash(spec=None):
    if spec is None: spec=canonical_operator_spec()
    raw=json.dumps(spec,sort_keys=True,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def assert_operator_lock():
    got=operator_spec_hash()
    if got != CANONICAL_OPERATOR_HASH:
        raise RuntimeError(f'Canonical operator hash drift: {got} != {CANONICAL_OPERATOR_HASH}')
    return got


def ads_summary(vals):
    vals=[float(x) for x in vals]
    return {
        'mean_ads':float(np.mean(vals)),
        'mid_layer_ads':float(np.mean(vals[4:9])),
        'layer4_ads':float(vals[3]),
        'per_layer_ads':vals,
        'argmax_layer_1based':int(np.argmax(vals)+1),
        'l12_over_l1':float(vals[11]/vals[0]) if vals[0]>0 else None,
    }


def save_per_image_npz(path, canonical_matrix, compat_matrix):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(p)+'.tmp.npz')
    np.savez_compressed(
        tmp,
        per_image_ads_canonical_logit_domain=np.asarray(canonical_matrix,dtype=np.float32),
        per_image_ads_prob_eps1e12=np.asarray(compat_matrix,dtype=np.float32),
        layer_indices=np.arange(canonical_matrix.shape[1],dtype=np.int16),
        operator_spec_hash=np.asarray([CANONICAL_OPERATOR_HASH]),
    )
    os.replace(tmp,p)
    return file_identity(str(p))


@torch.no_grad()
def measure_canonical_ads(clean_model, test_model, loader, device,
                          per_image_path=None, progress_label='ADS', progress_every=4,
                          retain_compat=True, retain_legacy=True):
    """Measure canonical ADS plus optional provenance-only compatibility fields."""
    assert_operator_lock()
    clean_model.eval(); test_model.eval()
    canonical_chunks={i:[] for i in range(12)}
    compat_chunks={i:[] for i in range(12)}
    clean_sums={}; test_sums={}
    zero_clean=torch.zeros(12,dtype=torch.int64,device=device)
    zero_test=torch.zeros(12,dtype=torch.int64,device=device)
    zero_mass_clean=torch.zeros(12,dtype=torch.float64,device=device)
    zero_mass_test=torch.zeros(12,dtype=torch.float64,device=device)
    n_entries=[0]*12
    total=0; n_batches=len(loader); t0=time.time()
    for bi,(images,_) in enumerate(loader,1):
        images=images.to(device,non_blocking=True)
        _,pc,zc=clean_model.forward_with_attention_logits(images)
        _,pt,zt=test_model.forward_with_attention_logits(images)
        if not (len(pc)==len(pt)==len(zc)==len(zt)==12):
            raise RuntimeError('Expected 12 attention layers')
        for i,(p,q,zp,zq) in enumerate(zip(pc,pt,zc,zt)):
            lp=F.log_softmax(zp.to(torch.float64),dim=-1)
            lq=F.log_softmax(zq.to(torch.float64),dim=-1)
            pd=lp.exp(); qd=lq.exp()
            row_kl=(pd*(lp-lq)).sum(dim=-1)
            if torch.any(row_kl < -1e-11):
                raise RuntimeError(f'Negative KL below tolerance at layer {i+1}: {float(row_kl.min())}')
            v=row_kl.clamp_min(0.0).mean(dim=(1,2))
            canonical_chunks[i].append(v.detach().cpu().numpy().astype(np.float64,copy=False))
            if retain_compat:
                cv=_kl_rows_typed(p,q,eps=1e-12,dtype=torch.float64).mean(dim=(1,2))
                compat_chunks[i].append(cv.detach().cpu().numpy().astype(np.float64,copy=False))
            if retain_legacy:
                ps=p.sum(dim=0).float(); qs=q.sum(dim=0).float()
                clean_sums[i]=ps if i not in clean_sums else clean_sums[i]+ps
                test_sums[i]=qs if i not in test_sums else test_sums[i]+qs
            p0=(p==0); q0=(q==0)
            zero_clean[i]+=p0.sum(); zero_test[i]+=q0.sum(); n_entries[i]+=int(p.numel())
            zero_mass_clean[i]+=pd.masked_select(p0).sum(); zero_mass_test[i]+=qd.masked_select(q0).sum()
            del lp,lq,pd,qd,row_kl,v,p0,q0
        total += int(images.size(0))
        del pc,pt,zc,zt,images
        if device=='cuda': torch.cuda.synchronize()
        if bi==1 or bi==n_batches or (progress_every and bi%progress_every==0):
            print(f'  [{progress_label}] ref batch {bi}/{n_batches} images={total} elapsed={time.time()-t0:.1f}s',flush=True)
    if total==0: raise RuntimeError('Empty reference loader')
    canonical_matrix=np.stack([np.concatenate(canonical_chunks[i]) for i in range(12)],axis=1)
    if canonical_matrix.shape!=(total,12): raise RuntimeError(f'Bad canonical matrix shape {canonical_matrix.shape}')
    canonical=ads_summary(canonical_matrix.mean(axis=0,dtype=np.float64))
    canonical.update({'operator':'mean(per-image KL) from pre-softmax logits','divergence_arithmetic':'float64','probability_floor':None,'operator_spec_hash':CANONICAL_OPERATOR_HASH,'n_images':total})

    compat=None; comparison=None
    if retain_compat:
        compat_matrix=np.stack([np.concatenate(compat_chunks[i]) for i in range(12)],axis=1)
        compat=ads_summary(compat_matrix.mean(axis=0,dtype=np.float64))
        compat.update({'operator':'mean(per-image KL) from native probabilities','epsilon_s':1e-12,'divergence_arithmetic':'float64','n_images':total})
        a=np.asarray(canonical['per_layer_ads']); b=np.asarray(compat['per_layer_ads'])
        rel=np.abs(a-b)/np.maximum(np.abs(a),1e-300)
        comparison={
            'max_abs_layer_difference':float(np.max(np.abs(a-b))),
            'max_relative_layer_difference':float(np.max(rel)),
            'mean_relative_layer_difference':float(np.mean(rel)),
            'l4_abs_difference':float(abs(canonical['layer4_ads']-compat['layer4_ads'])),
            'l4_relative_difference':float(abs(canonical['layer4_ads']-compat['layer4_ads'])/max(abs(canonical['layer4_ads']),1e-300)),
            'argmax_layer_same':bool(canonical['argmax_layer_1based']==compat['argmax_layer_1based']),
        }
    else:
        compat_matrix=np.empty((total,0),dtype=np.float64)

    legacy=None
    if retain_legacy:
        cm=[(clean_sums[i]/total).detach().cpu() for i in range(12)]
        tm=[(test_sums[i]/total).detach().cpu() for i in range(12)]
        legacy=compute_ads_from_means(cm,tm)
        legacy['operator']='KL(mean attention || mean attention)'; legacy['arithmetic']='float32'; legacy['n_images']=total
        legacy['argmax_layer_1based']=int(np.argmax(legacy['per_layer_ads'])+1)
        legacy['l12_over_l1']=float(legacy['per_layer_ads'][11]/legacy['per_layer_ads'][0]) if legacy['per_layer_ads'][0]>0 else None

    artifact=None
    if per_image_path:
        artifact=save_per_image_npz(per_image_path,canonical_matrix,compat_matrix if retain_compat else np.zeros_like(canonical_matrix))

    dist={
        'median_per_layer':np.median(canonical_matrix,axis=0).astype(float).tolist(),
        'q25_per_layer':np.quantile(canonical_matrix,.25,axis=0).astype(float).tolist(),
        'q75_per_layer':np.quantile(canonical_matrix,.75,axis=0).astype(float).tolist(),
        'max_per_layer':canonical_matrix.max(axis=0).astype(float).tolist(),
    }
    zero_diag={}
    for i in range(12):
        n=max(n_entries[i],1); cc=int(zero_clean[i].item()); tc=int(zero_test[i].item())
        zero_diag[str(i+1)]={
            'clean_zero_count':cc,'test_zero_count':tc,'n_probability_entries':int(n_entries[i]),
            'clean_float64_mass_at_native_zero':float(zero_mass_clean[i].item()),
            'test_float64_mass_at_native_zero':float(zero_mass_test[i].item()),
            'clean_zero_fraction':cc/n,'test_zero_fraction':tc/n,
        }
    return {
        'canonical':canonical,
        'compat_prob_eps1e12':compat,
        'legacy_refprofile':legacy,
        'canonical_vs_eps1e12':comparison,
        'per_image_distribution':dist,
        'native_probability_zero_diagnostics_per_layer':zero_diag,
        'per_image_artifact':artifact,
    }
