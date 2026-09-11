#!/usr/bin/env python3
"""
Stage-B saved-delta reevaluation — execution correction v2.8.1.

Scientific parent lock (UNCHANGED):
  ADS_TIFS_POST_AUDIT_CORRECTION_PROTOCOL_LOCK_v2_8_20260911

NO ATTACK OPTIMIZATION.
B1: exact clean/attacked accuracy on 4,744 non-reference ImageNet-100 validation images.
B2: exact saved-delta four-score benign-aware comparison on the disjoint 256-image holdout.

v2.8.1 is an execution/provenance correction of the v2.8 package. It fixes the
canonical transformed-cache SHA semantics (image tensor bytes only, never labels),
materializes the locked 5,000-image transformed validation cache once on GPU, derives
reference/holdout/non-reference views from that same cache, materializes each benign
holdout transform exactly once, records transformed-stream SHAs, computes all locked
secondary comparison diagnostics, verifies all 228 source deltas/cell identities, and
emits a fail-closed result manifest/audit bundle.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, platform, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import PIL
from PIL import Image, ImageFilter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from full_scale_experiment_v1_6 import VisionTransformer
from ads_attack_engine_v2_1 import apply_saved_delta, sha256_file, configure_deterministic_runtime
from ads_canonical_operator_v2_1 import assert_operator_lock, CANONICAL_OPERATOR_HASH

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
EXECUTION_ID='ADS_TIFS_STAGE_B_SAVED_DELTA_REEVALUATION_v2_8_2_20260911'
PARENT_PROTOCOL_ID='ADS_TIFS_POST_AUDIT_CORRECTION_PROTOCOL_LOCK_v2_8_20260911'
PARENT_LOCK_SHA='3ffb2ad0f5217d7a907590d8395150fe5b97846118a4e70d0fc6fca10fa2ba6e'
FULL_GRID_LOCK_SHA='2b242559d0ec523683ee05a33238c819790d17ebef919745cab3b426f161021f'
SOURCE_AGG_SHA='dcd2fa22d27a9a983dfadca3d49b56f776d316add01d201a2fe5563d757203d1'
REF_INDEX_SHA='1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509'
HOLDOUT_INDEX_SHA='835af1a98d05ac4f99cbb697f36d456b0a24bdd6905ada0ebac5da4eec0916a7'
REF_STREAM_SHA='01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4'
FULL_STREAM_SHA='90d47ea5212f5d06b9e8d12fa80ccb20cc722a46d79d2c926ef41be2917f0f71'
PES=['learned','sinusoidal','rope','alibi']
SEEDS=[42,123,456,789,1011,1213]
METHODS=['ADS_L4','Attn_L2','Diag_Mahalanobis','LogitKL']
BENIGN_SPECS=[('jpeg_q50','jpeg',50),('jpeg_q30','jpeg',30),('jpeg_q10','jpeg',10),
              ('blur_s1','blur',1),('blur_s2','blur',2),('blur_s3','blur',3),
              ('noise_005','noise',.05),('noise_010','noise',.10),('noise_020','noise',.20)]
LAYER=3
MEAN=torch.tensor([0.485,0.456,0.406],dtype=torch.float32).view(3,1,1)
STD=torch.tensor([0.229,0.224,0.225],dtype=torch.float32).view(3,1,1)
JPEG_OPTIONS={'subsampling':2,'optimize':False,'progressive':False}
IDENTITY_TOL=1e-12
FULL_ACC_TOL=1e-12


def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--models_dir',required=True)
    p.add_argument('--val_dir',required=True)
    p.add_argument('--full_grid_root',required=True,help='Extracted EXP-014 RESULTS root containing deltas/')
    p.add_argument('--source_aggregate',required=True)
    p.add_argument('--reference_indices',required=True)
    p.add_argument('--holdout_indices',required=True)
    p.add_argument('--parent_lock_json',required=True)
    p.add_argument('--output_root',required=True)
    p.add_argument('--measurement_batch_size',type=int,default=8)
    p.add_argument('--eval_batch_size',type=int,default=64)
    p.add_argument('--cache_load_batch_size',type=int,default=128)
    p.add_argument('--num_workers',type=int,default=0)
    return p.parse_args()


def hfile(p): return sha256_file(str(p))

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def atomic_json(p,o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=Path(str(p)+'.tmp')
    t.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n'); os.replace(t,p)

def atomic_npz(p,**kw):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=Path(str(p)+'.tmp.npz')
    np.savez_compressed(t,**kw); os.replace(t,p)

def atomic_csv(p,df):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=Path(str(p)+'.tmp')
    df.to_csv(t,index=False); os.replace(t,p)

def eps_tag(x): return f'{x:.6g}'.replace('.','p')

def ensure_fresh_output(out: Path):
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'output_root is not empty; refusing to overwrite verified/historical artifacts: {out}')
    out.mkdir(parents=True,exist_ok=True)


def stable_noise_seed(global_index,sigma):
    text=f'ADS_TIFS_BENIGN_NOISE_v2_3|20260910|{int(global_index)}|{sigma:.2f}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')%(2**63-1)

def denorm(x): return (x.cpu()*STD+MEAN).clamp(0,1)
def renorm(x): return (x-MEAN)/STD

def jpeg(x,q):
    p=TF.to_pil_image(denorm(x))
    b=io.BytesIO()
    # Explicitly freeze the encoder options that Pillow's RGB JPEG path otherwise
    # supplies as hidden defaults. subsampling=2 is 4:2:0 and reproduces the
    # previous default behavior while making it provenance-visible.
    p.save(b,format='JPEG',quality=int(q),subsampling=JPEG_OPTIONS['subsampling'],
           optimize=JPEG_OPTIONS['optimize'],progressive=JPEG_OPTIONS['progressive'])
    b.seek(0)
    with Image.open(b) as im:
        y=TF.to_tensor(im.convert('RGB'))
    return renorm(y)

def blur(x,r):
    p=TF.to_pil_image(denorm(x)).convert('RGB')
    return renorm(TF.to_tensor(p.filter(ImageFilter.GaussianBlur(radius=float(r)))))

def noise(x,sigma,idx):
    p=denorm(x)
    rng=np.random.Generator(np.random.PCG64(stable_noise_seed(idx,sigma)))
    n=torch.from_numpy(rng.standard_normal(tuple(p.shape),dtype=np.float32))*float(sigma)
    return renorm((p+n).clamp(0,1))


def load_dataset(val_dir):
    tf=transforms.Compose([
        transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize(MEAN.flatten().tolist(),STD.flatten().tolist())])
    return datasets.ImageFolder(val_dir,tf)


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().numpy().tobytes(order='C')


def canonical_xy_stream_sha_gpu(X: torch.Tensor, Y: torch.Tensor, batch_size: int) -> str:
    """Reproduce the EXP-014 canonical transformed-cache identity exactly.

    The byte stream is partition-sensitive because each batch contributes
    X_batch bytes followed immediately by Y_batch bytes.  Therefore the
    canonical batch sizes are part of the identity: 128 for the full 5000-image
    validation materialization and 32 for the 256-image reference view.
    """
    if len(X)!=len(Y): raise RuntimeError('canonical stream X/Y length mismatch')
    h=hashlib.sha256()
    for s in range(0,len(Y),batch_size):
        e=min(s+batch_size,len(Y))
        h.update(_tensor_bytes(X[s:e])); h.update(_tensor_bytes(Y[s:e]))
    return h.hexdigest()


def materialize_full_gpu_cache(ds,batch_size=128):
    """Decode/transform once and reproduce EXP-014's exact full-cache SHA semantics."""
    if int(batch_size)!=128:
        raise RuntimeError(f'cache_load_batch_size=128 is locked by EXP-014 cache identity, got {batch_size}')
    n=len(ds)
    X=torch.empty((n,3,224,224),dtype=torch.float32,device=DEVICE)
    Y=torch.empty((n,),dtype=torch.long,device=DEVICE)
    dl=DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,pin_memory=True)
    h=hashlib.sha256(); p=0; t0=time.time()
    for bi,(x,y) in enumerate(dl,1):
        if x.dtype!=torch.float32 or x.ndim!=4 or tuple(x.shape[1:])!=(3,224,224):
            raise RuntimeError(f'unexpected transformed batch layout {x.dtype} {tuple(x.shape)}')
        if y.dtype!=torch.long or y.ndim!=1:
            raise RuntimeError(f'unexpected label batch layout {y.dtype} {tuple(y.shape)}')
        # EXACT EXP-014 semantics (ads_primary_full_grid_v2_1.py):
        # for every 128-image source batch hash X bytes, then int64 Y bytes.
        h.update(x.contiguous().numpy().tobytes(order='C'))
        h.update(y.contiguous().numpy().tobytes(order='C'))
        e=p+len(y)
        X[p:e].copy_(x,non_blocking=True); Y[p:e].copy_(y,non_blocking=True); p=e
        if bi==1 or bi==len(dl) or bi%10==0:
            print(f'[cache] batch {bi}/{len(dl)} images={p}/{n} elapsed={time.time()-t0:.1f}s',flush=True)
    if p!=n: raise RuntimeError(f'full cache cardinality mismatch {p} != {n}')
    torch.cuda.synchronize()
    return X,Y,h.hexdigest(),time.time()-t0


def image_tensor_sha_gpu(X: torch.Tensor, batch_size=64) -> str:
    """Image-only SHA for new Stage-B transformed-condition provenance (not a canonical EXP-014 cache lock)."""
    h=hashlib.sha256()
    for s in range(0,len(X),batch_size):
        h.update(_tensor_bytes(X[s:s+batch_size]))
    return h.hexdigest()


def build_benign_holdout_cache(Xh,holdout_indices):
    """Build each deterministic benign condition once, cache on GPU, and hash its tensor stream."""
    Xhc=Xh.detach().cpu()
    cache={}
    manifest={}
    for j,(name,kind,val) in enumerate(BENIGN_SPECS,1):
        print(f'[benign-cache] {j}/{len(BENIGN_SPECS)} {name}',flush=True)
        T=torch.empty_like(Xhc)
        for i,gidx in enumerate(holdout_indices):
            if kind=='jpeg': q=jpeg(Xhc[i],val)
            elif kind=='blur': q=blur(Xhc[i],val)
            else: q=noise(Xhc[i],val,gidx)
            T[i]=q
        sh=hashlib.sha256(T.contiguous().numpy().tobytes(order='C')).hexdigest()
        cache[name]=T.to(DEVICE,non_blocking=False)
        manifest[name]={
            'kind':kind,'value':float(val) if kind!='jpeg' else int(val),
            'n_images':256,'shape':[256,3,224,224],'dtype':'float32',
            'transformed_tensor_stream_sha256':sh,
            'noise_rng':'numpy.random.PCG64' if kind=='noise' else None,
            'jpeg_options':dict(JPEG_OPTIONS) if kind=='jpeg' else None,
        }
    torch.cuda.synchronize()
    return cache,manifest


def load_model(models_dir,pe,seed,expected_sha):
    p=Path(models_dir)/f'{pe}_seed{seed}'/'best_model.pth'
    if not p.exists(): raise RuntimeError(f'missing checkpoint {p}')
    got=hfile(p)
    if got!=expected_sha: raise RuntimeError(f'checkpoint SHA mismatch {p}: {got} != {expected_sha}')
    m=VisionTransformer(img_size=224,patch_size=16,num_classes=100,embed_dim=768,depth=12,
                        num_heads=12,mlp_ratio=4.0,dropout=0.1,pe_type=pe)
    st=torch.load(p,map_location='cpu',weights_only=False)
    if isinstance(st,dict) and 'model_state_dict' in st: st=st['model_state_dict']
    m.load_state_dict({k.replace('_orig_mod.',''):v for k,v in st.items()},strict=True)
    return m.eval().to(DEVICE), {'path':str(p),'sha256':got,'bytes':p.stat().st_size}


@torch.no_grad()
def accuracy_cached(model,X,Y,indices,batch_size):
    idx=torch.as_tensor(indices,dtype=torch.long,device=DEVICE)
    corr=[]; model.eval()
    for s in range(0,len(idx),batch_size):
        ii=idx[s:s+batch_size]
        pred=model(X[ii]).argmax(1)
        corr.append((pred==Y[ii]).detach().cpu().numpy().astype(np.uint8))
    a=np.concatenate(corr) if corr else np.empty((0,),dtype=np.uint8)
    return float(a.mean()*100),a


@torch.no_grad()
def clean_native(model,X,bs):
    Z=[]; C=[]
    for s in range(0,len(X),bs):
        cls,_,zz=model.forward_with_attention_logits(X[s:s+bs])
        Z.append(zz[LAYER].detach()); C.append(cls.detach())
    return torch.cat(Z),torch.cat(C)


def key_feature(z):
    lp=F.log_softmax(z.to(torch.float64),dim=-1)
    return lp.exp().mean(dim=-2).reshape(z.shape[0],-1)


def variance_from_reference(clean_ref_z):
    f=key_feature(clean_ref_z)
    var=f.var(dim=0,unbiased=True)
    lam=1e-6*var.mean()+1e-18
    if not torch.isfinite(var).all() or not torch.isfinite(lam):
        raise RuntimeError('non-finite diagonal-Mahalanobis calibration')
    return var,lam


def paired(clean_z,test_z,clean_cls,test_cls,var,lam):
    lp=F.log_softmax(clean_z.to(torch.float64),dim=-1)
    lq=F.log_softmax(test_z.to(torch.float64),dim=-1)
    p=lp.exp(); q=lq.exp()
    row_kl=(p*(lp-lq)).sum(-1)
    if torch.any(row_kl < -1e-11):
        raise RuntimeError(f'negative ADS row KL below tolerance: {float(row_kl.min())}')
    ads=row_kl.clamp_min(0).mean((1,2))
    d=q.mean(-2).reshape(q.shape[0],-1)-p.mean(-2).reshape(p.shape[0],-1)
    l2=torch.linalg.vector_norm(d,dim=1)
    dm=torch.sqrt(torch.sum(d*d/(var+lam),dim=1))
    clp=F.log_softmax(clean_cls.to(torch.float64),dim=-1)
    tlp=F.log_softmax(test_cls.to(torch.float64),dim=-1)
    row_lkl=(clp.exp()*(clp-tlp)).sum(-1)
    if torch.any(row_lkl < -1e-11):
        raise RuntimeError(f'negative LogitKL below tolerance: {float(row_lkl.min())}')
    lkl=row_lkl.clamp_min(0)
    vals=[ads,l2,dm,lkl]
    out=[x.detach().cpu().numpy().astype(np.float64) for x in vals]
    if not all(np.isfinite(x).all() for x in out): raise RuntimeError('non-finite score')
    return out


@torch.no_grad()
def score(model,X,clean_z,clean_cls,var,lam,bs):
    out={m:[] for m in METHODS}
    for s in range(0,len(X),bs):
        cls,_,zz=model.forward_with_attention_logits(X[s:s+bs])
        vals=paired(clean_z[s:s+bs],zz[LAYER],clean_cls[s:s+bs],cls,var,lam)
        for m,v in zip(METHODS,vals): out[m].append(v)
    ans={m:np.concatenate(v) for m,v in out.items()}
    if any(x.shape!=(256,) for x in ans.values()):
        raise RuntimeError(f'bad score shape: { {m:x.shape for m,x in ans.items()} }')
    return ans


def rank_auc(p,n):
    p=np.asarray(p,float); n=np.asarray(n,float)
    if p.ndim!=1 or n.ndim!=1 or len(p)==0 or len(n)==0: raise ValueError('AUC arrays must be nonempty 1D')
    if not np.isfinite(p).all() or not np.isfinite(n).all(): raise RuntimeError('non-finite AUC input')
    ns=np.sort(n)
    l=np.searchsorted(ns,p,'left'); r=np.searchsorted(ns,p,'right')
    return float(np.mean((l+.5*(r-l))/len(n)))


def paired_benign_concordance(pos,benign):
    pos=np.asarray(pos,float).reshape(-1,1); benign=np.asarray(benign,float)
    if benign.shape!=(256,9) or pos.shape!=(256,1): raise RuntimeError('bad paired concordance shapes')
    return float(np.mean((pos>benign).astype(np.float64)+0.5*(pos==benign).astype(np.float64)))


def worst_benign_exceedance_rate(pos,benign):
    pos=np.asarray(pos,float); benign=np.asarray(benign,float)
    return float(np.mean(pos>np.max(benign,axis=1)))


def verify_source_aggregate(agg):
    md=agg.get('metadata',{})
    if md.get('expected_cells')!=228 or md.get('completed_cells')!=228:
        raise RuntimeError('source aggregate cell-count metadata mismatch')
    if md.get('operator_spec_hash')!=CANONICAL_OPERATOR_HASH:
        raise RuntimeError('source aggregate operator mismatch')
    if md.get('protocol_lock',{}).get('sha256')!=FULL_GRID_LOCK_SHA:
        raise RuntimeError('source aggregate full-grid lock mismatch')
    if md.get('reference_gpu_cache',{}).get('transformed_tensor_stream_sha256')!=REF_STREAM_SHA:
        raise RuntimeError('source aggregate reference-cache mismatch')
    if md.get('full_validation_gpu_cache',{}).get('transformed_full_validation_stream_sha256')!=FULL_STREAM_SHA:
        raise RuntimeError('source aggregate full-validation-cache mismatch')


def source_map(agg,root):
    verify_source_aggregate(agg)
    M={}; verification=[]
    for c in agg['cells']:
        md=c['metadata']; pe=md['pe_type']; seed=int(md['seed']); eps=float(c['epsilon'])
        if pe not in PES or seed not in SEEDS: raise RuntimeError(f'unexpected source identity {(pe,seed,eps)}')
        if md.get('operator_spec_hash')!=CANONICAL_OPERATOR_HASH: raise RuntimeError(f'cell operator mismatch {(pe,seed,eps)}')
        if md.get('protocol_lock',{}).get('sha256')!=FULL_GRID_LOCK_SHA: raise RuntimeError(f'cell protocol mismatch {(pe,seed,eps)}')
        if md.get('reference_indices',{}).get('sha256')!=REF_INDEX_SHA: raise RuntimeError(f'cell reference index mismatch {(pe,seed,eps)}')
        if md.get('reference_gpu_cache',{}).get('transformed_tensor_stream_sha256')!=REF_STREAM_SHA: raise RuntimeError(f'cell reference cache mismatch {(pe,seed,eps)}')
        if md.get('full_validation_gpu_cache',{}).get('transformed_full_validation_stream_sha256')!=FULL_STREAM_SHA: raise RuntimeError(f'cell full cache mismatch {(pe,seed,eps)}')
        di=c['delta_artifact']; dp=Path(root)/'deltas'/Path(di['path']).name
        got_delta=hfile(dp) if dp.exists() else None
        ok=bool(dp.exists() and got_delta==di['sha256'])
        if not ok: raise RuntimeError(f'delta failure {(pe,seed,eps)} path={dp} got={got_delta} expected={di["sha256"]}')
        key=(pe,seed,eps)
        if key in M: raise RuntimeError(f'duplicate source cell {key}')
        M[key]={
            'delta':dp,'delta_sha':di['sha256'],'ck_sha':md['checkpoint']['sha256'],
            'clean_full_acc':float(c['clean_full_validation_accuracy']),
            'full_damage':float(c['accuracy_drop_pp']),'full_acc':float(c['full_validation_accuracy'])}
        verification.append({
            'pe':pe,'seed':seed,'epsilon':eps,'checkpoint_sha256':md['checkpoint']['sha256'],
            'delta_path':str(dp),'expected_delta_sha256':di['sha256'],'observed_delta_sha256':got_delta,
            'reference_index_sha256':REF_INDEX_SHA,'reference_cache_sha256':REF_STREAM_SHA,
            'full_validation_cache_sha256':FULL_STREAM_SHA,'operator_spec_hash':CANONICAL_OPERATOR_HASH,
            'protocol_lock_sha256':FULL_GRID_LOCK_SHA,'verified':True})
    if len(M)!=228: raise RuntimeError(f'expected 228 source cells, got {len(M)}')
    expected={(pe,seed) for pe in PES for seed in SEEDS}
    if {(p,s) for p,s,e in M}!=expected: raise RuntimeError('source PE/seed coverage mismatch')
    return M,verification


def implementation_manifest(parent_lock_json, source_aggregate, reference_indices, holdout_indices):
    paths=[HERE/'stage_b_saved_delta_reeval_v2_8_2.py',HERE/'ads_attack_engine_v2_1.py',
           HERE/'ads_canonical_operator_v2_1.py',HERE/'full_scale_experiment_v1_6.py',
           ROOT/'protocol/EXECUTION_CORRECTION_v2_8_2.json',ROOT/'protocol/CACHE_HASH_PROVENANCE.md',Path(parent_lock_json),Path(source_aggregate),
           Path(reference_indices),Path(holdout_indices)]
    out={}
    for p in paths:
        if not p.exists(): raise RuntimeError(f'missing implementation/provenance file {p}')
        out[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]={
            'bytes':p.stat().st_size,'sha256':hfile(p)}
    return out


def build_output_manifest(out: Path):
    entries=[]
    excluded={'SHA256_RESULTS_MANIFEST.json'}
    for p in sorted(x for x in out.rglob('*') if x.is_file()):
        rel=str(p.relative_to(out))
        if rel in excluded: continue
        entries.append({'path':rel,'bytes':p.stat().st_size,'sha256':hfile(p)})
    atomic_json(out/'SHA256_RESULTS_MANIFEST.json',{'execution_id':EXECUTION_ID,'n_payload_files':len(entries),'files':entries})
    return entries


def make_results_zip(out: Path):
    z=out.parent/f'{out.name}_RESULTS.zip'
    if z.exists(): raise RuntimeError(f'refusing to overwrite existing results zip {z}')
    with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as Z:
        for p in sorted(x for x in out.rglob('*') if x.is_file()):
            Z.write(p,arcname=f'{out.name}/{p.relative_to(out)}')
    zh=hfile(z)
    side=Path(str(z)+'.sha256')
    side.write_text(f'{zh}  {z.name}\n')
    return z,zh,side


def main():
    a=parse_args()
    if DEVICE!='cuda': raise RuntimeError('CUDA required')
    if a.num_workers!=0: raise RuntimeError('num_workers=0 required')
    out=Path(a.output_root)
    ensure_fresh_output(out)

    # Fail closed before expensive work.
    if hfile(a.parent_lock_json)!=PARENT_LOCK_SHA: raise RuntimeError('parent lock SHA mismatch')
    if hfile(a.source_aggregate)!=SOURCE_AGG_SHA: raise RuntimeError('source aggregate SHA mismatch')
    if hfile(a.reference_indices)!=REF_INDEX_SHA: raise RuntimeError('reference index SHA mismatch')
    if hfile(a.holdout_indices)!=HOLDOUT_INDEX_SHA: raise RuntimeError('holdout index SHA mismatch')
    if assert_operator_lock()!=CANONICAL_OPERATOR_HASH: raise RuntimeError('operator lock mismatch')
    runtime=configure_deterministic_runtime(seed=20260911)

    ref=json.load(open(a.reference_indices)); hold=json.load(open(a.holdout_indices))
    if len(ref)!=256 or len(hold)!=256 or len(set(ref))!=256 or len(set(hold))!=256 or set(ref)&set(hold):
        raise RuntimeError('index cardinality/uniqueness/overlap failure')
    if not all(isinstance(i,int) and 0<=i<5000 for i in ref+hold): raise RuntimeError('index range/type failure')
    ds=load_dataset(a.val_dir)
    if len(ds)!=5000: raise RuntimeError(f'expected 5000 validation images, got {len(ds)}')
    allidx=list(range(5000)); refset=set(ref); nonref=[i for i in allidx if i not in refset]
    if len(nonref)!=4744: raise RuntimeError('non-reference cardinality failure')

    # Single canonical transformed full-validation materialization.
    Xfull,Yfull,got_full,cache_seconds=materialize_full_gpu_cache(ds,a.cache_load_batch_size)
    if got_full!=FULL_STREAM_SHA:
        raise RuntimeError(f'full transformed canonical X/Y stream SHA mismatch {got_full} != {FULL_STREAM_SHA}')
    ref_idx_gpu=torch.tensor(ref,dtype=torch.long,device=DEVICE)
    hold_idx_gpu=torch.tensor(hold,dtype=torch.long,device=DEVICE)
    Xref=Xfull[ref_idx_gpu]; Yref=Yfull[ref_idx_gpu]
    Xh=Xfull[hold_idx_gpu]; Yh=Yfull[hold_idx_gpu]
    got_ref=canonical_xy_stream_sha_gpu(Xref,Yref,32)
    hold_canonical_stream=canonical_xy_stream_sha_gpu(Xh,Yh,32)
    hold_image_stream=image_tensor_sha_gpu(Xh,64)
    if got_ref!=REF_STREAM_SHA:
        raise RuntimeError(f'reference transformed canonical X/Y stream SHA mismatch {got_ref} != {REF_STREAM_SHA}')

    agg=json.load(open(a.source_aggregate)); src,verification=source_map(agg,a.full_grid_root)
    atomic_json(out/'SOURCE_DELTA_VERIFICATION.json',{'status':'PASS_228_OF_228','n_verified':len(verification),'rows':verification})
    atomic_csv(out/'SOURCE_DELTA_VERIFICATION.csv',pd.DataFrame(verification))

    # Build holdout benign transformed caches ONCE, independent of PE/model seed.
    benign_cache,benign_transform_manifest=build_benign_holdout_cache(Xh,hold)
    transform_meta={
        'identity':{'n_images':256,'shape':[256,3,224,224],'image_dtype':'float32','label_dtype':'int64','canonical_xy_stream_sha256_batch32':hold_canonical_stream,'image_tensor_stream_sha256':hold_image_stream},
        'conditions':benign_transform_manifest,
        'pillow_version':PIL.__version__,'torchvision_version':torchvision.__version__,'numpy_version':np.__version__,
        'jpeg_options':JPEG_OPTIONS,
        'noise_seed_rule':'int.from_bytes(SHA256(ADS_TIFS_BENIGN_NOISE_v2_3|20260910|<global_index>|<sigma:.2f>)[:8], little) % (2**63-1)',
        'noise_rng':'numpy.random.PCG64','normalization_mean':[0.485,0.456,0.406],'normalization_std':[0.229,0.224,0.225]}
    atomic_json(out/'BENIGN_HOLDOUT_TRANSFORM_MANIFEST.json',transform_meta)

    (out/'per_image').mkdir(exist_ok=True)
    (out/'calibration').mkdir(exist_ok=True)
    (out/'cells').mkdir(exist_ok=True)
    nonref_rows=[]; auc_rows=[]; cell_rows=[]; clean_rows=[]
    expected_cells=len(src); done=0

    for pe in PES:
      epslist=sorted({e for P,S,e in src if P==pe})
      for seed in SEEDS:
        print(f'\n[model] {pe} seed={seed} eps={epslist}',flush=True)
        s0=src[(pe,seed,epslist[0])]
        clean,ckprov=load_model(a.models_dir,pe,seed,s0['ck_sha'])

        # Exact cache/checkpoint sanity gate against EXP-014 clean full accuracy.
        clean_nonref_acc,clean_nonref_correct=accuracy_cached(clean,Xfull,Yfull,nonref,a.eval_batch_size)
        clean_ref_acc,clean_ref_correct=accuracy_cached(clean,Xfull,Yfull,ref,a.eval_batch_size)
        reconstructed_clean=(clean_nonref_correct.sum()+clean_ref_correct.sum())/5000.0*100.0
        if abs(reconstructed_clean-s0['clean_full_acc'])>FULL_ACC_TOL:
            raise RuntimeError(f'clean full-accuracy reconstruction mismatch {(pe,seed)} {reconstructed_clean} != {s0["clean_full_acc"]}')

        clean_ref_z,_=clean_native(clean,Xref,a.measurement_batch_size)
        var,lam=variance_from_reference(clean_ref_z)
        clean_h_z,clean_h_cls=clean_native(clean,Xh,a.measurement_batch_size)
        del clean_ref_z

        # Identity is measured rather than assumed; it must be zero to tolerance.
        identity=score(clean,Xh,clean_h_z,clean_h_cls,var,lam,a.measurement_batch_size)
        identity_max=max(float(np.max(np.abs(v))) for v in identity.values())
        if identity_max>IDENTITY_TOL:
            raise RuntimeError(f'identity displacement nonzero above tolerance {(pe,seed)} max={identity_max}')

        benign={m:np.zeros((256,9),dtype=np.float64) for m in METHODS}
        for j,(name,kind,val) in enumerate(BENIGN_SPECS):
            sc=score(clean,benign_cache[name],clean_h_z,clean_h_cls,var,lam,a.measurement_batch_size)
            for m in METHODS: benign[m][:,j]=sc[m]
        if any(not np.isfinite(x).all() for x in benign.values()): raise RuntimeError('non-finite benign scores')

        cal=out/'calibration'/f'{pe}_seed{seed}_holdout_benign_scores.npz'
        atomic_npz(cal,holdout_indices=np.array(hold,dtype=np.int64),
                   condition_names=np.array([x[0] for x in BENIGN_SPECS]),
                   diag_variance=var.detach().cpu().numpy().astype(np.float64),
                   diag_lambda=np.array([float(lam.item())],dtype=np.float64),
                   **{f'identity_{m}':identity[m].astype(np.float64) for m in METHODS},
                   **{f'benign_{m}':benign[m].astype(np.float64) for m in METHODS})
        cal_sha=hfile(cal)
        pneg={m:np.concatenate([identity[m],benign[m].reshape(-1)]) for m in METHODS}
        bneg={m:benign[m].reshape(-1) for m in METHODS}
        clean_rows.append({'pe':pe,'seed':seed,'checkpoint_sha256':ckprov['sha256'],
                           'clean_nonreference_accuracy':clean_nonref_acc,'clean_reference_accuracy':clean_ref_acc,
                           'reconstructed_clean_full_accuracy':reconstructed_clean,
                           'source_clean_full_accuracy':s0['clean_full_acc'],'identity_max_abs_score':identity_max,
                           'diag_lambda':float(lam.item()),'calibration_npz_sha256':cal_sha})

        for eps in epslist:
            done+=1; s=src[(pe,seed,eps)]
            print(f'[cell {done}/{expected_cells}] {pe} seed={seed} eps={eps:g}',flush=True)
            attacked,artifact=apply_saved_delta(clean,pe,'pe_only',str(s['delta']),DEVICE,
                                                delta_convention='per_buffer',include_biases=False)
            atk_nonref_acc,atk_nonref_correct=accuracy_cached(attacked,Xfull,Yfull,nonref,a.eval_batch_size)
            atk_ref_acc,atk_ref_correct=accuracy_cached(attacked,Xfull,Yfull,ref,a.eval_batch_size)
            reconstructed_full=(atk_nonref_correct.sum()+atk_ref_correct.sum())/5000.0*100.0
            full_discrepancy=reconstructed_full-s['full_acc']
            if abs(full_discrepancy)>FULL_ACC_TOL:
                raise RuntimeError(f'attacked full-accuracy reconstruction mismatch {(pe,seed,eps)} {reconstructed_full} != {s["full_acc"]}')
            nonref_drop=clean_nonref_acc-atk_nonref_acc

            corrfile=out/'per_image'/f'{pe}_seed{seed}_eps{eps_tag(eps)}_partition_correctness.npz'
            atomic_npz(corrfile,
                       nonreference_indices=np.array(nonref,dtype=np.int64),reference_indices=np.array(ref,dtype=np.int64),
                       clean_nonreference_correct=clean_nonref_correct,attacked_nonreference_correct=atk_nonref_correct,
                       clean_reference_correct=clean_ref_correct,attacked_reference_correct=atk_ref_correct)

            pos=score(attacked,Xh,clean_h_z,clean_h_cls,var,lam,a.measurement_batch_size)
            scorefile=out/'per_image'/f'{pe}_seed{seed}_eps{eps_tag(eps)}_holdout_positive_scores.npz'
            atomic_npz(scorefile,holdout_indices=np.array(hold,dtype=np.int64),
                       **{f'positive_{m}':pos[m].astype(np.float64) for m in METHODS})
            cell_auc={}
            for m in METHODS:
                primary=rank_auc(pos[m],pneg[m]); benign_only=rank_auc(pos[m],bneg[m])
                paired_conc=paired_benign_concordance(pos[m],benign[m])
                worst=worst_benign_exceedance_rate(pos[m],benign[m])
                if not (0<=primary<=1 and 0<=benign_only<=1 and 0<=paired_conc<=1 and 0<=worst<=1):
                    raise RuntimeError('AUC/diagnostic out of range')
                auc_rows.append({'pe':pe,'seed':seed,'epsilon':eps,'method':m,
                                 'primary_auc':primary,'benign_only_auc':benign_only,
                                 'paired_benign_concordance':paired_conc,
                                 'worst_benign_exceedance_rate':worst,
                                 'nonref_damage_pp':nonref_drop,'full_validation_damage_pp':s['full_damage']})
                cell_auc[m]={'primary_auc':primary,'benign_only_auc':benign_only,
                             'paired_benign_concordance':paired_conc,'worst_benign_exceedance_rate':worst}

            row={'pe':pe,'seed':seed,'epsilon':eps,
                 'clean_nonreference_accuracy':clean_nonref_acc,'attacked_nonreference_accuracy':atk_nonref_acc,
                 'nonreference_damage_pp':nonref_drop,'attacked_reference_accuracy':atk_ref_acc,
                 'reconstructed_full_validation_accuracy':reconstructed_full,
                 'source_full_validation_accuracy':s['full_acc'],'full_accuracy_reconstruction_discrepancy':full_discrepancy,
                 'original_full_validation_damage_pp':s['full_damage'],'source_delta_sha256':s['delta_sha'],
                 'partition_correctness_npz':str(corrfile.relative_to(out)),'partition_correctness_sha256':hfile(corrfile),
                 'positive_scores_npz':str(scorefile.relative_to(out)),'positive_scores_sha256':hfile(scorefile),
                 'calibration_npz':str(cal.relative_to(out)),'calibration_sha256':cal_sha,'metrics':cell_auc}
            cellfile=out/'cells'/f'{pe}_seed{seed}_eps{eps_tag(eps)}.json'
            atomic_json(cellfile,row); row['cell_json_sha256']=hfile(cellfile); cell_rows.append(row)
            nonref_rows.append({'pe':pe,'seed':seed,'epsilon':eps,
                                'clean_nonref_accuracy':clean_nonref_acc,'attacked_nonref_accuracy':atk_nonref_acc,
                                'nonref_damage_pp':nonref_drop,'attacked_reference_accuracy':atk_ref_acc,
                                'reconstructed_full_validation_accuracy':reconstructed_full,
                                'original_full_validation_damage_pp':s['full_damage'],
                                'original_full_validation_accuracy':s['full_acc']})
            del attacked
            torch.cuda.empty_cache()
        del clean,clean_h_z,clean_h_cls,var,lam
        torch.cuda.empty_cache()

    if done!=228: raise RuntimeError(f'execution cell count mismatch {done} != 228')
    nr=pd.DataFrame(nonref_rows); au=pd.DataFrame(auc_rows); cl=pd.DataFrame(clean_rows)
    atomic_csv(out/'stage_b_nonreference_damage.csv',nr)
    atomic_csv(out/'stage_b_holdout_auc_seedwise.csv',au)
    atomic_csv(out/'stage_b_clean_partition_checks.csv',cl)

    sm=au.groupby(['pe','epsilon','method']).agg(
        primary_mean=('primary_auc','mean'),primary_sd=('primary_auc','std'),primary_min=('primary_auc','min'),
        benign_mean=('benign_only_auc','mean'),benign_sd=('benign_only_auc','std'),benign_min=('benign_only_auc','min'),
        paired_concordance_mean=('paired_benign_concordance','mean'),paired_concordance_sd=('paired_benign_concordance','std'),
        worst_benign_exceedance_mean=('worst_benign_exceedance_rate','mean'),worst_benign_exceedance_sd=('worst_benign_exceedance_rate','std'),
        nonref_damage_mean=('nonref_damage_pp','mean'),nonref_damage_sd=('nonref_damage_pp','std'),
        full_damage_mean=('full_validation_damage_pp','mean')).reset_index()
    atomic_csv(out/'stage_b_holdout_auc_summary.csv',sm)

    bd=[]
    for pe in PES:
      for method in METHODS:
       q=sm[(sm.pe==pe)&(sm.method==method)].sort_values('epsilon')
       for est,col in [('primary','primary_min'),('benign_only','benign_min')]:
        hit=q[q[col]>=.99]
        if hit.empty:
            bd.append({'pe':pe,'method':method,'estimator':est,'boundary_epsilon':None,'status':'NOT_ESTABLISHED_WITHIN_AVAILABLE_GRID'})
        else:
            r=hit.iloc[0]
            bd.append({'pe':pe,'method':method,'estimator':est,'boundary_epsilon':float(r.epsilon),
                       'nonref_damage_mean':float(r.nonref_damage_mean),'full_damage_mean':float(r.full_damage_mean),
                       'status':'ESTABLISHED_EXACT_TESTED_EPSILON'})
    atomic_csv(out/'stage_b_holdout_boundaries.csv',pd.DataFrame(bd))

    impl=implementation_manifest(a.parent_lock_json,a.source_aggregate,a.reference_indices,a.holdout_indices)
    max_full_recon=float(np.max(np.abs(nr['reconstructed_full_validation_accuracy']-nr['original_full_validation_accuracy'])))
    metadata={
        'status':'PASS_EXECUTION','execution_id':EXECUTION_ID,'scientific_parent_protocol_id':PARENT_PROTOCOL_ID,
        'parent_lock_sha256':PARENT_LOCK_SHA,'source_full_grid_lock_sha256':FULL_GRID_LOCK_SHA,
        'operator_spec_hash':CANONICAL_OPERATOR_HASH,'source_aggregate_sha256':SOURCE_AGG_SHA,
        'reference_index_sha256':REF_INDEX_SHA,'holdout_index_sha256':HOLDOUT_INDEX_SHA,
        'reference_stream_sha256':got_ref,'full_validation_stream_sha256':got_full,
        'holdout_identity_stream_sha256':hold_stream,'full_cache_load_seconds':cache_seconds,
        'full_cache_shape':list(Xfull.shape),'full_cache_dtype':str(Xfull.dtype),'full_cache_device':str(Xfull.device),
        'n_source_cells':len(src),'n_executed_cells':done,'n_nonreference':len(nonref),'n_reference':len(ref),'n_holdout':len(hold),
        'n_auc_rows':len(auc_rows),'max_abs_full_accuracy_reconstruction_discrepancy_pp':max_full_recon,
        'runtime':runtime,'package_versions':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
                                             'torchvision':torchvision.__version__,'numpy':np.__version__,'pandas':pd.__version__,
                                             'pillow':PIL.__version__},
        'jpeg_options':JPEG_OPTIONS,'noise_rng':'numpy.random.PCG64','implementation_manifest':impl}
    atomic_json(out/'STAGE_B_EXECUTION_METADATA.json',metadata)

    decision={
        'decision':'PASS_STAGE_B_EXECUTION','scientific_interpretation_status':'PENDING_EXTERNAL_AUDIT_AND_LEDGER_UPDATE',
        'gates':{
            'parent_lock_sha':True,'source_aggregate_sha':True,'reference_index_sha':True,'holdout_index_sha':True,
            'canonical_operator_hash':True,'full_transformed_cache_sha':got_full==FULL_STREAM_SHA,
            'reference_transformed_cache_sha':got_ref==REF_STREAM_SHA,'source_deltas_verified':'228/228',
            'executed_cells':'228/228','full_accuracy_reconstruction_all_cells':max_full_recon<=FULL_ACC_TOL,
            'identity_score_zero_all_models':bool(cl['identity_max_abs_score'].max()<=IDENTITY_TOL),
            'finite_scores':True,'secondary_diagnostics_present':True,'benign_transform_sha_manifest_present':True},
        'do_not_edit_manuscript_before':'external result audit + Scientific Results and Interpretation Ledger update'}
    atomic_json(out/'STAGE_B_DECISION.json',decision)

    entries=build_output_manifest(out)
    z,zh,side=make_results_zip(out)
    print(f'STAGE B EXECUTION PASS: {done}/228 cells',flush=True)
    print(f'RESULTS ZIP: {z}',flush=True)
    print(f'RESULTS ZIP SHA256: {zh}',flush=True)
    print(f'SIDECAR: {side}',flush=True)


if __name__=='__main__':
    main()
