#!/usr/bin/env python3
"""ADS/TIFS canonical primary fixed-absolute full grid v2.1.

Execution package implementing the locked scientific protocol:
  ADS_TIFS_FULL_GRID_PROTOCOL_LOCK_v2_0_20260909
  canonical JSON SHA-256:
  2b242559d0ec523683ee05a33238c819790d17ebef919745cab3b426f161021f

Locked scope:
  PE = learned, sinusoidal, rope, alibi
  seeds = 42,123,456,789,1011,1213
  common epsilon = 0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2
  high-tail +0.5,1.0 for learned/sinusoidal/rope only
  ALiBi stops at 0.2

Attack generator:
  corrected full-reference CE-only projected ascent; model.eval(); 20 steps;
  alpha=0.1*epsilon; zero init; raw final step-20 iterate; PE-only;
  per-buffer delta; absolute per-group L_inf; biases excluded.

Measurement:
  full 5000-image validation accuracy + canonical ADS on fixed 256 reference;
  per-image [256,12] canonical ADS saved for every attacked cell.

Execution:
  deterministic math-only SDPA; num_workers=0; full validation transform cached
  once on GPU; fixed reference view selected from that cache; both SHA-verified.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from full_scale_experiment_v1_6 import VisionTransformer
from ads_attack_engine_v2_1 import (
    projected_ce_ascent, measure_accuracy, configure_deterministic_runtime,
    runtime_provenance, file_identity, atomic_json_dump, sha256_file,
    collect_attack_groups,
)
from ads_canonical_operator_v2_1 import (
    measure_canonical_ads, canonical_operator_spec,
    CANONICAL_OPERATOR_HASH, assert_operator_lock,
)

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
PROTOCOL_LOCK_ID='ADS_TIFS_FULL_GRID_PROTOCOL_LOCK_v2_0_20260909'
PROTOCOL_LOCK_SHA='2b242559d0ec523683ee05a33238c819790d17ebef919745cab3b426f161021f'
EXECUTION='ADS_TIFS_CANONICAL_PRIMARY_FULL_GRID_v2_1_20260909'
PE_TYPES=('learned','sinusoidal','rope','alibi')
SEEDS=(42,123,456,789,1011,1213)
COMMON_EPS=(0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2)
HIGH_TAIL=(0.5,1.0)
EXPECTED_REF_SHA='1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509'
EXPECTED_TRANSFORMED_REF_SHA='01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4'
EXPECTED_FULL_VAL_SHA='90d47ea5212f5d06b9e8d12fa80ccb20cc722a46d79d2c926ef41be2917f0f71'
EXPECTED_GROUPS={'learned':1,'sinusoidal':1,'rope':24,'alibi':12}

class CachedTensorView:
    def __init__(self,images,labels,batch_size):
        if images.device!=labels.device: raise ValueError('cache device mismatch')
        if images.shape[0]!=labels.shape[0]: raise ValueError('cache length mismatch')
        self.images=images; self.labels=labels; self.batch_size=int(batch_size)
        self.dataset=range(int(labels.shape[0]))
    def __len__(self): return math.ceil(len(self.dataset)/self.batch_size)
    def __iter__(self):
        n=len(self.dataset)
        for s in range(0,n,self.batch_size):
            e=min(s+self.batch_size,n)
            yield self.images[s:e],self.labels[s:e]

def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--models_dir',required=True)
    p.add_argument('--val_dir',required=True)
    p.add_argument('--ref_indices_path',required=True)
    p.add_argument('--protocol_lock_json',required=True)
    p.add_argument('--output_root',required=True)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--alpha_ratio',type=float,default=0.1)
    p.add_argument('--attack_batch_size',type=int,default=32)
    p.add_argument('--measurement_batch_size',type=int,default=8)
    p.add_argument('--validation_batch_size',type=int,default=128)
    p.add_argument('--cache_load_batch_size',type=int,default=128)
    p.add_argument('--num_workers',type=int,default=0)
    p.add_argument('--delta_storage_dtype',choices=['float32'],default='float32')
    p.add_argument('--overwrite',action='store_true')
    return p.parse_args()

def eps_for_pe(pe): return COMMON_EPS if pe=='alibi' else COMMON_EPS+HIGH_TAIL

def eps_tag(x): return f'{x:.6g}'.replace('.','p').replace('-','m')

def _tensor_bytes(t): return t.detach().contiguous().cpu().numpy().tobytes(order='C')

def _hash_ref_stream(images,labels,batch_size):
    h=hashlib.sha256(); n=int(labels.shape[0])
    for s in range(0,n,batch_size):
        e=min(s+batch_size,n); h.update(_tensor_bytes(images[s:e])); h.update(_tensor_bytes(labels[s:e]))
    return h.hexdigest()

def build_gpu_validation_and_reference_cache(a):
    if DEVICE!='cuda': raise RuntimeError('CUDA/H200 required for locked GPU-cache execution path')
    if a.num_workers!=0: raise RuntimeError('FMLE lock requires num_workers=0')
    if a.attack_batch_size!=32: raise RuntimeError('attack_batch_size=32 locked for transformed-reference SHA partition')
    if a.cache_load_batch_size!=128: raise RuntimeError('cache_load_batch_size=128 locked for full-validation SHA partition')
    tf=transforms.Compose([
        transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds=datasets.ImageFolder(a.val_dir,tf)
    if len(ds)!=5000: raise RuntimeError(f'Expected ImageNet-100 validation size 5000, got {len(ds)}')
    got_ref_file=sha256_file(a.ref_indices_path)
    if got_ref_file!=EXPECTED_REF_SHA: raise RuntimeError(f'Reference-index SHA mismatch: {got_ref_file} != {EXPECTED_REF_SHA}')
    idx=json.load(open(a.ref_indices_path,'r',encoding='utf-8'))
    if not isinstance(idx,list) or len(idx)!=256 or len(set(idx))!=256: raise ValueError('Reference indices must be exactly 256 unique integers')
    if min(idx)<0 or max(idx)>=len(ds): raise ValueError('Reference indices incompatible with validation set')

    need=int(len(ds)*3*224*224*4 + len(ds)*8); free,total=torch.cuda.mem_get_info(); reserve=4*1024**3
    print(f'GPU CACHE PREFLIGHT: val={len(ds)} estimated={need/1024**3:.2f} GiB free={free/1024**3:.2f} GiB total={total/1024**3:.2f} GiB',flush=True)
    if free < need+reserve: raise RuntimeError('Insufficient free GPU memory for locked full-validation cache + 4 GiB reserve')

    source=DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,pin_memory=True)
    X=torch.empty((len(ds),3,224,224),device=DEVICE,dtype=torch.float32)
    Y=torch.empty((len(ds),),device=DEVICE,dtype=torch.long)
    h=hashlib.sha256(); pos=0; t0=time.time()
    print('GPU VALIDATION CACHE: decoding/transforming 5000 images exactly once ...',flush=True)
    for bi,(x,y) in enumerate(source,1):
        h.update(x.contiguous().numpy().tobytes(order='C')); h.update(y.contiguous().numpy().tobytes(order='C'))
        e=pos+int(y.numel()); X[pos:e].copy_(x.to(DEVICE,non_blocking=True)); Y[pos:e].copy_(y.to(DEVICE,non_blocking=True)); pos=e
        if bi==1 or bi==len(source) or bi%5==0: print(f'  cache batch {bi}/{len(source)} images={pos}/5000',flush=True)
    if pos!=5000: raise RuntimeError(f'Validation cache saw {pos}, expected 5000')
    torch.cuda.synchronize(); sec=time.time()-t0; full_sha=h.hexdigest()
    if full_sha!=EXPECTED_FULL_VAL_SHA: raise RuntimeError(f'Full validation transformed SHA mismatch: {full_sha} != {EXPECTED_FULL_VAL_SHA}')

    it=torch.tensor(idx,device=DEVICE,dtype=torch.long); XR=X.index_select(0,it).contiguous(); YR=Y.index_select(0,it).contiguous()
    ref_sha=_hash_ref_stream(XR,YR,32)
    if ref_sha!=EXPECTED_TRANSFORMED_REF_SHA: raise RuntimeError(f'Transformed reference SHA mismatch: {ref_sha} != {EXPECTED_TRANSFORMED_REF_SHA}')
    full_info={'mode':'single_materialization_gpu_resident_full_validation_shared_all_cells','n_images':5000,'images_shape':list(X.shape),'images_dtype':str(X.dtype),'labels_dtype':str(Y.dtype),'device':str(X.device),'cache_bytes':int(X.numel()*X.element_size()+Y.numel()*Y.element_size()),'cache_gib':float((X.numel()*X.element_size()+Y.numel()*Y.element_size())/1024**3),'load_transform_transfer_seconds':float(sec),'transformed_full_validation_stream_sha256':full_sha,'cache_load_batch_size':128,'source_num_workers':0,'scientific_effect':'NONE; execution-only deterministic transform cache'}
    ref_info={'mode':'gpu_reference_view_selected_from_locked_full_validation_cache','n_images':256,'images_shape':list(XR.shape),'images_dtype':str(XR.dtype),'labels_dtype':str(YR.dtype),'device':str(XR.device),'cache_bytes':int(XR.numel()*XR.element_size()+YR.numel()*YR.element_size()),'cache_mib':float((XR.numel()*XR.element_size()+YR.numel()*YR.element_size())/1024**2),'transformed_tensor_stream_sha256':ref_sha,'attack_batch_size':32,'measurement_batch_size':int(a.measurement_batch_size),'scientific_effect':'NONE; locked reference tensors/order'}
    print(f'GPU VALIDATION CACHE PASS: {full_sha}',flush=True); print(f'GPU REFERENCE CACHE PASS: {ref_sha}',flush=True)
    return CachedTensorView(X,Y,a.validation_batch_size),CachedTensorView(XR,YR,a.attack_batch_size),CachedTensorView(XR,YR,a.measurement_batch_size),full_info,ref_info

def checkpoint_path(models_dir,pe,seed): return Path(models_dir)/f'{pe}_seed{seed}'/'best_model.pth'

def load_clean_model(models_dir,pe,seed):
    ck=checkpoint_path(models_dir,pe,seed)
    if not ck.exists(): raise FileNotFoundError(ck)
    m=VisionTransformer(img_size=224,patch_size=16,num_classes=100,embed_dim=768,depth=12,num_heads=12,mlp_ratio=4.0,dropout=0.1,pe_type=pe)
    st=torch.load(ck,map_location='cpu')
    if isinstance(st,dict) and 'model_state_dict' in st: st=st['model_state_dict']
    st={k.replace('_orig_mod.',''):v for k,v in st.items()}
    m.load_state_dict(st,strict=True); m.eval().to(DEVICE); return m,ck

def reference_mean_ce(model,loader):
    model.eval(); crit=nn.CrossEntropyLoss(reduction='sum'); s=0.0; n=0
    with torch.no_grad():
        for x,y in loader:
            z=model(x.to(DEVICE,non_blocking=True)); yy=y.to(DEVICE,non_blocking=True); s+=float(crit(z,yy).item()); n+=int(yy.numel())
    if n!=256: raise RuntimeError(f'Reference CE saw {n}, expected 256')
    return s/n

def assert_topology(clean,pe):
    probe=copy.deepcopy(clean).to(DEVICE); probe.eval()
    groups=collect_attack_groups(probe,pe,'pe_only',DEVICE,delta_convention='per_buffer',include_biases=False)
    names=[g.name for g in groups]; count=len(groups); del probe,groups; torch.cuda.empty_cache()
    if count!=EXPECTED_GROUPS[pe]: raise RuntimeError(f'PE topology mismatch for {pe}: {count} != {EXPECTED_GROUPS[pe]}')
    return {'group_count':count,'group_names':names}

def stat(x):
    a=np.asarray(x,dtype=float); return {'n':int(a.size),'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if a.size>1 else 0.0,'min':float(a.min()),'max':float(a.max())}

def implementation_manifest():
    names=['ads_primary_full_grid_v2_1.py','ads_attack_engine_v2_1.py','ads_canonical_operator_v2_1.py','full_scale_experiment_v1_6.py']
    return {n:file_identity(str(HERE/n)) for n in names}

def same_impl(recorded,current):
    try: return all(recorded[n]['sha256']==current[n]['sha256'] for n in current)
    except Exception: return False

def validate_existing_cell(path,pe,seed,eps,cksha,impl,a):
    try: x=json.load(open(path,'r',encoding='utf-8'))
    except Exception: return None
    md=x.get('metadata',{}); atk=md.get('attack',{}); lock=md.get('protocol_lock',{}); rg=md.get('reference_gpu_cache',{}); fg=md.get('full_validation_gpu_cache',{})
    ok=(md.get('execution_package')==EXECUTION and md.get('pe_type')==pe and int(md.get('seed'))==int(seed) and abs(float(x.get('epsilon'))-float(eps))<1e-15 and md.get('operator_spec_hash')==CANONICAL_OPERATOR_HASH and
        lock.get('lock_id')==PROTOCOL_LOCK_ID and lock.get('sha256')==PROTOCOL_LOCK_SHA and md.get('checkpoint',{}).get('sha256')==cksha and
        rg.get('transformed_tensor_stream_sha256')==EXPECTED_TRANSFORMED_REF_SHA and fg.get('transformed_full_validation_stream_sha256')==EXPECTED_FULL_VAL_SHA and
        int(atk.get('steps',-1))==20 and abs(float(atk.get('alpha_ratio',-1))-0.1)<1e-15 and atk.get('objective')=='full_256_reference_mean_CE_ascent' and atk.get('surface')=='pe_only' and atk.get('delta_convention')=='per_buffer' and atk.get('budget_mode')=='absolute' and atk.get('delta_init')=='zero' and atk.get('selected_state')=='raw_final_step20' and
        same_impl(md.get('implementation_manifest',{}),impl))
    if not ok: return None
    da=x.get('delta_artifact') or x.get('attack',{}).get('delta_artifact'); pa=x.get('per_image_artifact') or x.get('canonical_ads',{}).get('per_image_artifact')
    for art in (da,pa):
        if not art or not art.get('path') or not art.get('sha256'): return None
        p=Path(art['path'])
        if not p.exists() or sha256_file(str(p))!=art['sha256']: return None
    if int(x.get('attack',{}).get('group_count',-1))!=EXPECTED_GROUPS[pe]: return None
    return x

def write_cell_csv(path,cells):
    fields=['pe_type','seed','epsilon','clean_accuracy','attacked_accuracy','drop_pp','reference_ce','mean_ads','l4_ads','argmax_layer','l12_over_l1','group_count','max_sign_flip_fraction','mean_saturation_fraction','delta_sha256','per_image_sha256']
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for x in cells:
            g=x['attack']['attack_geometry']; pg=g.get('per_group',[]); sat=float(np.mean([z['saturated_fraction'] for z in pg])) if pg else 0.0; c=x['canonical_ads']['canonical']
            w.writerow({'pe_type':x['metadata']['pe_type'],'seed':x['metadata']['seed'],'epsilon':x['epsilon'],'clean_accuracy':x['clean_full_validation_accuracy'],'attacked_accuracy':x['full_validation_accuracy'],'drop_pp':x['accuracy_drop_pp'],'reference_ce':x['attacked_reference_mean_ce'],'mean_ads':c['mean_ads'],'l4_ads':c['layer4_ads'],'argmax_layer':c['argmax_layer_1based'],'l12_over_l1':c['l12_over_l1'],'group_count':x['attack']['group_count'],'max_sign_flip_fraction':g['max_sign_flip_fraction'],'mean_saturation_fraction':sat,'delta_sha256':x['delta_artifact']['sha256'],'per_image_sha256':x['per_image_artifact']['sha256']})

def main():
    a=parse_args(); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    if a.steps!=20 or abs(a.alpha_ratio-0.1)>1e-15: raise RuntimeError('Attack generator drift: steps=20 and alpha_ratio=0.1 are locked')
    if a.num_workers!=0 or a.delta_storage_dtype!='float32': raise RuntimeError('Execution lock requires num_workers=0 and float32 delta artifacts')
    lock_id=file_identity(a.protocol_lock_json)
    if not lock_id or lock_id.get('sha256')!=PROTOCOL_LOCK_SHA: raise RuntimeError(f'Protocol lock SHA mismatch: {lock_id}')
    lock_json=json.load(open(a.protocol_lock_json,'r',encoding='utf-8'))
    if lock_json.get('lock_id')!=PROTOCOL_LOCK_ID: raise RuntimeError('Protocol lock ID mismatch')
    got=assert_operator_lock(); configure_deterministic_runtime(seed=SEEDS[0])
    impl=implementation_manifest()
    print('PROTOCOL LOCK:',PROTOCOL_LOCK_ID,PROTOCOL_LOCK_SHA); print('EXECUTION:',EXECUTION); print('OPERATOR HASH:',got); print('DEVICE:',DEVICE)
    print('CELLS EXPECTED: 228 = 4*6*8 common + 3*6*2 high-tail')
    atomic_json_dump(str(out/'IMPLEMENTATION_MANIFEST_RUNTIME.json'),{'execution_package':EXECUTION,'protocol_lock':lock_id,'implementation_files':impl})

    ck_manifest={}
    for pe in PE_TYPES:
        ck_manifest[pe]={}
        for seed in SEEDS:
            ck=checkpoint_path(a.models_dir,pe,seed)
            if not ck.exists(): raise FileNotFoundError(ck)
            ck_manifest[pe][str(seed)]=file_identity(str(ck))
    atomic_json_dump(str(out/'CHECKPOINT_MANIFEST_RUNTIME.json'),{'execution_package':EXECUTION,'protocol_lock':lock_id,'checkpoints':ck_manifest})
    print('CHECKPOINT PREFLIGHT PASS: 24 checkpoints found and SHA-256 recorded.',flush=True)

    val_view,attack_ref,measure_ref,full_cache,ref_cache=build_gpu_validation_and_reference_cache(a)
    cell_dir=out/'cells'; delta_dir=out/'deltas'; per_dir=out/'per_image'
    for d in (cell_dir,delta_dir,per_dir): d.mkdir(parents=True,exist_ok=True)
    cells=[]; clean_by_pe_seed={}; topology_by_pe_seed={}

    for pi,pe in enumerate(PE_TYPES,1):
        clean_by_pe_seed[pe]={}; topology_by_pe_seed[pe]={}
        print('\n'+'@'*100); print(f'PE {pe.upper()} ({pi}/4)'); print('@'*100,flush=True)
        for si,seed in enumerate(SEEDS,1):
            configure_deterministic_runtime(seed=seed); clean,_=load_clean_model(a.models_dir,pe,seed); ckid=ck_manifest[pe][str(seed)]
            topo=assert_topology(clean,pe); topology_by_pe_seed[pe][str(seed)]=topo
            clean_acc=measure_accuracy(clean,val_view,DEVICE); clean_ce=reference_mean_ce(clean,attack_ref)
            clean_by_pe_seed[pe][str(seed)]={'full_validation_accuracy':clean_acc,'reference_mean_ce':clean_ce,'checkpoint':ckid,'topology':topo,'runtime':runtime_provenance(seed=seed)}
            print(f'CLEAN {pe} seed={seed}: acc={clean_acc:.4f}% refCE={clean_ce:.9f} groups={topo["group_count"]} SHA={ckid["sha256"]}',flush=True)
            for eps in eps_for_pe(pe):
                tag=eps_tag(eps); cp=cell_dir/f'{pe}_seed{seed}_eps{tag}.json'; dp=delta_dir/f'imagenet_{pe}_seed{seed}_eps{tag}_absolute_pgd20_ar0p1_v2_1.pt'; npz=per_dir/f'{pe}_seed{seed}_eps{tag}_per_image_ads.npz'
                if cp.exists() and not a.overwrite:
                    ex=validate_existing_cell(cp,pe,seed,eps,ckid['sha256'],impl,a)
                    if ex is not None:
                        print(f'RESUME PASS: {pe} seed={seed} eps={eps:g}',flush=True); cells.append(ex); continue
                    raise RuntimeError(f'Existing cell failed fail-closed resume validation: {cp}')
                print('\n'+'='*92); print(f'{pe.upper()} seed={seed} epsilon={eps:g}'); print('='*92,flush=True)
                attacked,attack=projected_ce_ascent(clean,pe,'pe_only',eps,attack_ref,DEVICE,steps=20,alpha_ratio=0.1,delta_convention='per_buffer',include_biases=False,budget_mode='absolute',delta_save_path=str(dp),delta_storage_dtype='float32',progress_label=f'{pe} seed={seed} eps={eps:g}',artifact_protocol_version=PROTOCOL_LOCK_ID,artifact_execution_implementation=EXECUTION)
                if int(attack.get('group_count',-1))!=EXPECTED_GROUPS[pe]: raise RuntimeError(f'Attack topology drift {pe}: {attack.get("group_count")}')
                acc=measure_accuracy(attacked,val_view,DEVICE); drop=clean_acc-acc; attacked_ce=reference_mean_ce(attacked,attack_ref)
                ads=measure_canonical_ads(clean,attacked,measure_ref,DEVICE,per_image_path=str(npz),progress_label=f'{pe} seed={seed} eps={eps:g} ADS',retain_compat=True,retain_legacy=True)
                cell={'metadata':{'execution_package':EXECUTION,'protocol_lock':{'lock_id':PROTOCOL_LOCK_ID,'path':lock_id['path'],'sha256':PROTOCOL_LOCK_SHA},'scientific_role':'canonical_PE_only_fixed_absolute_primary_grid','pe_type':pe,'seed':seed,'operator_spec_hash':CANONICAL_OPERATOR_HASH,'operator_spec':canonical_operator_spec(),'checkpoint':ckid,'reference_indices':file_identity(a.ref_indices_path),'reference_gpu_cache':ref_cache,'full_validation_gpu_cache':full_cache,'implementation_manifest':impl,'runtime':runtime_provenance(seed=seed),'attack':{'steps':20,'alpha_ratio':0.1,'objective':'full_256_reference_mean_CE_ascent','surface':'pe_only','delta_convention':'per_buffer','budget_mode':'absolute','delta_init':'zero','model_mode':'eval','selected_state':'raw_final_step20','restarts':1,'biases_included':False},'interpretation_lock':'fixed absolute epsilon is implementation-space tampering stress; cross-family robustness interpretation uses damage-matched analysis'},'epsilon':eps,'clean_full_validation_accuracy':clean_acc,'full_validation_accuracy':acc,'accuracy_drop_pp':drop,'relative_accuracy_reduction':drop/clean_acc if clean_acc else None,'clean_reference_mean_ce':clean_ce,'attacked_reference_mean_ce':attacked_ce,'damage_flags':{'nontrivial_ge_5pp':bool(drop>=5.0),'ge_10pp':bool(drop>=10.0),'ge_20pp':bool(drop>=20.0),'severe_le_50pct_clean':bool(acc<=0.5*clean_acc)},'attack':attack,'canonical_ads':ads,'delta_artifact':attack['delta_artifact'],'per_image_artifact':ads['per_image_artifact']}
                atomic_json_dump(str(cp),cell); cells.append(cell)
                c=ads['canonical']; print(f'RESULT {pe} seed={seed} eps={eps:g}: acc={acc:.4f}% drop={drop:.4f}pp refCE={attacked_ce:.6f} meanADS={c["mean_ads"]:.6f} L4={c["layer4_ads"]:.6f} argmax=L{c["argmax_layer_1based"]}',flush=True)
                del attacked; torch.cuda.empty_cache()
            del clean; torch.cuda.empty_cache()

    expected=sum(len(eps_for_pe(pe))*len(SEEDS) for pe in PE_TYPES); cells=sorted(cells,key=lambda x:(PE_TYPES.index(x['metadata']['pe_type']),x['metadata']['seed'],x['epsilon']))
    status='COMPLETE' if len(cells)==expected else 'INCOMPLETE'
    aggregates=[]
    for pe in PE_TYPES:
        for eps in eps_for_pe(pe):
            xs=[x for x in cells if x['metadata']['pe_type']==pe and abs(x['epsilon']-eps)<1e-15]
            if not xs: continue
            aggregates.append({'pe_type':pe,'epsilon':eps,'n':len(xs),'attacked_accuracy':stat([x['full_validation_accuracy'] for x in xs]),'accuracy_drop_pp':stat([x['accuracy_drop_pp'] for x in xs]),'reference_ce':stat([x['attacked_reference_mean_ce'] for x in xs]),'canonical_mean_ads':stat([x['canonical_ads']['canonical']['mean_ads'] for x in xs]),'canonical_l4_ads':stat([x['canonical_ads']['canonical']['layer4_ads'] for x in xs]),'l12_over_l1':stat([x['canonical_ads']['canonical']['l12_over_l1'] for x in xs]),'argmax_layer_counts':dict(Counter(int(x['canonical_ads']['canonical']['argmax_layer_1based']) for x in xs)),'ge5_count':sum(x['damage_flags']['nontrivial_ge_5pp'] for x in xs),'severe_count':sum(x['damage_flags']['severe_le_50pct_clean'] for x in xs)})
    source_cells=[]
    for p in sorted(cell_dir.glob('*.json')): source_cells.append(file_identity(str(p)))
    agg={'metadata':{'execution_package':EXECUTION,'protocol_lock':{'lock_id':PROTOCOL_LOCK_ID,'sha256':PROTOCOL_LOCK_SHA},'status':status,'pe_types':list(PE_TYPES),'seeds':list(SEEDS),'common_eps':list(COMMON_EPS),'high_tail':list(HIGH_TAIL),'expected_cells':expected,'completed_cells':len(cells),'operator_spec_hash':CANONICAL_OPERATOR_HASH,'reference_gpu_cache':ref_cache,'full_validation_gpu_cache':full_cache,'implementation_manifest':impl,'attack_generator':{'steps':20,'alpha_ratio':0.1,'objective':'full_256_reference_mean_CE_ascent','surface':'pe_only','delta_convention':'per_buffer','budget_mode':'absolute','delta_init':'zero','selected_state':'raw_final_step20','restarts':1},'damage_matched_primary_target_pp':5.0},'checkpoint_manifest':ck_manifest,'topology_by_pe_seed':topology_by_pe_seed,'clean_by_pe_seed':clean_by_pe_seed,'aggregate_by_pe_epsilon':aggregates,'source_cell_json_artifacts':source_cells,'cells':cells}
    ap=out/'ads_tifs_canonical_primary_full_grid_v2_1.json'; atomic_json_dump(str(ap),agg)
    cp=out/'ads_tifs_canonical_primary_full_grid_cell_level_v2_1.csv'; write_cell_csv(cp,cells)
    with open(out/'FULL_GRID_DECISION_INPUT.txt','w',encoding='utf-8') as f:
        f.write(f'Canonical primary full grid v2.1 status: {status}; cells {len(cells)}/{expected}\n')
        f.write(f'protocol lock SHA: {PROTOCOL_LOCK_SHA}\noperator hash: {CANONICAL_OPERATOR_HASH}\n')
        f.write('Next: run damage-matched derivation script under DAMAGE_MATCHED_ANALYSIS_SPEC.md. Primary 5-pp target must bracket all 24 PE×seed units or fail closed.\n')
    print('\nFULL GRID COMPLETE' if status=='COMPLETE' else '\nFULL GRID INCOMPLETE'); print('Aggregate:',ap); print('Cell CSV:',cp)

if __name__=='__main__': main()
