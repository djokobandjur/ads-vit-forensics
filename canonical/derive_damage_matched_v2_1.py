#!/usr/bin/env python3
"""Damage-matched derivation for ADS/TIFS full-grid protocol lock v2.0.

Normative targets: 5 pp primary; 10/20 pp secondary; severe = attacked <= 0.5 clean.
First crossing, linear-in-damage interpolation, identical weight for all ADS
quantities and [256,12] per-image arrays, no extrapolation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

LOCK_SHA='2b242559d0ec523683ee05a33238c819790d17ebef919745cab3b426f161021f'
PE_TYPES=('learned','sinusoidal','rope','alibi'); SEEDS=(42,123,456,789,1011,1213)

def sha(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    p=Path(path); tmp=Path(str(p)+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)); tmp.replace(p)

def load_matrix(cell):
    art=cell['per_image_artifact']; p=Path(art['path'])
    if not p.exists() or sha(p)!=art['sha256']: raise RuntimeError(f'Per-image artifact mismatch: {p}')
    with np.load(p,allow_pickle=False) as z:
        a=np.asarray(z['per_image_ads_canonical_logit_domain'],dtype=np.float64)
    if a.shape!=(256,12): raise RuntimeError(f'Bad per-image shape {a.shape} from {p}')
    return a

def crossings(xs,target):
    # Exact observed hits are direct candidates and are not double-counted as
    # adjacent interpolation intervals. Strict crossings are added separately.
    out=[]
    exact=[i for i,x in enumerate(xs) if abs(float(x['accuracy_drop_pp'])-float(target)) <= 1e-12]
    for i in exact:
        out.append(('exact',i,i,0.0))
    for i in range(len(xs)-1):
        d0=float(xs[i]['accuracy_drop_pp']); d1=float(xs[i+1]['accuracy_drop_pp'])
        if abs(d0-target)<=1e-12 or abs(d1-target)<=1e-12 or d1==d0:
            continue
        if (d0<target<d1) or (d1<target<d0):
            w=(target-d0)/(d1-d0); out.append(('interp',i,i+1,w))
    out.sort(key=lambda z: (xs[z[1]]['epsilon'], 0 if z[0]=='exact' else 1))
    return out

def derive_one(xs,target,label,out_dir):
    xs=sorted(xs,key=lambda x:x['epsilon']); cr=crossings(xs,target)
    if not cr: return {'target':label,'target_damage_pp':float(target),'estimable':False,'reason':'NO_BRACKET_NO_EXTRAPOLATION','nonmonotonic_damage':False}
    kind,i,j,w=cr[0]; a=xs[i]; b=xs[j]; nonmono=len(cr)>1
    if kind=='exact': b=a; w=0.0
    def lerp(v0,v1): return (1-w)*float(v0)+w*float(v1)
    ca=a['canonical_ads']['canonical']; cb=b['canonical_ads']['canonical']
    layers=[lerp(x,y) for x,y in zip(ca['per_layer_ads'],cb['per_layer_ads'])]
    ma=load_matrix(a); mb=ma if i==j else load_matrix(b); mm=(1-w)*ma+w*mb
    pe=a['metadata']['pe_type']; seed=int(a['metadata']['seed']); npz=out_dir/f'{pe}_seed{seed}_{label}_per_image_ads.npz'; np.savez_compressed(npz,per_image_ads_canonical_logit_domain=mm.astype(np.float32),target=np.asarray([label]),target_damage_pp=np.asarray([target],dtype=np.float64),interpolation_weight=np.asarray([w],dtype=np.float64))
    src=[]
    for x in (a,b):
        p=Path(x['_cell_json_path']); src.append({'path':str(p),'sha256':sha(p),'epsilon':x['epsilon'],'damage_pp':x['accuracy_drop_pp']})
    return {'target':label,'target_damage_pp':float(target),'estimable':True,'source_mode':kind,'endpoint_epsilon_low':float(a['epsilon']),'endpoint_epsilon_high':float(b['epsilon']),'endpoint_damage_low_pp':float(a['accuracy_drop_pp']),'endpoint_damage_high_pp':float(b['accuracy_drop_pp']),'interpolation_weight':float(w),'interpolated_epsilon':lerp(a['epsilon'],b['epsilon']),'mean_ads':lerp(ca['mean_ads'],cb['mean_ads']),'l4_ads':lerp(ca['layer4_ads'],cb['layer4_ads']),'per_layer_ads':layers,'argmax_layer_1based':int(np.argmax(layers)+1),'nonmonotonic_damage':bool(nonmono),'crossing_count':len(cr),'source_cells':src,'per_image_artifact':{'path':str(npz.resolve()),'sha256':sha(npz),'shape':[256,12],'dtype':'float32'}}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--full_grid_root',required=True); p.add_argument('--output_root',required=True); a=p.parse_args()
    root=Path(a.full_grid_root); out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True); per=out/'per_image'; per.mkdir(exist_ok=True)
    agg=json.load(open(root/'ads_tifs_canonical_primary_full_grid_v2_1.json','r',encoding='utf-8'))
    if agg['metadata']['status']!='COMPLETE' or agg['metadata']['protocol_lock']['sha256']!=LOCK_SHA: raise RuntimeError('Full grid is not COMPLETE under the locked v2.0 protocol')
    cells=[]
    expected_sources={Path(x['path']).resolve():x['sha256'] for x in agg.get('source_cell_json_artifacts',[]) if x.get('path') and x.get('sha256')}
    for pth in sorted((root/'cells').glob('*.json')):
        rp=pth.resolve(); actual_sha=sha(pth)
        if expected_sources and expected_sources.get(rp)!=actual_sha:
            raise RuntimeError(f'Source cell JSON identity mismatch vs aggregate: {pth}')
        x=json.load(open(pth,'r',encoding='utf-8'))
        md=x.get('metadata',{})
        if md.get('protocol_lock',{}).get('sha256')!=LOCK_SHA or md.get('operator_spec_hash')!='093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362':
            raise RuntimeError(f'Cell protocol/operator mismatch: {pth}')
        for key in ('delta_artifact','per_image_artifact'):
            art=x.get(key)
            if not art or not art.get('path') or not art.get('sha256') or not Path(art['path']).exists() or sha(art['path'])!=art['sha256']:
                raise RuntimeError(f'Cell artifact identity mismatch ({key}): {pth}')
        x['_cell_json_path']=str(rp); cells.append(x)
    if len(cells)!=228: raise RuntimeError(f'Expected 228 verified source cells, got {len(cells)}')
    rows=[]
    for pe in PE_TYPES:
        for seed in SEEDS:
            xs=[x for x in cells if x['metadata']['pe_type']==pe and int(x['metadata']['seed'])==seed]
            clean=float(xs[0]['clean_full_validation_accuracy'])
            for label,target in [('damage_5pp',5.0),('damage_10pp',10.0),('damage_20pp',20.0),('severe_50pct_clean',0.5*clean)]:
                r=derive_one(xs,target,label,per); r.update({'pe_type':pe,'seed':seed,'clean_accuracy':clean}); rows.append(r)
    coverage={}
    for label in ['damage_5pp','damage_10pp','damage_20pp','severe_50pct_clean']:
        rr=[r for r in rows if r['target']==label]; n=sum(bool(r['estimable']) for r in rr); coverage[label]={'estimable_units':n,'total_units':24,'common_cross_family_estimand':bool(n==24)}
    primary_pass=coverage['damage_5pp']['common_cross_family_estimand']
    result={'metadata':{'protocol_lock_sha256':LOCK_SHA,'source_full_grid':str((root/'ads_tifs_canonical_primary_full_grid_v2_1.json').resolve()),'source_full_grid_sha256':sha(root/'ads_tifs_canonical_primary_full_grid_v2_1.json'),'primary_5pp_status':'PASS_ALL_24' if primary_pass else 'FAIL_CLOSED_TARGETED_REFINEMENT_REQUIRED','rule':'first adjacent crossing; exact hit direct; linear in damage; identical weight for epsilon/ADS/profile/per-image; no extrapolation'},'coverage':coverage,'rows':rows}
    atomic_json(out/'ads_tifs_damage_matched_v2_1.json',result)
    fields=['pe_type','seed','target','target_damage_pp','estimable','interpolated_epsilon','mean_ads','l4_ads','argmax_layer_1based','endpoint_epsilon_low','endpoint_epsilon_high','interpolation_weight','nonmonotonic_damage','crossing_count']
    with open(out/'ads_tifs_damage_matched_v2_1.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in fields})
    with open(out/'DAMAGE_MATCHED_DECISION.txt','w',encoding='utf-8') as f:
        f.write('Primary 5-pp status: '+result['metadata']['primary_5pp_status']+'\n')
        for k,v in coverage.items(): f.write(f'{k}: {v["estimable_units"]}/24; common={v["common_cross_family_estimand"]}\n')
        if not primary_pass: f.write('STOP: do not report a common 5-pp cross-family estimand; create separately locked targeted refinement.\n')
    print('Damage-matched derivation complete.'); print(json.dumps(coverage,indent=2)); print('Primary:',result['metadata']['primary_5pp_status'])

if __name__=='__main__': main()
