#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib, math, os, sys, time, zipfile
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
    projected_ce_ascent, apply_saved_delta, measure_accuracy,
    configure_deterministic_runtime, runtime_provenance, file_identity,
    atomic_json_dump, sha256_file, collect_pe_groups
)
from ads_canonical_operator_v2_1 import measure_canonical_ads, CANONICAL_OPERATOR_HASH, assert_operator_lock
from canonical_adaptive_engine_v2_7 import (
    adaptive_projected_ascent, build_clean_logits_cache, assert_grad_logits_path_exact,
    target_value_no_grad
)

EXECUTION="ADS_TIFS_CANONICAL_ADAPTIVE_PILOT_v2_7_1_20260910"
PROTOCOL_ID="ADS_TIFS_CANONICAL_ADAPTIVE_PROTOCOL_LOCK_v2_7_20260910"
LOCK_SHA="1b558a44b947a0c3e74525e4acfc713c49f4757e13b5bd622512e3d11c4f251f"
EXPECTED_OPERATOR="093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362"
EXPECTED_REF_SHA="1dda480be595238fdbb1fb3d07898251862ab96fa72c235e5061a28c69981509"
EXPECTED_REF_CACHE_SHA="01bfd5a5d6e5e2f452e31e737550b9591d25a01c7979c1f82cb3483585537dc4"
EXPECTED_FULL_CACHE_SHA="90d47ea5212f5d06b9e8d12fa80ccb20cc722a46d79d2c926ef41be2917f0f71"
EXPECTED_HOLDOUT_SHA="835af1a98d05ac4f99cbb697f36d456b0a24bdd6905ada0ebac5da4eec0916a7"
EXPECTED_SOURCE_AGG_SHA="dcd2fa22d27a9a983dfadca3d49b56f776d316add01d201a2fe5563d757203d1"
EXPECTED_SOURCE_RESULTS_ZIP_SHA="ac40ded10f58d02d7b2b0eab7627f58b26c29569b330364a89d94aac31e0988b"

PE_CFG={
    "learned":{"seed":42,"epsilon":0.2},
    "rope":{"seed":42,"epsilon":0.5},
}
OBJECTIVES=("l4","profile_lse")
LAMBDAS=(0.0,1.0,10.0,50.0)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

class CachedTensorView:
    def __init__(self,images,labels,batch_size):
        if images.device!=labels.device: raise ValueError("cache device mismatch")
        self.images=images; self.labels=labels; self.batch_size=int(batch_size)
        self.dataset=range(int(labels.shape[0]))
    def __len__(self): return math.ceil(len(self.dataset)/self.batch_size)
    def __iter__(self):
        for s in range(0,len(self.dataset),self.batch_size):
            e=min(s+self.batch_size,len(self.dataset))
            yield self.images[s:e],self.labels[s:e]

def tensor_bytes(t):
    return t.detach().contiguous().cpu().numpy().tobytes(order="C")

def hash_stream(images,labels,batch_size):
    h=hashlib.sha256()
    for s in range(0,int(labels.shape[0]),batch_size):
        e=min(s+batch_size,int(labels.shape[0]))
        h.update(tensor_bytes(images[s:e])); h.update(tensor_bytes(labels[s:e]))
    return h.hexdigest()

def eps_tag(x): return f"{x:.6g}".replace(".","p").replace("-","m")
def lam_tag(x): return f"{x:.6g}".replace(".","p").replace("-","m")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--models_dir",required=True)
    p.add_argument("--val_dir",required=True)
    p.add_argument("--package_root",required=True)
    p.add_argument("--output_root",required=True)
    p.add_argument("--num_workers",type=int,default=0)
    p.add_argument("--attack_batch_size",type=int,default=32)
    p.add_argument("--measurement_batch_size",type=int,default=8)
    p.add_argument("--validation_batch_size",type=int,default=128)
    p.add_argument("--cache_load_batch_size",type=int,default=128)
    return p.parse_args()

def load_clean(models_dir,pe,seed,expected_ck_sha):
    ck=Path(models_dir)/f"{pe}_seed{seed}"/"best_model.pth"
    if not ck.exists(): raise FileNotFoundError(ck)
    got=sha256_file(str(ck))
    if got!=expected_ck_sha:
        raise RuntimeError(f"checkpoint SHA mismatch {pe} seed{seed}: {got} != {expected_ck_sha}")
    m=VisionTransformer(img_size=224,patch_size=16,num_classes=100,embed_dim=768,depth=12,
                        num_heads=12,mlp_ratio=4.0,dropout=0.1,pe_type=pe)
    st=torch.load(ck,map_location="cpu")
    if isinstance(st,dict) and "model_state_dict" in st: st=st["model_state_dict"]
    st={k.replace("_orig_mod.",""):v for k,v in st.items()}
    m.load_state_dict(st,strict=True)
    m.eval().to(DEVICE)
    return m, file_identity(str(ck))

def build_caches(a,ref_indices,hold_indices):
    if DEVICE!="cuda":
        raise RuntimeError("LOCK-003 FMLE pilot requires CUDA")
    if a.num_workers!=0 or a.attack_batch_size!=32 or a.measurement_batch_size!=8 or a.cache_load_batch_size!=128:
        raise RuntimeError("locked runtime requires workers=0, attack batch=32, measurement batch=8, cache batch=128")
    tf=transforms.Compose([
        transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds=datasets.ImageFolder(a.val_dir,tf)
    if len(ds)!=5000: raise RuntimeError(f"expected 5000 validation images, got {len(ds)}")
    if set(ref_indices)&set(hold_indices): raise RuntimeError("reference/holdout overlap")
    source=DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,pin_memory=True)
    need=int(len(ds)*3*224*224*4 + len(ds)*8)
    free,total=torch.cuda.mem_get_info()
    if free < need + 8*1024**3:
        raise RuntimeError("insufficient GPU memory for validation cache + 8 GiB reserve")
    X=torch.empty((5000,3,224,224),device=DEVICE,dtype=torch.float32)
    Y=torch.empty((5000,),device=DEVICE,dtype=torch.long)
    h=hashlib.sha256(); pos=0
    print("GPU CACHE: decoding/transforming validation set once",flush=True)
    for bi,(x,y) in enumerate(source,1):
        h.update(x.contiguous().numpy().tobytes(order="C"))
        h.update(y.contiguous().numpy().tobytes(order="C"))
        e=pos+int(y.numel())
        X[pos:e].copy_(x.to(DEVICE,non_blocking=True)); Y[pos:e].copy_(y.to(DEVICE,non_blocking=True)); pos=e
        if bi==1 or bi==len(source) or bi%5==0:
            print(f"  cache batch {bi}/{len(source)} images={pos}/5000",flush=True)
    full_sha=h.hexdigest()
    if full_sha!=EXPECTED_FULL_CACHE_SHA:
        raise RuntimeError(f"full transformed cache SHA mismatch {full_sha}")
    ri=torch.tensor(ref_indices,device=DEVICE,dtype=torch.long)
    hi=torch.tensor(hold_indices,device=DEVICE,dtype=torch.long)
    XR=X.index_select(0,ri).contiguous(); YR=Y.index_select(0,ri).contiguous()
    XH=X.index_select(0,hi).contiguous(); YH=Y.index_select(0,hi).contiguous()
    ref_sha=hash_stream(XR,YR,32)
    if ref_sha!=EXPECTED_REF_CACHE_SHA:
        raise RuntimeError(f"transformed reference SHA mismatch {ref_sha}")
    hold_sha=hash_stream(XH,YH,8)
    print("GPU CACHE PASS")
    print(" transformed full:",full_sha)
    print(" transformed ref :",ref_sha)
    print(" transformed hold:",hold_sha)
    return (
        CachedTensorView(X,Y,a.validation_batch_size),
        CachedTensorView(XR,YR,32),
        CachedTensorView(XR,YR,8),
        CachedTensorView(XH,YH,8),
        {"full_validation_sha256":full_sha,"reference_sha256":ref_sha,"holdout_runtime_sha256":hold_sha}
    )

def load_source_bundle(pkg):
    sroot=pkg/"source_exp014"
    agg=sroot/"ads_tifs_canonical_primary_full_grid_v2_1.json"
    if sha256_file(str(agg))!=EXPECTED_SOURCE_AGG_SHA:
        raise RuntimeError("embedded EXP-014 aggregate SHA mismatch")
    A=json.load(open(agg,"r",encoding="utf-8"))
    manifest=json.load(open(sroot/"SOURCE_CONTROL_MANIFEST.json","r",encoding="utf-8"))
    if manifest["source_aggregate_sha256"]!=EXPECTED_SOURCE_AGG_SHA:
        raise RuntimeError("source manifest aggregate drift")
    controls={}
    for pe,cfg in PE_CFG.items():
        tag=eps_tag(cfg["epsilon"])
        cp=sroot/"cells"/f"{pe}_seed42_eps{tag}.json"
        dp=sroot/"deltas"/f"imagenet_{pe}_seed42_eps{tag}_absolute_pgd20_ar0p1_v2_1.pt"
        npz=sroot/"per_image"/f"{pe}_seed42_eps{tag}_per_image_ads.npz"
        c=json.load(open(cp,"r",encoding="utf-8"))
        if sha256_file(str(dp))!=c["delta_artifact"]["sha256"]:
            raise RuntimeError("source delta SHA mismatch")
        if sha256_file(str(npz))!=c["per_image_artifact"]["sha256"]:
            raise RuntimeError("source per-image SHA mismatch")
        matches=[x for x in A["cells"] if x["metadata"]["pe_type"]==pe and int(x["metadata"]["seed"])==42
                 and abs(float(x["epsilon"])-float(cfg["epsilon"]))<1e-15]
        if len(matches)!=1 or matches[0]!=c:
            raise RuntimeError("source cell does not exactly match aggregate record")
        controls[pe]={"cell":c,"delta_path":dp,"per_image_path":npz}
    return controls

def delta_tensor_digest(artifact):
    h=hashlib.sha256()
    for g in artifact["groups"]:
        h.update(g["name"].encode("utf-8")+b"\0")
        t=g["delta"].detach().contiguous().cpu()
        h.update(str(tuple(t.shape)).encode()+b"\0")
        h.update(t.numpy().tobytes(order="C"))
    return h.hexdigest()

def compare_delta_artifacts_bitwise(path_a,path_b):
    A=torch.load(path_a,map_location="cpu",weights_only=False)
    B=torch.load(path_b,map_location="cpu",weights_only=False)
    ga={g["name"]:g for g in A["groups"]}; gb={g["name"]:g for g in B["groups"]}
    if set(ga)!=set(gb): return False,{"reason":"group-name mismatch"}
    details={}
    for n in sorted(ga):
        ta=ga[n]["delta"]; tb=gb[n]["delta"]
        eq=tuple(ta.shape)==tuple(tb.shape) and torch.equal(ta,tb)
        details[n]={"equal":bool(eq),"shape":list(ta.shape),
                     "max_abs_diff":float((ta.float()-tb.float()).abs().max().item()) if tuple(ta.shape)==tuple(tb.shape) else None}
        if not eq: return False,details
    return True,details

def profile_lse(vals,T=0.01):
    x=torch.tensor(vals,dtype=torch.float64)
    return float((T*torch.logsumexp(x/T,dim=0)).item())

def source_ref_matrix(path):
    z=np.load(path,allow_pickle=False)
    if "per_image_ads_canonical_logit_domain" not in z:
        raise RuntimeError("source per-image canonical key missing")
    m=z["per_image_ads_canonical_logit_domain"]
    if m.shape!=(256,12) or m.dtype!=np.float32 or not np.isfinite(m).all():
        raise RuntimeError("bad source per-image canonical matrix")
    return m

def main():
    a=parse_args()
    pkg=Path(a.package_root); out=Path(a.output_root)
    out.mkdir(parents=True,exist_ok=True)
    for d in ["cells","deltas","per_image","lambda0_regression"]:
        (out/d).mkdir(exist_ok=True)

    if sha256_file(str(pkg/"protocol_lock_v2_7/ADAPTIVE_PROTOCOL_LOCK_v2_7.json"))!=LOCK_SHA:
        raise RuntimeError("LOCK-003 JSON SHA mismatch")
    if sha256_file(str(pkg/"protocol_lock_v2_7/CANONICAL_REFERENCE_INDICES.json"))!=EXPECTED_REF_SHA:
        raise RuntimeError("reference index SHA mismatch")
    if sha256_file(str(pkg/"protocol_lock_v2_7/CANONICAL_ADAPTIVE_HOLDOUT_INDICES.json"))!=EXPECTED_HOLDOUT_SHA:
        raise RuntimeError("holdout index SHA mismatch")
    if assert_operator_lock()!=EXPECTED_OPERATOR or CANONICAL_OPERATOR_HASH!=EXPECTED_OPERATOR:
        raise RuntimeError("canonical operator hash mismatch")
    configure_deterministic_runtime(seed=42)

    ref_indices=json.load(open(pkg/"protocol_lock_v2_7/CANONICAL_REFERENCE_INDICES.json","r"))
    hold_indices=json.load(open(pkg/"protocol_lock_v2_7/CANONICAL_ADAPTIVE_HOLDOUT_INDICES.json","r"))
    if len(ref_indices)!=256 or len(set(ref_indices))!=256:
        raise RuntimeError("bad reference cohort")
    if len(hold_indices)!=256 or len(set(hold_indices))!=256 or set(ref_indices)&set(hold_indices):
        raise RuntimeError("bad holdout cohort")

    controls=load_source_bundle(pkg)
    val_view,attack_ref,measure_ref,measure_hold,cache_meta=build_caches(a,ref_indices,hold_indices)
    runtime=runtime_provenance(seed=42)
    cells=[]; lambda0_checks={}; implementation_checks={}

    for pe,cfg in PE_CFG.items():
        print("\n"+"#"*96); print(f"PILOT PE={pe} seed=42 eps={cfg['epsilon']}"); print("#"*96,flush=True)
        src=controls[pe]; sc=src["cell"]
        clean,ckid=load_clean(a.models_dir,pe,42,sc["metadata"]["checkpoint"]["sha256"])
        configure_deterministic_runtime(seed=42)

        # Gradient path == canonical measurement path, bitwise, before attacks.
        first_images=next(iter(measure_ref))[0]
        implementation_checks[pe]=assert_grad_logits_path_exact(clean,first_images,tuple(range(12)))
        print("CANONICAL GRAD-LOGITS PATH EXACT: PASS",flush=True)

        # Build exact all-layer clean-logits cache once for both objectives.
        print("Building clean canonical logits cache (12 layers x 256 ref images) ...",flush=True)
        clean_cache=build_clean_logits_cache(clean,measure_ref,DEVICE)
        print("CLEAN LOGITS CACHE PASS",flush=True)

        # lambda=0 implementation regression: regenerate via exact EXP-014 CE engine.
        reg_delta=out/"lambda0_regression"/f"{pe}_seed42_eps{eps_tag(cfg['epsilon'])}_lambda0_regenerated.pt"
        regen,regen_meta=projected_ce_ascent(
            clean,pe,"pe_only",cfg["epsilon"],attack_ref,DEVICE,
            steps=20,alpha_ratio=0.1,delta_convention="per_buffer",include_biases=False,
            budget_mode="absolute",delta_save_path=str(reg_delta),delta_storage_dtype="float32",
            progress_label=f"PILOT lambda0 regression {pe}",
            artifact_protocol_version=PROTOCOL_ID,
            artifact_execution_implementation=EXECUTION+"_LAMBDA0_REGRESSION"
        )
        eq,detail=compare_delta_artifacts_bitwise(reg_delta,src["delta_path"])
        if not eq:
            raise RuntimeError(f"lambda0 bitwise regression FAILED for {pe}: {detail}")
        reg_acc=measure_accuracy(regen,val_view,DEVICE)
        if abs(float(reg_acc)-float(sc["full_validation_accuracy"]))>1e-12:
            raise RuntimeError(f"lambda0 full-val accuracy regression failed {pe}: {reg_acc} vs {sc['full_validation_accuracy']}")
        lambda0_checks[pe]={
            "bitwise_delta_equal":True,
            "source_delta_sha256":sha256_file(str(src["delta_path"])),
            "regenerated_delta_sha256":sha256_file(str(reg_delta)),
            "source_delta_tensor_digest":delta_tensor_digest(torch.load(src["delta_path"],map_location="cpu",weights_only=False)),
            "regenerated_delta_tensor_digest":delta_tensor_digest(torch.load(reg_delta,map_location="cpu",weights_only=False)),
            "group_detail":detail,
            "full_validation_accuracy_exact":float(reg_acc),
        }
        del regen; torch.cuda.empty_cache()
        print("LAMBDA0 EXP-014 BITWISE REGRESSION: PASS",flush=True)

        # Exact source control is used for logical lambda=0 rows.
        src_model,src_art=apply_saved_delta(clean,pe,"pe_only",str(src["delta_path"]),DEVICE,delta_convention="per_buffer")
        source_ref=measure_canonical_ads(clean,src_model,measure_ref,DEVICE,
            per_image_path=str(out/"per_image"/f"{pe}_seed42_eps{eps_tag(cfg['epsilon'])}_lambda0_ref.npz"),
            progress_label=f"source lambda0 {pe} ref",retain_compat=False,retain_legacy=False)
        source_hold=measure_canonical_ads(clean,src_model,measure_hold,DEVICE,
            per_image_path=str(out/"per_image"/f"{pe}_seed42_eps{eps_tag(cfg['epsilon'])}_lambda0_hold.npz"),
            progress_label=f"source lambda0 {pe} hold",retain_compat=False,retain_legacy=False)
        src_ref_matrix=source_ref_matrix(src["per_image_path"])
        new_ref_matrix=np.load(out/"per_image"/f"{pe}_seed42_eps{eps_tag(cfg['epsilon'])}_lambda0_ref.npz",allow_pickle=False)["per_image_ads_canonical_logit_domain"]
        # Source NPZ is float32 storage; exact storage-level equality is required.
        if not np.array_equal(src_ref_matrix,new_ref_matrix):
            raise RuntimeError(f"source lambda0 independent ref per-image regression failed for {pe}")
        source_control={
            "full_validation_accuracy":float(sc["full_validation_accuracy"]),
            "clean_full_validation_accuracy":float(sc["clean_full_validation_accuracy"]),
            "accuracy_drop_pp":float(sc["accuracy_drop_pp"]),
            "ref_ads":source_ref,
            "holdout_ads":source_hold,
            "delta_artifact":file_identity(str(src["delta_path"])),
            "source_cell":file_identity(str(pkg/"source_exp014/cells"/f"{pe}_seed42_eps{eps_tag(cfg['epsilon'])}.json")),
        }

        for objective in OBJECTIVES:
            # logical lambda0 row (source reuse)
            c0={
                "metadata":{"execution":EXECUTION,"protocol":PROTOCOL_ID,"lock_sha256":LOCK_SHA,
                    "operator_hash":EXPECTED_OPERATOR,"reference_index_sha256":EXPECTED_REF_SHA,
                    "holdout_index_sha256":EXPECTED_HOLDOUT_SHA,"pe_type":pe,"seed":42,
                    "objective_kind":objective,"scientific_role":"implementation_pilot_source_control",
                    "source_reuse":"EXP-014 exact lambda0"},
                "epsilon":float(cfg["epsilon"]),"lambda":0.0,
                **source_control,
            }
            cp=out/"cells"/f"{pe}_seed42_{objective}_eps{eps_tag(cfg['epsilon'])}_lambda0.json"
            atomic_json_dump(str(cp),c0); cells.append(c0)

            for lam in (1.0,10.0,50.0):
                print("\n"+"="*88)
                print(f"{pe} objective={objective} eps={cfg['epsilon']} lambda={lam}")
                print("="*88,flush=True)
                dp=out/"deltas"/f"{pe}_seed42_{objective}_eps{eps_tag(cfg['epsilon'])}_lambda{lam_tag(lam)}.pt"
                attacked,attack=adaptive_projected_ascent(
                    clean,pe,cfg["epsilon"],lam,objective,
                    attack_ref,measure_ref,clean_cache,DEVICE,str(dp),
                    protocol_version=PROTOCOL_ID,execution_implementation=EXECUTION,
                    steps=20,alpha_ratio=0.1,lse_temperature=0.01
                )
                # Positive lambda must differ from source lambda0.
                neq,detail_same=compare_delta_artifacts_bitwise(dp,src["delta_path"])
                if neq:
                    raise RuntimeError(f"positive lambda unexpectedly bitwise identical to lambda0: {pe} {objective} {lam}")

                acc=measure_accuracy(attacked,val_view,DEVICE)
                ref_npz=out/"per_image"/f"{pe}_seed42_{objective}_eps{eps_tag(cfg['epsilon'])}_lambda{lam_tag(lam)}_ref.npz"
                hold_npz=out/"per_image"/f"{pe}_seed42_{objective}_eps{eps_tag(cfg['epsilon'])}_lambda{lam_tag(lam)}_hold.npz"
                ref_ads=measure_canonical_ads(clean,attacked,measure_ref,DEVICE,per_image_path=str(ref_npz),
                    progress_label=f"{pe} {objective} lambda{lam} ref",retain_compat=False,retain_legacy=False)
                hold_ads=measure_canonical_ads(clean,attacked,measure_hold,DEVICE,per_image_path=str(hold_npz),
                    progress_label=f"{pe} {objective} lambda{lam} hold",retain_compat=False,retain_legacy=False)

                # Completely independent final-state target recomputation.
                target2,layer_means2=target_value_no_grad(attacked,clean_cache,measure_ref,DEVICE,objective,lse_temperature=0.01)
                canon=ref_ads["canonical"]
                independent=float(canon["layer4_ads"]) if objective=="l4" else profile_lse(canon["per_layer_ads"],0.01)
                agreement=abs(float(target2)-independent)
                if agreement>1e-9:
                    raise RuntimeError(f"canonical objective/measurement target disagreement {agreement} > 1e-9")

                c={
                    "metadata":{"execution":EXECUTION,"protocol":PROTOCOL_ID,"lock_sha256":LOCK_SHA,
                        "operator_hash":EXPECTED_OPERATOR,"reference_index_sha256":EXPECTED_REF_SHA,
                        "holdout_index_sha256":EXPECTED_HOLDOUT_SHA,"pe_type":pe,"seed":42,
                        "objective_kind":objective,"scientific_role":"implementation_pilot_positive_lambda",
                        "pilot_not_manuscript_evidence":True},
                    "epsilon":float(cfg["epsilon"]),"lambda":float(lam),
                    "clean_full_validation_accuracy":float(sc["clean_full_validation_accuracy"]),
                    "full_validation_accuracy":float(acc),
                    "accuracy_drop_pp":float(sc["clean_full_validation_accuracy"]-acc),
                    "attack":attack,
                    "ref_ads":ref_ads,
                    "holdout_ads":hold_ads,
                    "objective_target_recomputed":float(target2),
                    "objective_target_independent_measurement":float(independent),
                    "objective_measurement_abs_difference":float(agreement),
                    "positive_lambda_delta_differs_from_lambda0":True,
                    "delta_artifact":file_identity(str(dp)),
                }
                cp=out/"cells"/f"{pe}_seed42_{objective}_eps{eps_tag(cfg['epsilon'])}_lambda{lam_tag(lam)}.json"
                atomic_json_dump(str(cp),c); cells.append(c)
                print(f"PILOT CELL PASS: acc={acc:.4f} target={independent:.9f} agreement={agreement:.3e}",flush=True)
                del attacked; torch.cuda.empty_cache()

        del src_model,clean_cache,clean
        torch.cuda.empty_cache()

    if len(cells)!=16:
        raise RuntimeError(f"expected 16 logical pilot cells, got {len(cells)}")

    # Final fail-closed implementation audit.
    positives=[c for c in cells if float(c["lambda"])>0]
    gates={
        "logical_cells_16":len(cells)==16,
        "positive_cells_12":len(positives)==12,
        "lambda0_bitwise_regression_both_pe":all(lambda0_checks[p]["bitwise_delta_equal"] for p in PE_CFG),
        "grad_logits_path_exact_both_pe":all(all(v==0.0 for v in implementation_checks[p].values()) for p in PE_CFG),
        "positive_delta_differs_from_lambda0":all(c["positive_lambda_delta_differs_from_lambda0"] for c in positives),
        "objective_measurement_agreement_le_1e9":all(c["objective_measurement_abs_difference"]<=1e-9 for c in positives),
        "projection_valid":all(float(c["attack"]["attack_geometry"]["max_delta_linf"])<=float(c["epsilon"])+1e-7 for c in positives),
        "ref_holdout_shapes_finite":True,
    }
    # NPZ integrity/shape.
    for p in (out/"per_image").glob("*.npz"):
        z=np.load(p,allow_pickle=False)
        arr=z["per_image_ads_canonical_logit_domain"]
        if arr.shape!=(256,12) or arr.dtype!=np.float32 or not np.isfinite(arr).all():
            gates["ref_holdout_shapes_finite"]=False
            break

    status="PASS_IMPLEMENTATION_PILOT" if all(gates.values()) else "FAIL_IMPLEMENTATION_PILOT"
    audit={
        "status":status,"execution":EXECUTION,"protocol":PROTOCOL_ID,"lock_sha256":LOCK_SHA,
        "operator_hash":EXPECTED_OPERATOR,"reference_index_sha256":EXPECTED_REF_SHA,
        "holdout_index_sha256":EXPECTED_HOLDOUT_SHA,"cache_runtime":cache_meta,
        "runtime":runtime,"gates":gates,"lambda0_regression":lambda0_checks,
        "grad_logits_path_check":implementation_checks,
        "logical_cells":len(cells),"positive_lambda_cells":len(positives),
        "note":"Implementation pilot only. No pilot effect size is manuscript evidence."
    }
    atomic_json_dump(str(out/"FINAL_PILOT_AUDIT.json"),audit)
    agg={
        "metadata":{"status":status,"execution":EXECUTION,"protocol":PROTOCOL_ID,
            "lock_sha256":LOCK_SHA,"operator_hash":EXPECTED_OPERATOR,
            "pilot_not_manuscript_evidence":True},
        "audit":audit,"cells":cells
    }
    atomic_json_dump(str(out/"ads_tifs_canonical_adaptive_pilot_v2_7_1.json"),agg)
    with open(out/"PILOT_DECISION.txt","w",encoding="utf-8") as f:
        f.write(f"{status}\n")
        for k,v in gates.items(): f.write(f"{k}: {v}\n")
        f.write("If PASS: audit results and update Scientific Results and Interpretation Ledger before confirmatory execution.\n")
    print("\nFINAL PILOT STATUS:",status,flush=True)
    if status!="PASS_IMPLEMENTATION_PILOT":
        raise SystemExit(2)

if __name__=="__main__":
    main()
