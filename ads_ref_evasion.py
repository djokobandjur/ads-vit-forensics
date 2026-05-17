"""
Reference Set Evasion Experiment
==================================
Tests whether an attacker who knows the reference set indices can craft
a PE perturbation that:
  - Preserves attention on the 256 reference images (evades ADS)
  - Degrades accuracy on the full validation set

Attack loss:
    L_evasion = -L_CE(logits_val, y) + lambda * ADS(delta, ref_images)

where ref_images are the known reference set.

We compare:
  1. Standard PGD (unknown ref set) -> high ADS on ref, high acc drop
  2. Evasion PGD (known ref set) -> low ADS on ref, but does it still drop acc?

Key question: Does evasion of a known reference set generalize to unseen images?
If the attacker evades ADS on ref but not on held-out images -> ADS is still useful
If evasion generalizes -> reference set secrecy is required

Output: ads_ref_evasion.json
"""

import os, sys, json, copy
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR      = '/content/drive/MyDrive/pe_experiment/results'
DATA_DIR         = '/content/imagenet100'
SAVE_PATH        = '/content/drive/MyDrive/pe_experiment/results/ADS/ads_ref_evasion.json'
REF_INDICES_PATH = '/content/drive/MyDrive/pe_experiment/results/ADS/ads_ref_indices.json'

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

PE_TYPES  = ['learned', 'rope']
SEEDS     = [42, 123, 456]
EPSILONS  = [0.1, 0.2, 0.5]
LAMBDAS   = [0.0, 1.0, 5.0, 10.0, 50.0]  # evasion regularization

PGD_STEPS       = 20
PGD_ALPHA_RATIO = 0.1

# ============================================================
# DATA
# ============================================================
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transform)

with open(REF_INDICES_PATH) as f:
    ref_indices = json.load(f)

# Reference loader (known to attacker)
ref_loader = DataLoader(
    Subset(val_dataset, ref_indices),
    batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# Held-out loader (unseen by attacker — different 256 images)
all_indices = list(range(len(val_dataset)))
non_ref = [i for i in all_indices if i not in set(ref_indices)]
torch.manual_seed(999)
holdout_indices = torch.randperm(len(non_ref))[:256].tolist()
holdout_indices = [non_ref[i] for i in holdout_indices]
holdout_loader = DataLoader(
    Subset(val_dataset, holdout_indices),
    batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# Full val loader for accuracy
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

print(f"Reference set: {len(ref_indices)} images (known to attacker)")
print(f"Holdout set:   {len(holdout_indices)} images (unseen by attacker)")

# ============================================================
# MODEL
# ============================================================
def load_model(pe_type, seed):
    model = VisionTransformer(
        img_size=224, patch_size=16, num_classes=100,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
    )
    ckpt = os.path.join(RESULTS_DIR, f'{pe_type}_seed{seed}', 'best_model.pth')
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
    model.eval().to(device)
    return model

# ============================================================
# ATTENTION + ADS
# ============================================================
@torch.no_grad()
def get_attn_l4(model, loader):
    layers = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, attns = model.forward_with_attention(imgs)
        layers.append(attns[3].mean(0).cpu())
    return torch.stack(layers).mean(0)

def compute_ads(clean_attn, perturbed_attn):
    eps = 1e-10
    P = clean_attn.float() + eps
    Q = perturbed_attn.float() + eps
    P = P / P.sum(-1, keepdim=True)
    Q = Q / Q.sum(-1, keepdim=True)
    return float((P * torch.log(P / Q)).sum(-1).mean().item())

# ============================================================
# ACCURACY
# ============================================================
@torch.no_grad()
def measure_accuracy(model, loader):
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, pred = model(imgs).max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

# ============================================================
# PE PARAMS
# ============================================================
def get_pe_params(model, pe_type):
    pe_params = []
    for name, buf in list(model.named_buffers()):
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['.pe', 'cos_cached', 'sin_cached']):
            if 'rel_dist' not in clean:
                parts = clean.split('.')
                obj = model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                delattr(obj, parts[-1])
                new_p = nn.Parameter(buf.clone(), requires_grad=True)
                setattr(obj, parts[-1], new_p)
                pe_params.append(new_p)
    for name, mod in model.named_modules():
        if type(mod).__name__ == 'ALiBi':
            sd = mod.slopes.clone().to(device)
            del mod.slopes
            mod.slopes = nn.Parameter(sd, requires_grad=True)
            pe_params.append(mod.slopes)
    for name, param in model.named_parameters():
        clean = name.replace('_orig_mod.', '')
        if any(k in clean for k in ['pos_embed', 'inv_freq']):
            param.requires_grad_(True)
            pe_params.append(param)
    return pe_params

# ============================================================
# EVASION PGD
# ============================================================
def evasion_pgd(model, clean_ref_attn, pe_type, epsilon, lam):
    """
    Reference-set evasion attack.
    Loss: -L_CE(full_batch) + lambda * ADS(ref_images)
    
    The attacker knows ref_indices and tries to minimize ADS on those
    specific images while maximizing accuracy damage on the full val set.
    """
    pm = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    alpha = epsilon * PGD_ALPHA_RATIO

    # Attack batch: use ref images for gradient (attacker knows these)
    images, labels = next(iter(ref_loader))
    images, labels = images.to(device), labels.to(device)

    pe_params = get_pe_params(pm, pe_type)
    if not pe_params:
        return pm

    deltas = [torch.zeros_like(p) for p in pe_params]
    pm.train()

    for step in range(PGD_STEPS):
        with torch.no_grad():
            for p, d in zip(pe_params, deltas):
                p.data = p.data - (d if step > 0 else 0) + d

        for p in pe_params:
            if p.grad is not None:
                p.grad.zero_()

        # CE loss on ref batch (damage)
        ce_loss = criterion(pm(images), labels)

        if lam > 0:
            # ADS minimization on ref images (evasion)
            _, attns = pm.forward_with_attention(images)
            attn_pert = attns[3].mean(0).float() + 1e-10
            attn_pert = attn_pert / attn_pert.sum(-1, keepdim=True)
            attn_pert = attn_pert.to(device)

            P = clean_ref_attn.detach().to(device).float() + 1e-10
            P = P / P.sum(-1, keepdim=True)

            ads_loss = (P * torch.log(P / attn_pert)).sum(-1).mean()
            total_loss = -ce_loss + lam * ads_loss
        else:
            total_loss = -ce_loss

        total_loss.backward()

        with torch.no_grad():
            new_d = []
            for p, d in zip(pe_params, deltas):
                if p.grad is not None:
                    nd = (d + alpha * p.grad.sign()).clamp(-epsilon, epsilon)
                    new_d.append(nd)
                else:
                    new_d.append(d)
            deltas = new_d

    with torch.no_grad():
        for p, d in zip(pe_params, deltas):
            p.data += d

    pm.eval()
    return pm

# ============================================================
# MAIN
# ============================================================
def run():
    results = {}
    
    # Clean noise floors
    noise_floor = {'learned': 0.015498, 'rope': 0.013538}

    for pe_type in PE_TYPES:
        results[pe_type] = {}
        print(f"\n{'='*60}")
        print(f"PE TYPE: {pe_type.upper()}")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            model = load_model(pe_type, seed)

            # Baseline attention on both sets
            clean_ref_attn  = get_attn_l4(model, ref_loader)
            clean_hold_attn = get_attn_l4(model, holdout_loader)
            clean_acc = measure_accuracy(model, val_loader)
            nf = noise_floor[pe_type]

            results[pe_type][str(seed)] = {}

            for eps in EPSILONS:
                results[pe_type][str(seed)][str(eps)] = {}
                print(f"\n    ε={eps}:")
                print(f"    {'λ':>6}  {'Acc drop':>10}  {'ADS(ref)':>10}  {'ADS(hold)':>10}  {'Evasion?':>10}  {'Generalized?':>13}")

                for lam in LAMBDAS:
                    pm = evasion_pgd(model, clean_ref_attn, pe_type, eps, lam)

                    # ADS on reference set (what attacker is evading)
                    pm_ref_attn  = get_attn_l4(pm, ref_loader)
                    ads_ref = compute_ads(clean_ref_attn, pm_ref_attn)

                    # ADS on held-out set (unseen by attacker)
                    pm_hold_attn = get_attn_l4(pm, holdout_loader)
                    ads_hold = compute_ads(clean_hold_attn, pm_hold_attn)

                    # Accuracy
                    acc = measure_accuracy(pm, val_loader)
                    acc_drop = clean_acc - acc

                    # Detection threshold (from benign analysis)
                    det_thresh = {'learned': 2.65, 'rope': 5.01}[pe_type]
                    evades_ref  = (ads_ref  / nf) < det_thresh
                    evades_hold = (ads_hold / nf) < det_thresh

                    results[pe_type][str(seed)][str(eps)][str(lam)] = {
                        'lambda': lam,
                        'acc_drop': float(acc_drop),
                        'ads_ref':  float(ads_ref),
                        'ads_hold': float(ads_hold),
                        'ads_ref_ratio':  float(ads_ref / nf),
                        'ads_hold_ratio': float(ads_hold / nf),
                        'evades_ref':  bool(evades_ref),
                        'evades_hold': bool(evades_hold),
                    }

                    print(f"    {lam:>6.0f}  {acc_drop:>10.2f}pp  "
                          f"{ads_ref/nf:>8.2f}x  {ads_hold/nf:>9.2f}x  "
                          f"{'YES' if evades_ref else 'NO':>10}  "
                          f"{'YES' if evades_hold else 'NO':>13}")

                    del pm
                    torch.cuda.empty_cache()

            del model
            torch.cuda.empty_cache()

        with open(SAVE_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {SAVE_PATH}")

    return results

# ============================================================
# ANALYSIS
# ============================================================
def analyze(results):
    seeds = ['42', '123', '456']
    print("\n" + "=" * 75)
    print("REFERENCE SET EVASION ANALYSIS")
    print("=" * 75)
    print("Key question: If attacker evades ref set ADS, does it generalize?")

    for pe_type in PE_TYPES:
        print(f"\n{pe_type.upper()}:")
        for eps in EPSILONS:
            print(f"\n  ε={eps}:")
            for lam in LAMBDAS:
                drops = [results[pe_type][s][str(eps)][str(lam)]['acc_drop'] for s in seeds]
                ev_ref  = [results[pe_type][s][str(eps)][str(lam)]['evades_ref'] for s in seeds]
                ev_hold = [results[pe_type][s][str(eps)][str(lam)]['evades_hold'] for s in seeds]
                r_ref  = [results[pe_type][s][str(eps)][str(lam)]['ads_ref_ratio'] for s in seeds]
                r_hold = [results[pe_type][s][str(eps)][str(lam)]['ads_hold_ratio'] for s in seeds]

                evades = all(ev_ref)
                generalizes = all(ev_hold)
                print(f"    λ={lam:4.0f}: drop={np.mean(drops):.1f}pp, "
                      f"ref={np.mean(r_ref):.2f}x ({'evades' if evades else 'detected'}), "
                      f"hold={np.mean(r_hold):.2f}x ({'evades' if generalizes else 'detected'})")


if __name__ == '__main__':
    print("Reference Set Evasion Experiment")
    print(f"PE types: {PE_TYPES}, Seeds: {SEEDS}")
    print(f"Epsilons: {EPSILONS}, Lambdas: {LAMBDAS}")
    print(f"Reference: 256 known images, Holdout: 256 unseen images")
    print(f"Estimated time: ~1.5h on A100")
    print()

    results = run()
    analyze(results)
    print("\n✅ Done!")
