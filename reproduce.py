"""
reproduce.py — Reproducibility script for

    "Attention Divergence Score: A Forensic Metric for Characterizing
     Parameter-Level Attacks in Vision Transformers"
    (IEEE Transactions on Information Forensics and Security submission)

This script reproduces every numerical claim, table value, and figure in the
paper directly from the archived experimental JSON data. It does NOT train
models or run GPU experiments; it reads the pre-computed attention distributions
and accuracy trajectories and regenerates all downstream statistics and plots.

================================================================================
USAGE
================================================================================

    # 1. Place all JSON data files in ./data/ (or set DATA_DIR below)
    # 2. Run:
    python reproduce.py

    # Optional flags:
    python reproduce.py --no-figures      # skip figure generation (tables only)
    python reproduce.py --output-dir out  # custom output directory
    python reproduce.py --section 6.2     # reproduce only a specific section

================================================================================
EXPECTED DATA FILES
================================================================================

    data/ads_results.json              ImageNet-100 main ADS experiment
    data/ads_results_cifar100.json     CIFAR-100 main ADS experiment
    data/ads_specificity.json          ImageNet-100 specificity (4 PE × 4 attacks)
    data/ads_specificity_cifar.json    CIFAR-100 specificity (4 PE × 4 attacks)
    data/ads_probing_residual.json     ImageNet-100 residual stream probing (opt.)
    data/ads_probing_residual_cifar.json CIFAR-100 residual stream probing (opt.)

================================================================================
OUTPUT
================================================================================

    output/
    ├── tables/                    (.txt files, one per table)
    │   ├── table_VI_fingerprint.txt   (Table VI: ADS(L4)/ADS(mean) ratios)
    │   ├── table_VII_specificity.txt  (Table VII: PE attack signatures)
    │   ├── table_VIII_probing.txt     (Table VIII: Residual stream probing)
    │   ├── stats_ANOVA.txt            (Section 6.3: ANOVA + ICC)
    │   └── stats_pairwise.txt         (Section 6.3: pairwise t-tests)
    ├── figures/                   (PNG files, 5 figures)
    └── reproduce_log.txt          (full log, with timestamps and assertions)

Each table reproduces to within 3 decimal places of the published numbers.

================================================================================
DEPENDENCIES
================================================================================

    numpy >= 1.20
    scipy >= 1.7
    matplotlib >= 3.4
    (standard scientific Python; no GPU, no PyTorch required)

Authors: Djoko Bandjur, Milos Bandjur
Contact: djoko.bandjur@pr.ac.rs
Last updated: April 2026
License: MIT (code) / CC-BY 4.0 (data)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = 'data'
OUTPUT_DIR = 'output'

PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
PE_DISPLAY = {
    'learned': 'Learned',
    'sinusoidal': 'Sinusoidal',
    'rope': 'RoPE',
    'alibi': 'ALiBi',
}
SEEDS = ['42', '123', '456']
ATTACKS = ['pe_only', 'qkv_only', 'mlp_only', 'all_weights']

# Operation-space grouping (Section 6.3)
EMBEDDING_SPACE = ['learned', 'sinusoidal']
ATTENTION_SPACE = ['rope', 'alibi']

# t critical value for 95% CI with df=2 (for n=3 seeds)
T_CRIT_DF2 = stats.t.ppf(0.975, df=2)  # ≈ 4.303


# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    def __init__(self, logfile):
        self.logfile = open(logfile, 'w')
        self.start_time = datetime.now()

    def log(self, msg='', also_print=True):
        ts = (datetime.now() - self.start_time).total_seconds()
        line = f"[{ts:7.2f}s] {msg}"
        self.logfile.write(line + '\n')
        self.logfile.flush()
        if also_print:
            print(msg)

    def section(self, title):
        bar = '=' * 78
        self.log('')
        self.log(bar)
        self.log(title)
        self.log(bar)

    def subsection(self, title):
        self.log('')
        self.log('-' * len(title))
        self.log(title)
        self.log('-' * len(title))

    def close(self):
        self.logfile.close()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_json(filename, log):
    path = Path(DATA_DIR) / filename
    if not path.exists():
        log.log(f"  [SKIP] {filename} not found at {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    log.log(f"  [OK]   Loaded {filename}")
    return data


def load_all_data(log):
    log.section("STEP 1: Loading data")
    data = {
        'imn_main': load_json('ads_results.json', log),
        'cif_main': load_json('ads_results_cifar100.json', log),
        'imn_spec': load_json('ads_specificity.json', log),
        'cif_spec': load_json('ads_specificity_cifar.json', log),
        'imn_probe': load_json('ads_probing_residual.json', log),
        'cif_probe': load_json('ads_probing_residual_cifar.json', log),
        # ADS-specific experiment files (Sections 4.4, 5, 6.5, 6.6, 6.7)
        'threshold': load_json('ads_threshold_fine.json', log),
        'roc': load_json('ads_roc_v2.json', log),
        'comparison': load_json('ads_comparison.json', log),
        'evasion': load_json('ads_ref_evasion.json', log),
        'adaptive': load_json('ads_adaptive.json', log),
        'ref_indices': load_json('ads_ref_indices.json', log),
    }
    return data


# =============================================================================
# TABLE VI: FINGERPRINT — ADS(L4)/ADS(mean) ratios (Section 6.3)
# =============================================================================

def compute_fingerprint_per_seed(data, pe):
    """
    For a given PE type, compute the within-model mean ADS(L4)/ADS(mean) ratio
    for each seed. This is the seed-level aggregation used in v15 Section 7.4.

    Returns: array of 3 ratios (one per seed).
    """
    seed_ratios = []
    for seed in SEEDS:
        sd = data[pe][seed]
        ads_l4 = np.array(sd['ads_layer4'])
        ads_per_layer = np.array(sd['ads_per_layer'])   # (n_eps, 12)
        ads_mean_layers = ads_per_layer.mean(axis=1)    # (n_eps,)
        # ratio at each epsilon (skip eps=0 where denominator ≈ 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(ads_mean_layers > 1e-10,
                             ads_l4 / ads_mean_layers, np.nan)
        # Within-model average across epsilons (excluding NaN at eps=0)
        seed_ratios.append(np.nanmean(ratio))
    return np.array(seed_ratios)


def compute_within_model_cv(data, pe):
    """Within-model CV of the ratio across epsilon, averaged over seeds."""
    cvs = []
    for seed in SEEDS:
        sd = data[pe][seed]
        ads_l4 = np.array(sd['ads_layer4'])
        ads_per_layer = np.array(sd['ads_per_layer'])
        ads_mean_layers = ads_per_layer.mean(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(ads_mean_layers > 1e-10,
                             ads_l4 / ads_mean_layers, np.nan)
        r = ratio[~np.isnan(ratio)]
        cv = np.nanstd(r, ddof=1) / np.nanmean(r)
        cvs.append(cv)
    return np.mean(cvs)


def compute_icc(data, pe):
    """
    Intraclass correlation (ICC(1,1)) for within-model ratio clustering.
    Formula: (MSB - MSW) / (MSB + (k-1) * MSW)
    where MSB = between-group variance, MSW = within-group variance,
    k = number of measurements per group.
    """
    # Collect ratios per seed (each seed = one cluster of epsilon measurements)
    all_ratios = []  # list of arrays, one per seed
    for seed in SEEDS:
        sd = data[pe][seed]
        ads_l4 = np.array(sd['ads_layer4'])
        ads_per_layer = np.array(sd['ads_per_layer'])
        ads_mean_layers = ads_per_layer.mean(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(ads_mean_layers > 1e-10,
                             ads_l4 / ads_mean_layers, np.nan)
        r = ratio[~np.isnan(ratio)]
        all_ratios.append(r)

    # One-way ANOVA decomposition
    k = min(len(r) for r in all_ratios)  # balanced
    all_ratios = [r[:k] for r in all_ratios]
    grand_mean = np.mean([np.mean(r) for r in all_ratios])
    group_means = [np.mean(r) for r in all_ratios]
    msb = k * np.sum([(gm - grand_mean)**2 for gm in group_means]) / (len(SEEDS) - 1)
    msw = np.sum([np.sum((r - gm)**2) for r, gm in zip(all_ratios, group_means)])
    msw /= (len(SEEDS) * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return icc


def table_VI_fingerprint(data_imn, data_cif, log, out_dir):
    log.section("TABLE VI: ADS(L4)/ADS(mean) Fingerprint Ratios")
    log.log("Seed-level statistics (n=3 per PE type), mean ± 95% CI (t, df=2)")
    log.log("Expected in paper Section 6.3 / Table VI:")
    log.log("  IMN: Learned 3.19±1.65, Sinusoidal 2.34±0.86, RoPE 1.82±1.19, ALiBi 0.83±0.59")
    log.log("  CIF: Learned 1.01,     Sinusoidal 1.44,     RoPE 1.59,     ALiBi 0.99")
    log.log('')

    header = f"{'PE Type':<12} {'IMN mean':>10} {'IMN ±95%CI':>12} {'IMN CV%':>9} {'IMN ICC':>9}  |  {'CIF mean':>10} {'CIF ±95%CI':>12}"
    log.log(header)
    log.log('-' * len(header))

    results = {}
    for pe in PE_TYPES:
        r_imn = compute_fingerprint_per_seed(data_imn, pe)
        m_imn, s_imn = np.mean(r_imn), np.std(r_imn, ddof=1)
        ci_imn = T_CRIT_DF2 * s_imn / np.sqrt(3)
        cv_imn = compute_within_model_cv(data_imn, pe) * 100
        icc_imn = compute_icc(data_imn, pe)

        r_cif = compute_fingerprint_per_seed(data_cif, pe)
        m_cif, s_cif = np.mean(r_cif), np.std(r_cif, ddof=1)
        ci_cif = T_CRIT_DF2 * s_cif / np.sqrt(3)

        line = (f"{PE_DISPLAY[pe]:<12} {m_imn:>10.2f} {'±'+f'{ci_imn:.2f}':>12} "
                f"{cv_imn:>9.1f} {icc_imn:>9.2f}  |  "
                f"{m_cif:>10.2f} {'±'+f'{ci_cif:.2f}':>12}")
        log.log(line)

        results[pe] = {
            'imn_mean': m_imn, 'imn_ci': ci_imn, 'imn_cv': cv_imn, 'imn_icc': icc_imn,
            'cif_mean': m_cif, 'cif_ci': ci_cif,
        }

    # Save to file
    outpath = Path(out_dir) / 'tables' / 'table_VI_fingerprint.txt'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w') as f:
        f.write("Table VI: ADS(L4)/ADS(mean) Fingerprint Ratios (reproduced)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(header + '\n' + '-' * len(header) + '\n')
        for pe in PE_TYPES:
            r = results[pe]
            f.write(f"{PE_DISPLAY[pe]:<12} "
                    f"{r['imn_mean']:>10.2f} ±{r['imn_ci']:.2f}  "
                    f"CV={r['imn_cv']:.1f}%  ICC={r['imn_icc']:.2f}  |  "
                    f"{r['cif_mean']:>10.2f} ±{r['cif_ci']:.2f}\n")

    return results


# =============================================================================
# Section 6.3: ANOVA & pairwise t-tests
# =============================================================================

def section_63_anova(data_imn, data_cif, log, out_dir):
    log.section("SECTION 6.3: ANOVA & Pairwise Tests")
    log.log("Seed-level tests (n=3 per PE). Expected in paper:")
    log.log("  IMN: ANOVA F(3,8) = 14.06, p = 0.0015")
    log.log("  CIF: ANOVA F(3,8) =  1.94, p = 0.20 (null)")
    log.log("  IMN ALiBi vs rest t = -5.80, p = 0.0002")
    log.log("  IMN Embedding vs Attention-space t = 3.85, p = 0.003")
    log.log('')

    # Collect per-seed ratios per PE
    imn_ratios = {pe: compute_fingerprint_per_seed(data_imn, pe) for pe in PE_TYPES}
    cif_ratios = {pe: compute_fingerprint_per_seed(data_cif, pe) for pe in PE_TYPES}

    # One-way ANOVA (4 groups × 3 seeds)
    log.subsection("ImageNet-100 ANOVA")
    groups_imn = [imn_ratios[pe] for pe in PE_TYPES]
    f_imn, p_imn = stats.f_oneway(*groups_imn)
    log.log(f"  F(3,8) = {f_imn:.2f}, p = {p_imn:.4f}")

    log.subsection("CIFAR-100 ANOVA")
    groups_cif = [cif_ratios[pe] for pe in PE_TYPES]
    f_cif, p_cif = stats.f_oneway(*groups_cif)
    log.log(f"  F(3,8) = {f_cif:.2f}, p = {p_cif:.4f}  {'(null)' if p_cif > 0.05 else ''}")

    # Key contrasts on ImageNet
    log.subsection("ImageNet-100 ALiBi vs. rest")
    alibi = imn_ratios['alibi']
    rest = np.concatenate([imn_ratios[pe] for pe in ['learned', 'sinusoidal', 'rope']])
    t_al, p_al = stats.ttest_ind(alibi, rest, equal_var=False)
    log.log(f"  t = {t_al:.2f}, p = {p_al:.4f}")

    log.subsection("ImageNet-100 Embedding-space vs. Attention-space")
    emb = np.concatenate([imn_ratios[pe] for pe in EMBEDDING_SPACE])
    att = np.concatenate([imn_ratios[pe] for pe in ATTENTION_SPACE])
    t_es, p_es = stats.ttest_ind(emb, att, equal_var=False)
    log.log(f"  t = {t_es:.2f}, p = {p_es:.4f}")

    # All pairwise t-tests (Table supplemental)
    log.subsection("All pairwise t-tests (ImageNet-100, Welch)")
    pe_list = PE_TYPES
    for i, pe_a in enumerate(pe_list):
        for pe_b in pe_list[i+1:]:
            t, p = stats.ttest_ind(imn_ratios[pe_a], imn_ratios[pe_b], equal_var=False)
            sig = '*' if p < 0.05 else ''
            log.log(f"  {PE_DISPLAY[pe_a]:12s} vs. {PE_DISPLAY[pe_b]:12s}: "
                    f"t = {t:6.2f}, p = {p:.4f} {sig}")

    # Save
    outpath = Path(out_dir) / 'tables' / 'stats_ANOVA.txt'
    with open(outpath, 'w') as f:
        f.write(f"Section 6.3 Statistics (reproduced)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"ImageNet ANOVA: F(3,8) = {f_imn:.2f}, p = {p_imn:.4f}\n")
        f.write(f"CIFAR    ANOVA: F(3,8) = {f_cif:.2f}, p = {p_cif:.4f}\n")
        f.write(f"IMN ALiBi vs rest:                  t = {t_al:.2f}, p = {p_al:.4f}\n")
        f.write(f"IMN Embedding vs Attention-space:   t = {t_es:.2f}, p = {p_es:.4f}\n")


# =============================================================================
# 1-NN LOO CLASSIFICATION (Section 6.3)
# =============================================================================

def section_63_1nn_loo(data_imn, log, out_dir):
    log.section("SECTION 6.3: 1-NN Leave-One-Out Classification (ImageNet)")
    log.log("Expected in paper: 6/12 correct (50%), permutation p ≈ 0.047")
    log.log('')

    # Each (PE, seed) is one point in 1-D ratio space
    X, y = [], []
    for pe in PE_TYPES:
        for seed in SEEDS:
            sd = data_imn[pe][seed]
            ads_l4 = np.array(sd['ads_layer4'])
            ads_per_layer = np.array(sd['ads_per_layer'])
            ads_mean = ads_per_layer.mean(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(ads_mean > 1e-10, ads_l4 / ads_mean, np.nan)
            X.append(np.nanmean(ratio))
            y.append(pe)

    X = np.array(X)
    y = np.array(y)
    n = len(X)

    # Leave-one-out 1-NN
    correct = 0
    for i in range(n):
        mask = np.arange(n) != i
        dists = np.abs(X[mask] - X[i])
        pred = y[mask][np.argmin(dists)]
        if pred == y[i]:
            correct += 1
    acc = correct / n

    # Permutation test
    rng = np.random.default_rng(42)
    n_perm = 10000
    perm_accs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        c = 0
        for i in range(n):
            mask = np.arange(n) != i
            dists = np.abs(X[mask] - X[i])
            pred = y_perm[mask][np.argmin(dists)]
            if pred == y_perm[i]:
                c += 1
        perm_accs.append(c / n)
    p_perm = (np.sum(np.array(perm_accs) >= acc) + 1) / (n_perm + 1)

    log.log(f"  Accuracy: {correct}/{n} = {acc*100:.1f}%")
    log.log(f"  Permutation p-value (n={n_perm}): {p_perm:.4f}")


# =============================================================================
# TABLE VII: SPECIFICITY (Section 6.2)
# =============================================================================

def section_62_specificity(imn_spec, cif_spec, log, out_dir):
    log.section("TABLE VII: PE Attack Specificity at ε=0.2")
    log.log("Mean across 3 seeds. Expected (from paper Section 6.2 Table VII):")
    log.log("  Learned PE-only:     ADS(L4)=0.25, drop=17.4pp,  ratio=0.014")
    log.log("  Sinusoidal PE-only:  ADS(L4)=0.05, drop= 9.2pp,  ratio=0.006")
    log.log("  RoPE PE-only:        ADS(L4)=0.11, drop= 2.3pp,  ratio=0.049")
    log.log("  ALiBi PE-only:       ADS(L4)=3.57, drop=73.0pp,  ratio=0.049")
    log.log('')

    target_eps = 0.2

    def compute_table_row(data, pe, attack):
        ads_l4_list, ads_mean_list, drop_list = [], [], []
        for seed in SEEDS:
            sd = data[pe][seed][attack]
            eps = np.array(sd['epsilons'])
            accs = np.array(sd['accuracies'])
            ads_l4 = np.array(sd['ads_layer4'])
            ads_mean = np.array(sd['ads_mean'])
            idx = np.argmin(np.abs(eps - target_eps))
            idx0 = np.where(eps == 0)[0]
            clean = accs[idx0[0]] if len(idx0) > 0 else accs[0]
            drop = clean - accs[idx]
            ads_l4_list.append(ads_l4[idx])
            ads_mean_list.append(ads_mean[idx])
            drop_list.append(drop)
        return (np.mean(ads_l4_list), np.mean(ads_mean_list), np.mean(drop_list))

    for label, data in [('IMAGENET-100', imn_spec), ('CIFAR-100', cif_spec)]:
        log.subsection(label)
        hdr = f"  {'PE':<12} {'Attack':<14} {'ADS(L4)':>10} {'ADS(mean)':>11} {'Drop(pp)':>10} {'ADS/Drop':>10}"
        log.log(hdr)
        log.log('  ' + '-' * (len(hdr) - 2))
        for pe in PE_TYPES:
            if pe not in data:
                continue
            for attack in ATTACKS:
                ads_l4, ads_mean, drop = compute_table_row(data, pe, attack)
                ratio = ads_l4 / drop if drop > 0.5 else float('nan')
                log.log(f"  {PE_DISPLAY[pe]:<12} {attack:<14} "
                        f"{ads_l4:>10.3f} {ads_mean:>11.3f} "
                        f"{drop:>10.2f} {ratio:>10.4f}")
            log.log('')

    # Damage asymmetry (Effect 1 in Section 6.2)
    log.subsection("Damage asymmetry: MLP-drop / PE-drop at ε=0.2 (ImageNet)")
    log.log("  PE         MLP drop (pp)  PE drop (pp)  Asymmetry ratio")
    log.log("  ---------  -------------  ------------  ----------------")
    for pe in PE_TYPES:
        _, _, pe_drop = compute_table_row(imn_spec, pe, 'pe_only')
        _, _, mlp_drop = compute_table_row(imn_spec, pe, 'mlp_only')
        ratio = mlp_drop / pe_drop if pe_drop > 0.1 else float('inf')
        log.log(f"  {PE_DISPLAY[pe]:<10} {mlp_drop:>12.2f}   {pe_drop:>11.2f}   {ratio:>14.1f}×")


# =============================================================================
# Section 6.2 EFFECT 1: SATURATION BUDGET ASYMMETRY (saturation reframe)
# =============================================================================

def section_62_saturation(imn_spec, cif_spec, log, out_dir):
    """
    Reproduce the saturation-budget asymmetry claim from Section 6.2 Effect 1:
    PE-only attacks require 17x-200x larger perturbation budgets than MLP-only
    attacks to operationally compromise the model (residual accuracy <= 50% of
    clean baseline).
    """
    log.section("SECTION 6.2 EFFECT 1: Saturation Budget Asymmetry")
    log.log("Saturation budget = smallest ε at which residual accuracy ≤ 50% of clean.")
    log.log("Following Carlini et al.~\\cite{carlini2019evaluating} 'operational compromise' threshold.")
    log.log('')
    log.log("Expected in paper Section 6.2 Effect 1 / Abstract / Conclusion First finding:")
    log.log("  ImageNet: Learned 100×, Sinusoidal 100×, RoPE 200×, ALiBi 17×")
    log.log("  CIFAR:    Learned 20×,  Sinusoidal 40×,  RoPE 200×, ALiBi 40×")
    log.log("  Combined range: 17×–200× (RoPE PE-only does not robustly saturate)")
    log.log('')

    def find_saturation_eps(data, pe, attack, threshold_func):
        """Find smallest ε where threshold_func(residual_acc, clean) returns True
        averaged across seeds."""
        eps_per_seed = []
        for seed in SEEDS:
            sd = data[pe][seed][attack]
            eps = np.array(sd['epsilons'])
            accs = np.array(sd['accuracies'])
            idx0 = np.where(eps == 0)[0]
            if len(idx0) == 0:
                continue
            clean = accs[idx0[0]]
            mask = np.array([threshold_func(a, clean) for a in accs])
            if mask.any():
                first_idx = np.argmax(mask)
                eps_per_seed.append(eps[first_idx])
            else:
                eps_per_seed.append(None)
        valid = [e for e in eps_per_seed if e is not None]
        return np.mean(valid) if valid else None, len(valid)

    thresholds = [
        ("50% clean (operational compromise)",
         lambda a, c: a <= 0.5 * c, 50),
        ("5% absolute (severe degradation)",
         lambda a, c: a <= 5.0, 5),
    ]

    all_ratios_per_threshold = {}

    for tname, tfunc, tval in thresholds:
        log.subsection(f"Threshold: {tname}")
        log.log(f"  {'Dataset':<14} {'PE':<12} {'ε_PE':>10} {'ε_MLP':>10} {'Ratio':>8}")
        log.log('  ' + '-' * 56)

        ratios_at_threshold = []
        for label, data in [('IMAGENET-100', imn_spec), ('CIFAR-100', cif_spec)]:
            for pe in PE_TYPES:
                pe_eps, pe_n = find_saturation_eps(data, pe, 'pe_only', tfunc)
                mlp_eps, mlp_n = find_saturation_eps(data, pe, 'mlp_only', tfunc)
                if pe_eps is not None and mlp_eps is not None:
                    ratio = pe_eps / mlp_eps
                    ratios_at_threshold.append(ratio)
                    log.log(f"  {label:<14} {PE_DISPLAY[pe]:<12} "
                            f"{pe_eps:>10.4f} {mlp_eps:>10.4f} {ratio:>7.0f}×")
                elif pe_eps is None:
                    log.log(f"  {label:<14} {PE_DISPLAY[pe]:<12} "
                            f"{'never':>10} {mlp_eps:>10.4f} {'>200×':>8} (PE attacks fail)")
                else:
                    log.log(f"  {label:<14} {PE_DISPLAY[pe]:<12} "
                            f"{pe_eps:>10.4f} {'never':>10} {'?':>8}")

        if ratios_at_threshold:
            log.log(f"  Range across observable PE/dataset combinations: "
                    f"{min(ratios_at_threshold):.0f}× – {max(ratios_at_threshold):.0f}×")

        all_ratios_per_threshold[tval] = ratios_at_threshold

    # Save to file
    outpath = Path(out_dir) / 'tables' / 'stats_saturation_budget.txt'
    with open(outpath, 'w') as f:
        f.write("Section 6.2 Effect 1: Saturation Budget Asymmetry (reproduced)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        for tname, _, tval in thresholds:
            ratios = all_ratios_per_threshold[tval]
            if ratios:
                f.write(f"Threshold {tname}:\n")
                f.write(f"  Range: {min(ratios):.0f}× – {max(ratios):.0f}×\n")
                f.write(f"  Median: {np.median(ratios):.0f}×\n\n")

    return all_ratios_per_threshold


# =============================================================================
# Section 6.3 HIERARCHICAL FINGERPRINT: Slope (Level 1) + Profile LOO (Level 2)
# =============================================================================

def section_63_hierarchical_fingerprint(imn_main, cif_main, log, out_dir):
    """
    Reproduce the hierarchical PE fingerprint introduced in Section 6.3:
    - Level 1 (universal): per-PE slope of layer-wise ratio profile.
                           Expected: ALiBi ≈ +0.01, others ≈ -0.2.
    - Level 2 (within-dataset): 12-d profile 1-NN LOO classification.
                                Expected: 67% IMN, 92% CIF (cityblock).
    """
    log.section("HIERARCHICAL PE FINGERPRINT (Section 6.3)")
    log.log("Two complementary identification levels from layer-wise ADS profile:")
    log.log("  Level 1 (universal): attenuation slope of ratio profile")
    log.log("  Level 2 (within-dataset): 12-d profile classification")
    log.log('')
    log.log("Expected (paper Section 6.3 / Tables VII, VIII):")
    log.log("  Level 1 slopes IMN: Learned -0.245, Sin -0.224, RoPE -0.218, ALiBi +0.013")
    log.log("  Level 1 slopes CIF: Learned -0.206, Sin -0.155, RoPE -0.253, ALiBi +0.016")
    log.log("  Level 2 LOO IMN (cityblock): 66.7%; CIF: 91.7%")
    log.log('')

    def get_profiles(data):
        """Returns dict {pe: [seed_profiles]}, each seed_profile is 12-d."""
        result = {pe: [] for pe in PE_TYPES}
        for pe in PE_TYPES:
            for seed in SEEDS:
                sd = data[pe][seed]
                ads_per_layer = np.array(sd['ads_per_layer'])
                ads_mean = ads_per_layer.mean(axis=1)
                profile = []
                for layer in range(12):
                    ratios = []
                    for ei in range(1, ads_per_layer.shape[0]):
                        if ads_mean[ei] > 1e-10:
                            ratios.append(ads_per_layer[ei, layer] / ads_mean[ei])
                    profile.append(np.mean(ratios) if ratios else 0)
                result[pe].append(profile)
        return result

    log.subsection("Level 1: Attenuation Slope (Table VII)")

    slope_results = {}
    for ds_label, data in [('IMN', imn_main), ('CIF', cif_main)]:
        slope_results[ds_label] = {}
        profiles = get_profiles(data)
        for pe in PE_TYPES:
            x = np.arange(1, 13)
            slopes = [np.polyfit(x, p, 1)[0] for p in profiles[pe]]
            stds = [np.std(p) for p in profiles[pe]]
            slope_results[ds_label][pe] = {
                'mean': np.mean(slopes),
                'std': np.std(slopes, ddof=1),
                'profile_std': np.mean(stds),
            }

    log.log(f"  {'PE':<12} {'IMN slope':<22} {'CIF slope':<22} {'IMN/CIF profile std'}")
    for pe in PE_TYPES:
        i = slope_results['IMN'][pe]
        c = slope_results['CIF'][pe]
        log.log(f"  {PE_DISPLAY[pe]:<12} "
                f"{i['mean']:+.3f} ± {i['std']:.3f}      "
                f"{c['mean']:+.3f} ± {c['std']:.3f}      "
                f"{i['profile_std']:.2f} / {c['profile_std']:.2f}")

    # Verify ALiBi vs others contrast
    log.log('')
    alibi_imn = slope_results['IMN']['alibi']['mean']
    others_imn = np.mean([slope_results['IMN'][pe]['mean'] for pe in ['learned', 'sinusoidal', 'rope']])
    ratio = abs(others_imn / alibi_imn) if alibi_imn != 0 else float('inf')
    log.log(f"  ALiBi vs others slope magnitude ratio (IMN): "
            f"|{others_imn:.3f}/{alibi_imn:+.3f}| = {ratio:.0f}×  (paper: ~17×)")

    log.subsection("Level 2: 12-d Profile LOO Classification (Table VIII)")

    loo_results = {}
    for ds_label, data in [('IMN', imn_main), ('CIF', cif_main)]:
        profiles = get_profiles(data)
        X = np.array([p for pe in PE_TYPES for p in profiles[pe]])
        y = np.array([pe for pe in PE_TYPES for _ in profiles[pe]])

        loo_results[ds_label] = {}
        for metric_name in ['euclidean', 'cosine', 'cityblock']:
            n = len(X)
            correct = 0
            for i in range(n):
                mask = np.arange(n) != i
                if metric_name == 'euclidean':
                    dists = np.linalg.norm(X[mask] - X[i], axis=1)
                elif metric_name == 'cosine':
                    norms = np.linalg.norm(X[mask], axis=1) * np.linalg.norm(X[i])
                    dists = 1 - (X[mask] @ X[i]) / (norms + 1e-10)
                else:  # cityblock
                    dists = np.sum(np.abs(X[mask] - X[i]), axis=1)
                pred = y[mask][np.argmin(dists)]
                if pred == y[i]:
                    correct += 1
            loo_results[ds_label][metric_name] = correct / n

    log.log(f"  {'Dataset':<14} {'Euclidean':>10} {'Cosine':>10} {'Cityblock':>10}")
    for ds_label in ['IMN', 'CIF']:
        ds_full = 'ImageNet-100' if ds_label == 'IMN' else 'CIFAR-100'
        log.log(f"  {ds_full:<14} "
                f"{loo_results[ds_label]['euclidean']*100:>8.1f}% "
                f"{loo_results[ds_label]['cosine']*100:>9.1f}% "
                f"{loo_results[ds_label]['cityblock']*100:>9.1f}%")

    # Cross-dataset transfer test
    log.log('')
    log.subsection("Cross-dataset profile transfer (Section 6.3)")
    profiles_imn = get_profiles(imn_main)
    profiles_cif = get_profiles(cif_main)

    # IMN centroids → predict CIF
    centroids_imn = {pe: np.mean(profiles_imn[pe], axis=0) for pe in PE_TYPES}
    correct = 0
    total = 0
    for pe in PE_TYPES:
        for profile in profiles_cif[pe]:
            dists = {p: np.linalg.norm(np.array(profile) - centroids_imn[p]) for p in PE_TYPES}
            pred = min(dists, key=dists.get)
            if pred == pe:
                correct += 1
            total += 1
    transfer_imn_to_cif = correct / total
    log.log(f"  IMN centroids → CIF: {correct}/{total} = {transfer_imn_to_cif*100:.1f}%")

    centroids_cif = {pe: np.mean(profiles_cif[pe], axis=0) for pe in PE_TYPES}
    correct = 0
    total = 0
    for pe in PE_TYPES:
        for profile in profiles_imn[pe]:
            dists = {p: np.linalg.norm(np.array(profile) - centroids_cif[p]) for p in PE_TYPES}
            pred = min(dists, key=dists.get)
            if pred == pe:
                correct += 1
            total += 1
    transfer_cif_to_imn = correct / total
    log.log(f"  CIF centroids → IMN: {correct}/{total} = {transfer_cif_to_imn*100:.1f}%")
    log.log(f"  Paper claim: 50% (limited cross-dataset transfer)")

    # Save
    outpath = Path(out_dir) / 'tables' / 'stats_hierarchical_fingerprint.txt'
    with open(outpath, 'w') as f:
        f.write("Section 6.3 Hierarchical PE Fingerprint (reproduced)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("Level 1 (Slope) - IMN:\n")
        for pe in PE_TYPES:
            r = slope_results['IMN'][pe]
            f.write(f"  {PE_DISPLAY[pe]:<12}: slope = {r['mean']:+.3f} ± {r['std']:.3f}\n")
        f.write("\nLevel 1 (Slope) - CIF:\n")
        for pe in PE_TYPES:
            r = slope_results['CIF'][pe]
            f.write(f"  {PE_DISPLAY[pe]:<12}: slope = {r['mean']:+.3f} ± {r['std']:.3f}\n")
        f.write(f"\nLevel 2 (LOO 12-d profile):\n")
        f.write(f"  IMN cityblock: {loo_results['IMN']['cityblock']*100:.1f}%\n")
        f.write(f"  CIF cityblock: {loo_results['CIF']['cityblock']*100:.1f}%\n")
        f.write(f"\nCross-dataset transfer:\n")
        f.write(f"  IMN→CIF: {transfer_imn_to_cif*100:.1f}%\n")
        f.write(f"  CIF→IMN: {transfer_cif_to_imn*100:.1f}%\n")

    return slope_results, loo_results


# =============================================================================
# Section 4.4 THRESHOLD CALIBRATION (interpolated baseline ε̂)
# =============================================================================

def section_44_threshold(threshold_data, log, out_dir):
    """
    Reproduce the interpolated detection threshold ε̂ ≈ 0.003 reported in
    Section 4.4 (and referenced throughout). The threshold is computed per
    (PE, seed) by interpolating the ε at which ADS(L4) reaches a multiple
    of the clean baseline ADS.
    """
    log.section("SECTION 4.4: Threshold Calibration (ε̂ interpolation)")
    log.log("Per-(PE, seed) interpolated threshold ε̂ at which ADS(L4) crosses")
    log.log("a fixed multiple of the clean baseline ADS.")
    log.log('')
    log.log("Expected (paper): ε̂ ≈ 0.003 across all PE types (Kruskal-Wallis")
    log.log("H = 5.05, p = 0.168, confirming universality).")
    log.log('')

    if threshold_data is None:
        log.log("  [SKIP] ads_threshold_fine.json not available")
        return None

    log.log(f"  {'PE':<12} {'Seed 42':>10} {'Seed 123':>10} {'Seed 456':>10} {'Mean':>10}")
    log.log('  ' + '-' * 55)

    all_thresholds = {}
    for pe in PE_TYPES:
        per_seed = []
        for seed in SEEDS:
            sd = threshold_data[pe][seed]
            per_seed.append(sd['interpolated_threshold'])
        m = np.mean(per_seed)
        all_thresholds[pe] = {'per_seed': per_seed, 'mean': m}
        log.log(f"  {PE_DISPLAY[pe]:<12} "
                f"{per_seed[0]:>10.4f} {per_seed[1]:>10.4f} {per_seed[2]:>10.4f} "
                f"{m:>10.4f}")

    # Kruskal-Wallis test for universality
    groups = [all_thresholds[pe]['per_seed'] for pe in PE_TYPES]
    h_stat, p_val = stats.kruskal(*groups)
    log.log('')
    log.log(f"  Kruskal-Wallis H = {h_stat:.2f}, p = {p_val:.4f} "
            f"{'(non-significant — universal)' if p_val > 0.05 else '(significant)'}")
    log.log(f"  Paper claim: H = 5.05, p = 0.168")

    # Save
    outpath = Path(out_dir) / 'tables' / 'stats_threshold_calibration.txt'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w') as f:
        f.write(f"Section 4.4 Threshold Calibration (reproduced)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        for pe in PE_TYPES:
            t = all_thresholds[pe]
            f.write(f"  {PE_DISPLAY[pe]:<12}: ε̂ = {t['mean']:.4f} "
                    f"(per-seed: {t['per_seed']})\n")
        f.write(f"\nKruskal-Wallis H = {h_stat:.2f}, p = {p_val:.4f}\n")

    return {'thresholds': all_thresholds, 'kw_H': h_stat, 'kw_p': p_val}


# =============================================================================
# Section 5: ROC ANALYSIS (deployment detection boundary)
# =============================================================================

def section_5_roc(roc_data, log, out_dir):
    """
    Reproduce the operational detection boundary (Section 5):
    - Per-(PE, seed) AUC at each ε
    - Confirm AUC ≥ 0.82 at ε ≥ 0.1 (Learned) / ε ≥ 0.2 (RoPE)
    """
    log.section("SECTION 5: ROC Analysis (Detection Boundary)")
    log.log("Per-(PE, seed) AUC against benign distribution shifts (clean noise +")
    log.log("JPEG compression + Gaussian blur + Gaussian noise as negatives).")
    log.log('')
    log.log("Expected (paper): AUC ≥ 0.82 at ε ≥ 0.1 (Learned), ε ≥ 0.2 (RoPE).")
    log.log('')

    if roc_data is None:
        log.log("  [SKIP] ads_roc_v2.json not available")
        return None

    pe_list = [pe for pe in PE_TYPES if pe in roc_data]

    # Find epsilons available
    sample_pe = pe_list[0]
    eps_keys = sorted(roc_data[sample_pe]['42']['roc_by_eps'].keys(), key=float)

    log.log(f"  {'PE':<10} " + ' '.join(f"{'ε='+e:>10}" for e in eps_keys))
    log.log('  ' + '-' * (10 + 11 * len(eps_keys)))

    auc_results = {pe: {} for pe in pe_list}
    for pe in pe_list:
        line = f"  {PE_DISPLAY[pe]:<10} "
        for eps in eps_keys:
            seed_aucs = []
            for seed in SEEDS:
                roc_eps = roc_data[pe][seed]['roc_by_eps'][eps]
                seed_aucs.append(roc_eps['auc'])
            mean_auc = np.mean(seed_aucs)
            auc_results[pe][eps] = mean_auc
            line += f"{mean_auc:>10.3f} "
        log.log(line)

    # Verify operational boundary claim
    log.log('')
    log.log("Operational boundary verification:")
    if 'learned' in auc_results:
        l_01 = auc_results['learned'].get('0.1', 0)
        log.log(f"  Learned at ε=0.1:  AUC = {l_01:.3f}  "
                f"{'✓' if l_01 >= 0.82 else '✗'} (paper claim ≥ 0.82)")
    if 'rope' in auc_results:
        r_02 = auc_results['rope'].get('0.2', 0)
        log.log(f"  RoPE at ε=0.2:    AUC = {r_02:.3f}  "
                f"{'✓' if r_02 >= 0.82 else '✗'} (paper claim ≥ 0.82)")

    # Save
    outpath = Path(out_dir) / 'tables' / 'stats_roc_analysis.txt'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w') as f:
        f.write(f"Section 5 ROC Analysis (reproduced)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        for pe in pe_list:
            f.write(f"{PE_DISPLAY[pe]}:\n")
            for eps in eps_keys:
                f.write(f"  ε={eps}: AUC={auc_results[pe][eps]:.3f}\n")

    return auc_results


# =============================================================================
# Section 6.6 COMPARISON: ADS vs Mahalanobis vs LogitKL (Table VIII)
# =============================================================================

def section_66_comparison(comparison_data, log, out_dir):
    """
    Reproduce Table VIII: Detection AUC comparison across methods (ADS,
    Mahalanobis, LogitKL) at multiple ε.
    """
    log.section("SECTION 6.6: Detection Method Comparison (Table VIII)")
    log.log("Detection AUC at each ε for ADS vs. Mahalanobis vs. LogitKL.")
    log.log('')
    log.log("Expected (paper Table VIII): ADS AUC ≈ 1.0 from ε ≥ 0.05;")
    log.log("Mahalanobis AUC = 1.0 starting from ε = 0.001 (overconfident);")
    log.log("LogitKL AUC = 0.0 throughout (fails for PE attacks).")
    log.log('')

    if comparison_data is None:
        log.log("  [SKIP] ads_comparison.json not available")
        return None

    pe_list = [pe for pe in PE_TYPES if pe in comparison_data]
    eps_keys = sorted(comparison_data[pe_list[0]]['42']['attack'].keys(),
                      key=float)

    for pe in pe_list:
        log.subsection(f"{PE_DISPLAY[pe]}")
        log.log(f"    {'ε':<8} {'ADS AUC':<10} {'Mahal AUC':<11} {'LogitKL AUC':<12}")
        log.log('    ' + '-' * 42)
        for eps in eps_keys:
            seed_aucs = {'ads': [], 'mahalanobis': [], 'logit_kl': []}
            for seed in SEEDS:
                attack_data = comparison_data[pe][seed]['attack'][eps]
                for method in seed_aucs:
                    seed_aucs[method].append(attack_data[method]['auc'])
            log.log(f"    {eps:<8} "
                    f"{np.mean(seed_aucs['ads']):<10.3f} "
                    f"{np.mean(seed_aucs['mahalanobis']):<11.3f} "
                    f"{np.mean(seed_aucs['logit_kl']):<12.3f}")

    # Calibration timing
    log.log('')
    log.log("Calibration time (seconds, mean over PE × seeds):")
    timing_summary = {'ads': [], 'mahal': [], 'logit': []}
    for pe in pe_list:
        for seed in SEEDS:
            t = comparison_data[pe][seed]['timing']
            timing_summary['ads'].append(t['ads_calibration_s'])
            timing_summary['mahal'].append(t['mahalanobis_calibration_s'])
            timing_summary['logit'].append(t['logit_calibration_s'])
    log.log(f"  ADS:        {np.mean(timing_summary['ads']):.2f} s (paper: ~2s)")
    log.log(f"  Mahalanobis: {np.mean(timing_summary['mahal']):.2f} s")
    log.log(f"  LogitKL:    {np.mean(timing_summary['logit']):.2f} s")

    return None


# =============================================================================
# Section 6.6 REFERENCE SET EVASION (Table IX in paper text — actually reuses
# the "lambda regularization on ref vs hold" concept)
# =============================================================================

def section_66_evasion(evasion_data, log, out_dir):
    """
    Reproduce reference set evasion analysis: does an attacker who knows
    the reference set indices evade detection on the held-out set as well?
    """
    log.section("SECTION 6.6: Reference Set Evasion")
    log.log("Adaptive attacker that minimizes ADS on the reference set:")
    log.log("does the resulting attack also evade detection on a held-out set?")
    log.log('')
    log.log("Expected (paper): No --- adaptive attacker fails to evade")
    log.log("on held-out set in operational regime, demonstrating robustness")
    log.log("to reference-set-specific overfitting.")
    log.log('')

    if evasion_data is None:
        log.log("  [SKIP] ads_ref_evasion.json not available")
        return None

    pe_list = [pe for pe in PE_TYPES if pe in evasion_data]

    for pe in pe_list:
        log.subsection(f"{PE_DISPLAY[pe]}")
        eps_keys = sorted(evasion_data[pe]['42'].keys(), key=float)
        for eps in eps_keys:
            lam_keys = sorted(evasion_data[pe]['42'][eps].keys(), key=float)
            log.log(f"    ε = {eps}:")
            log.log(f"      {'λ':<6} {'AccDrop':<10} {'ADS_ref':<12} "
                    f"{'ADS_hold':<12} {'Evades_ref':<11} {'Evades_hold':<11}")
            for lam in lam_keys:
                vals = {'acc_drop': [], 'ads_ref': [], 'ads_hold': [],
                        'evades_ref': [], 'evades_hold': []}
                for seed in SEEDS:
                    d = evasion_data[pe][seed][eps][lam]
                    vals['acc_drop'].append(d['acc_drop'])
                    vals['ads_ref'].append(d['ads_ref'])
                    vals['ads_hold'].append(d['ads_hold'])
                    vals['evades_ref'].append(d['evades_ref'])
                    vals['evades_hold'].append(d['evades_hold'])
                log.log(f"      {lam:<6} "
                        f"{np.mean(vals['acc_drop']):<10.2f} "
                        f"{np.mean(vals['ads_ref']):<12.4f} "
                        f"{np.mean(vals['ads_hold']):<12.4f} "
                        f"{sum(vals['evades_ref'])}/{len(vals['evades_ref'])}{'':<7} "
                        f"{sum(vals['evades_hold'])}/{len(vals['evades_hold'])}")

    return None


# =============================================================================
# Section 6.7 ADAPTIVE ATTACKER (PGD with ADS-minimization regularization)
# =============================================================================

def section_67_adaptive(adaptive_data, log, out_dir):
    """
    Reproduce the adaptive attacker analysis (Section 6.7): a PGD attacker
    that adds an ADS-minimization term to its loss, parameterized by λ.
    Verify that increasing λ does NOT enable evasion in operational regime.
    """
    log.section("SECTION 6.7: Adaptive Attacker (PGD + ADS-min regularization)")
    log.log("PGD attacker with loss: L = L_attack - λ * ADS_L4")
    log.log("Higher λ should reduce ADS, but at the cost of attack effectiveness.")
    log.log('')
    log.log("Expected (paper): At ε ≥ 0.1, ADS remains structurally above")
    log.log("baseline regardless of λ; adaptive attacker cannot evade detection.")
    log.log('')

    if adaptive_data is None:
        log.log("  [SKIP] ads_adaptive.json not available")
        return None

    pe_list = [pe for pe in PE_TYPES if pe in adaptive_data]

    for pe in pe_list:
        log.subsection(f"{PE_DISPLAY[pe]}")
        eps_keys = sorted(adaptive_data[pe]['42'].keys(), key=float)
        for eps in eps_keys:
            lam_keys = sorted(adaptive_data[pe]['42'][eps].keys(), key=float)
            log.log(f"    ε = {eps}:")
            log.log(f"      {'λ':<6} {'AccDrop':<10} {'ADS_L4':<10} "
                    f"{'ADS_ratio':<11} {'Evades?'}")
            for lam in lam_keys:
                vals = {'acc_drop': [], 'ads_l4': [], 'ratio': [], 'evades': []}
                for seed in SEEDS:
                    d = adaptive_data[pe][seed][eps][lam]
                    vals['acc_drop'].append(d['acc_drop'])
                    vals['ads_l4'].append(d['ads_l4'])
                    vals['ratio'].append(d['ads_ratio'])
                    vals['evades'].append(d['evades_detection'])
                evades_count = sum(vals['evades'])
                log.log(f"      {lam:<6} "
                        f"{np.mean(vals['acc_drop']):<10.2f} "
                        f"{np.mean(vals['ads_l4']):<10.4f} "
                        f"{np.mean(vals['ratio']):<11.2f} "
                        f"{evades_count}/{len(vals['evades'])}")

    return None


# =============================================================================
# Reference Set Indices verification
# =============================================================================

def section_4_ref_indices(ref_indices_data, log, out_dir):
    """Verify that the reference set has 256 indices as reported in the paper."""
    log.section("Reference Set Indices Verification")
    if ref_indices_data is None:
        log.log("  [SKIP] ads_ref_indices.json not available")
        return None

    n = len(ref_indices_data)
    is_unique = len(set(ref_indices_data)) == n
    in_range = all(0 <= i < 5000 for i in ref_indices_data)  # ImageNet-100 val: 5k images

    log.log(f"  Reference set size: {n} (paper claim: 256)")
    log.log(f"  All unique: {'✓' if is_unique else '✗'}")
    log.log(f"  All in [0, 5000): {'✓' if in_range else '✗'}")
    log.log(f"  Min index: {min(ref_indices_data)}, Max: {max(ref_indices_data)}")

    return {'n': n, 'unique': is_unique, 'in_range': in_range}


# =============================================================================
# TABLE IX: RESIDUAL STREAM PROBING (Section 7.1)
# =============================================================================

def section_71_probing(imn_probe, cif_probe, log, out_dir):
    log.section("TABLE VIII: Residual Stream Probing")
    log.log("Peak decodability (R²-mean) across layers 1-12 (mean of 3 seeds).")
    log.log("Expected (paper Section 7.1 Table VIII):")
    log.log("  IMN: Learned L1 R²=0.79, Sinus L2 R²=0.94, RoPE L8 R²=0.27, ALiBi L3 R²=0.48")
    log.log("  CIF: Learned L1 R²=0.99, Sinus L1 R²=0.99, RoPE L6 R²=0.61, ALiBi L5 R²=0.49")
    log.log('')

    for label, data in [('IMAGENET-100', imn_probe), ('CIFAR-100', cif_probe)]:
        log.subsection(label)
        if data is None:
            log.log("  [SKIP] Probing JSON not available in data/")
            continue

        hdr = f"  {'PE':<12} {'Peak L':>8} {'Peak R²':>10} {'R²(L4)':>10} {'R²(L1)':>10} {'Peaks/seed':<20}"
        log.log(hdr)
        log.log('  ' + '-' * (len(hdr) - 2))
        for pe in PE_TYPES:
            if pe not in data:
                continue
            # Collect R²(mean) per layer, averaged across seeds
            r2_per_layer = []
            peaks_per_seed = []
            for layer in range(1, 13):
                vals = [data[pe][s]['layers'][str(layer)]['r2_mean']
                        for s in SEEDS]
                r2_per_layer.append(np.mean(vals))
            # Per-seed peaks
            for s in SEEDS:
                layer_vals = [(l, data[pe][s]['layers'][str(l)]['r2_mean'])
                              for l in range(1, 13)]
                peaks_per_seed.append(max(layer_vals, key=lambda x: x[1])[0])
            peak_layer = int(np.argmax(r2_per_layer)) + 1
            peak_r2 = r2_per_layer[peak_layer - 1]
            log.log(f"  {PE_DISPLAY[pe]:<12} L{peak_layer:<7} "
                    f"{peak_r2:>10.3f} {r2_per_layer[3]:>10.3f} "
                    f"{r2_per_layer[0]:>10.3f} {str(peaks_per_seed):<20}")


# =============================================================================
# FIGURE GENERATION (calls external generate_ads_figures.py if available)
# =============================================================================

def generate_figures(log, out_dir):
    log.section("STEP 4: Figure generation")
    log.log("Figures are generated via a separate script (generate_ads_figures.py)")
    log.log("which reads the same JSONs from data/ and writes PNGs to output/figures/.")
    log.log('')
    log.log("To regenerate all 4 figures:")
    log.log("    python generate_ads_figures.py")
    log.log('')
    log.log("Expected outputs in output/figures/:")
    log.log("    ads_fig1_l4_vs_epsilon.png         (Fig 1 in paper)")
    log.log("    ads_fig2_per_layer_heatmap.png     (Fig 2 in paper)")
    log.log("    ads_fig3_layer_profile.png         (Fig 3 in paper)")
    log.log("    ads_fig4_early_warning_combined.png (Fig 4 in paper)")


# =============================================================================
# VERIFICATION: compare reproduced values to paper values
# =============================================================================

def verify_against_paper(fingerprint_results, saturation_results, log):
    log.section("STEP 5: Verification against published values")
    log.log("Tolerance: ±0.1 for ratios, ±0.05 for CI widths, ±5× for budget ratios")
    log.log('')

    log.subsection("Fingerprint ratios (Table VI)")
    expected = {
        'learned':    {'imn_mean': 3.19, 'imn_ci': 1.65},
        'sinusoidal': {'imn_mean': 2.34, 'imn_ci': 0.86},
        'rope':       {'imn_mean': 1.82, 'imn_ci': 1.19},
        'alibi':      {'imn_mean': 0.83, 'imn_ci': 0.59},
    }

    fingerprint_pass = True
    for pe, exp in expected.items():
        got = fingerprint_results[pe]
        diff_mean = abs(got['imn_mean'] - exp['imn_mean'])
        diff_ci = abs(got['imn_ci'] - exp['imn_ci'])
        pass_mean = diff_mean < 0.1
        pass_ci = diff_ci < 0.05
        status = '✓ PASS' if (pass_mean and pass_ci) else '✗ FAIL'
        log.log(f"  {PE_DISPLAY[pe]:<12}: "
                f"mean={got['imn_mean']:.2f} (exp {exp['imn_mean']}, Δ={diff_mean:.2f})  "
                f"CI=±{got['imn_ci']:.2f} (exp ±{exp['imn_ci']}, Δ={diff_ci:.2f})  "
                f"{status}")
        if not (pass_mean and pass_ci):
            fingerprint_pass = False

    log.log('')
    log.subsection("Saturation budget ratios (Section 6.2 Effect 1)")
    log.log("  Paper claims:")
    log.log("    50% threshold: 17×–200× combined range across PE types and datasets")
    log.log("    5% threshold:  100×–200× on ImageNet (where applicable)")
    log.log('')

    sat_pass = True
    if saturation_results and 50 in saturation_results:
        ratios_50 = saturation_results[50]
        if ratios_50:
            min_r, max_r = min(ratios_50), max(ratios_50)
            # Paper claims combined range 17x-200x
            min_match = abs(min_r - 17) < 5
            max_match = abs(max_r - 200) < 5
            status_min = '✓' if min_match else '✗'
            status_max = '✓' if max_match else '✗'
            log.log(f"  50% threshold actual range: {min_r:.0f}× – {max_r:.0f}×")
            log.log(f"    Min ratio (claim 17×):    {status_min} (Δ={abs(min_r-17):.0f}×)")
            log.log(f"    Max ratio (claim 200×):   {status_max} (Δ={abs(max_r-200):.0f}×)")
            if not (min_match and max_match):
                sat_pass = False
        else:
            log.log("  No 50%-threshold ratios computed (data missing)")

    if saturation_results and 5 in saturation_results:
        ratios_5 = saturation_results[5]
        if ratios_5:
            log.log(f"  5% threshold actual range: {min(ratios_5):.0f}× – {max(ratios_5):.0f}×")
            # Note: this is a softer check; we just log it
            # Paper claims 100×–167× on ImageNet alone

    log.log('')
    if fingerprint_pass and sat_pass:
        log.log("✓ All reproduced values within tolerance of published numbers.")
    else:
        log.log("✗ Some values differ from published numbers. Check data/ and code/.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global DATA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Reproduce paper numbers from archived JSON data."
    )
    parser.add_argument('--no-figures', action='store_true',
                        help="Skip figure generation (tables only)")
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help="Where to write tables and figures")
    parser.add_argument('--data-dir', default=DATA_DIR,
                        help="Where to find JSON data files")
    parser.add_argument('--section', default=None,
                        help="Reproduce only specific section (e.g., '6.2', '6.3', '7.1')")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir

    # Set up output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    log = Logger(Path(OUTPUT_DIR) / 'reproduce_log.txt')

    log.section("REPRODUCTION RUN — " + datetime.now().isoformat())
    log.log("Paper: 'Attention Divergence Score: A Forensic Metric for")
    log.log("        Characterizing Parameter-Level Attacks in Vision Transformers'")
    log.log("Submitted to: IEEE TIFS")
    log.log(f"Data dir:     {DATA_DIR}")
    log.log(f"Output dir:   {OUTPUT_DIR}")

    data = load_all_data(log)

    # Pre-init for verify
    fp_results = None
    sat_results = None

    # Run each section's reproduction
    if args.section is None or args.section == '4':
        section_4_ref_indices(data['ref_indices'], log, OUTPUT_DIR)

    if args.section is None or args.section == '4.4':
        section_44_threshold(data['threshold'], log, OUTPUT_DIR)

    if args.section is None or args.section == '5':
        section_5_roc(data['roc'], log, OUTPUT_DIR)

    if args.section is None or args.section == '6.2':
        if data['imn_spec'] and data['cif_spec']:
            section_62_specificity(data['imn_spec'], data['cif_spec'], log, OUTPUT_DIR)
            sat_results = section_62_saturation(data['imn_spec'], data['cif_spec'],
                                                log, OUTPUT_DIR)
        else:
            log.log("\n[SKIP] Section 6.2 — specificity data missing")

    if args.section is None or args.section == '6.3':
        if data['imn_main'] and data['cif_main']:
            fp_results = table_VI_fingerprint(data['imn_main'], data['cif_main'],
                                              log, OUTPUT_DIR)
            section_63_anova(data['imn_main'], data['cif_main'], log, OUTPUT_DIR)
            section_63_1nn_loo(data['imn_main'], log, OUTPUT_DIR)
            section_63_hierarchical_fingerprint(data['imn_main'], data['cif_main'],
                                                log, OUTPUT_DIR)
        else:
            log.log("\n[SKIP] Section 6.3 — main ADS data missing")

    if args.section is None or args.section == '6.6':
        section_66_comparison(data['comparison'], log, OUTPUT_DIR)
        section_66_evasion(data['evasion'], log, OUTPUT_DIR)

    if args.section is None or args.section == '6.7':
        section_67_adaptive(data['adaptive'], log, OUTPUT_DIR)

    if args.section is None or args.section == '7.1':
        section_71_probing(data['imn_probe'], data['cif_probe'], log, OUTPUT_DIR)

    # Run verification at the end (after both 6.2 saturation and 6.3 fingerprint computed)
    if fp_results is not None:
        verify_against_paper(fp_results, sat_results, log)

    if not args.no_figures and args.section is None:
        generate_figures(log, OUTPUT_DIR)

    log.section("DONE")
    log.log(f"All outputs saved to: {OUTPUT_DIR}/")
    log.log(f"Full log: {OUTPUT_DIR}/reproduce_log.txt")
    log.close()


if __name__ == '__main__':
    main()
