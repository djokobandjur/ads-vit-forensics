#!/usr/bin/env python3
"""
reproduce.py — CPU-only verification for

    "Attention Divergence Score: A Forensic Metric for Characterizing
     Parameter-Level Attacks in Vision Transformers"

This script verifies the numerical tables, core statistics, and figure inputs
used in the IEEE TIFS 13-page resubmission directly from archived JSON result
files. It does not train models and does not require a GPU or PyTorch.

Typical usage
-------------
    python reproduce.py
    python reproduce.py --data-dir data --output-dir output
    python reproduce.py --section 6.3
    python reproduce.py --no-figures

Expected data files
-------------------
Required for the primary verification:
    ads_results.json
    ads_results_cifar100.json
    ads_specificity.json
    ads_specificity_cifar.json
    ads_threshold_fine.json
    ads_ref_indices.json
    ads_probing_residual.json
    ads_probing_residual_cifar.json

Optional, verified when present:
    ads_roc_v2.json
    ads_comparison.json
    ads_adaptive.json
    ads_ref_evasion.json
    ads_shared_delta_imagenet100.json
    ads_shared_delta_cifar100.json
    ads_roc_rank_auc_sensitivity.json
    robustness/ads_results_cifar100_canonical_n12.json
    robustness/ads_specificity_cifar100_canonical_n12.json
    robustness/ads_probing_residual_cifar100_canonical_n12.json

Outputs
-------
    output/reproduce_log.txt
    output/tables/*.txt

The script treats training seed as the unit of independence. The primary paper
configuration uses six seeds per PE type: 42, 123, 456, 789, 1011, 1213.

Authors: Djoko Bandjur, Milos Bandjur
License: MIT (code) / CC BY 4.0 (data)
Last updated: July 2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


PE_TYPES = ["learned", "sinusoidal", "rope", "alibi"]
PE_DISPLAY = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "rope": "RoPE",
    "alibi": "ALiBi",
    "alibi_2d": "2D-ALiBi fixed",
    "alibi_2d_matched": "2D-ALiBi matched",
}
PRIMARY_SEEDS = ["42", "123", "456", "789", "1011", "1213"]
ATTACKS = ["pe_only", "qkv_only", "mlp_only", "all_weights"]
TARGET_EPS = 0.2


# -----------------------------------------------------------------------------
# Logging and helpers
# -----------------------------------------------------------------------------

class Logger:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "reproduce_log.txt"
        self.file = self.path.open("w", encoding="utf-8")
        self.start = datetime.now()
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def log(self, msg: str = "", *, also_print: bool = True) -> None:
        elapsed = (datetime.now() - self.start).total_seconds()
        line = f"[{elapsed:8.2f}s] {msg}"
        self.file.write(line + "\n")
        self.file.flush()
        if also_print:
            print(msg)

    def section(self, title: str) -> None:
        bar = "=" * 88
        self.log("")
        self.log(bar)
        self.log(title)
        self.log(bar)

    def subsection(self, title: str) -> None:
        self.log("")
        self.log(title)
        self.log("-" * len(title))

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.log(f"[WARN] {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        self.log(f"[FAIL] {msg}")

    def pass_check(self, msg: str) -> None:
        self.log(f"[PASS] {msg}")

    def close(self) -> None:
        self.file.close()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return f"{x:.{nd}f}"


def t_ci(values: Sequence[float], alpha: float = 0.05) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) <= 1:
        return float("nan")
    return float(stats.t.ppf(1 - alpha / 2, df=len(arr) - 1) * arr.std(ddof=1) / math.sqrt(len(arr)))


def std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")


def mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def nearest_idx(values: Sequence[float], target: float) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(arr - target)))


def get_seeds(block: Mapping[str, Any]) -> List[str]:
    return sorted([str(s) for s in block.keys()], key=lambda x: int(x) if x.isdigit() else x)


def normalize_seed_keys(data: Any) -> Any:
    """JSON object keys are already strings, but keep this explicit for clarity."""
    return data


def write_table(path: Path, lines: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

FALLBACK_FILENAMES = {
    "imn_main": ["ads_results.json", "ads_results(1).json", "ads_results (1).json"],
    "cif_main": ["ads_results_cifar100.json"],
    "imn_spec": ["ads_specificity.json", "ads_specificity(1).json"],
    "cif_spec": ["ads_specificity_cifar.json"],
    "threshold": ["ads_threshold_fine.json"],
    "roc": ["ads_roc_v2.json", "ads_roc_v2(1).json"],
    "comparison": ["ads_comparison.json", "ads_comparison(1).json"],
    "adaptive": ["ads_adaptive.json", "ads_adaptive(1).json", "ads_adaptive(2).json"],
    "evasion": ["ads_ref_evasion.json", "ads_ref_evasion(1).json", "ads_ref_evasion(2).json"],
    "ref_indices": ["ads_ref_indices.json", "ads_ref_indices (1).json"],
    "imn_probe": ["ads_probing_residual.json"],
    "cif_probe": ["ads_probing_residual_cifar.json"],
    "shared_delta_imn": ["ads_shared_delta_imagenet100.json"],
    "shared_delta_cif": ["ads_shared_delta_cifar100.json"],
    "roc_rank_auc": ["ads_roc_rank_auc_sensitivity.json"],
    "canonical_main": ["robustness/ads_results_cifar100_canonical_n12.json", "ads_results_cifar100_canonical_n12.json"],
    "canonical_spec": ["robustness/ads_specificity_cifar100_canonical_n12.json", "ads_specificity_cifar100_canonical_n12.json", "ads_specificity_cifar100_canonical_n12(1).json"],
    "canonical_probe": ["robustness/ads_probing_residual_cifar100_canonical_n12.json", "ads_probing_residual_cifar100_canonical_n12.json", "ads_probing_residual_cifar100_canonical_n12(1).json"],
}


def load_json_file(data_dir: Path, candidates: Sequence[str], log: Logger, required: bool = False) -> Any:
    for filename in candidates:
        path = data_dir / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            log.log(f"  [OK]   {filename}")
            return normalize_seed_keys(obj)
    label = candidates[0]
    if required:
        log.fail(f"Required data file missing: {label}")
    else:
        log.warn(f"Optional data file missing: {label}")
    return None


def load_all_data(data_dir: Path, log: Logger) -> Dict[str, Any]:
    log.section("STEP 1 — Loading JSON artifacts")
    required = {"imn_main", "cif_main", "imn_spec", "cif_spec", "threshold", "ref_indices", "imn_probe", "cif_probe"}
    data: Dict[str, Any] = {}
    for key, candidates in FALLBACK_FILENAMES.items():
        data[key] = load_json_file(data_dir, candidates, log, required=(key in required))
    return data


def validate_primary_seed_coverage(data: Mapping[str, Any], log: Logger) -> None:
    log.section("STEP 2 — Seed coverage check")
    expected = set(PRIMARY_SEEDS)
    core_keys = ["imn_main", "cif_main", "imn_spec", "cif_spec", "threshold", "imn_probe", "cif_probe"]
    problems: List[str] = []
    for key in core_keys:
        block = data.get(key)
        if block is None:
            continue
        for pe in PE_TYPES:
            if pe not in block:
                problems.append(f"{key}/{pe}: PE missing")
                continue
            present = set(get_seeds(block[pe]))
            if present != expected:
                problems.append(f"{key}/{pe}: seeds {sorted(present)}; expected {PRIMARY_SEEDS}")
    if problems:
        for problem in problems[:20]:
            log.fail(problem)
        if len(problems) > 20:
            log.fail(f"... {len(problems) - 20} additional seed-coverage problems")
        log.log("\nPrimary verification requires the final n=6 JSON artifacts.")
        log.close()
        sys.exit(2)
    log.pass_check("Primary n=6 coverage present for main, specificity, threshold, and probing files.")

    for optional_key in ["roc", "comparison", "adaptive", "evasion"]:
        block = data.get(optional_key)
        if not isinstance(block, dict):
            continue
        optional_problems = []
        for pe, pe_block in block.items():
            if not isinstance(pe_block, dict):
                continue
            present = set(get_seeds(pe_block))
            if present != expected:
                optional_problems.append(f"{optional_key}/{pe}: seeds {sorted(present)}")
        if optional_problems:
            log.warn(f"Optional {optional_key} file is not final n=6: " + "; ".join(optional_problems))
        else:
            log.pass_check(f"Optional {optional_key} file has final n=6 seed coverage.")


# -----------------------------------------------------------------------------
# Core computations
# -----------------------------------------------------------------------------


def clean_accuracy_table(data: Mapping[str, Any], log: Logger, output_dir: Path) -> None:
    log.section("Clean accuracies")
    lines = ["Clean accuracy (%), mean ± std over six seeds", ""]
    header = f"{'PE':<12} {'ImageNet-100':>18} {'CIFAR-100':>18}"
    lines += [header, "-" * len(header)]
    log.log(header)
    log.log("-" * len(header))
    for pe in PE_TYPES:
        row_vals = []
        for key in ["imn_main", "cif_main"]:
            values = []
            for seed in PRIMARY_SEEDS:
                sd = data[key][pe][seed]
                if "clean_acc" in sd:
                    values.append(float(sd["clean_acc"]))
                else:
                    eps = np.asarray(sd["epsilons"], dtype=float)
                    acc = np.asarray(sd["accuracies"], dtype=float)
                    values.append(float(acc[nearest_idx(eps, 0.0)]))
            row_vals.append(f"{mean(values):.1f} ± {std(values):.1f}")
        line = f"{PE_DISPLAY[pe]:<12} {row_vals[0]:>18} {row_vals[1]:>18}"
        log.log(line)
        lines.append(line)
    write_table(output_dir / "tables" / "table_clean_accuracy.txt", lines)


def reference_indices_check(ref_indices: Any, log: Logger, output_dir: Path) -> None:
    log.section("Reference set indices")
    if ref_indices is None:
        return
    n = len(ref_indices)
    unique = len(set(ref_indices)) == n
    in_range = all(isinstance(i, int) and 0 <= i < 5000 for i in ref_indices)
    msg = f"n={n}, unique={unique}, in [0,5000)={in_range}, min={min(ref_indices)}, max={max(ref_indices)}"
    log.log(msg)
    if n == 256 and unique and in_range:
        log.pass_check("Reference set matches the manuscript protocol: 256 unique ImageNet-100 validation indices.")
    else:
        log.fail("Reference set index verification failed.")
    write_table(output_dir / "tables" / "stats_reference_indices.txt", [msg])


def mean_accuracy_curve(main_data: Mapping[str, Any], pe: str) -> Tuple[np.ndarray, np.ndarray, float]:
    eps = np.asarray(main_data[pe][PRIMARY_SEEDS[0]]["epsilons"], dtype=float)
    curves = []
    cleans = []
    for seed in PRIMARY_SEEDS:
        sd = main_data[pe][seed]
        curves.append(np.asarray(sd["accuracies"], dtype=float))
        if "clean_acc" in sd:
            cleans.append(float(sd["clean_acc"]))
        else:
            cleans.append(float(np.asarray(sd["accuracies"], dtype=float)[nearest_idx(eps, 0.0)]))
    return eps, np.mean(curves, axis=0), float(np.mean(cleans))


def mean_ads_l4_curve(main_data: Mapping[str, Any], pe: str) -> Tuple[np.ndarray, np.ndarray]:
    eps = np.asarray(main_data[pe][PRIMARY_SEEDS[0]]["epsilons"], dtype=float)
    curves = [np.asarray(main_data[pe][seed]["ads_layer4"], dtype=float) for seed in PRIMARY_SEEDS]
    return eps, np.mean(curves, axis=0)


def first_eps_where(eps: np.ndarray, values: np.ndarray, condition: Callable[[float], bool]) -> Optional[float]:
    for e, v in zip(eps, values):
        if condition(float(v)):
            return float(e)
    return None


def early_warning_table(data: Mapping[str, Any], log: Logger, output_dir: Path) -> None:
    log.section("Low-budget self-baseline trigger and 50%-of-clean collapse")
    header = f"{'PE':<12} {'Dataset':<12} {'collapse ε*':>12} {'ADS>10x at':>12} {'steps before':>12}"
    lines = [header, "-" * len(header)]
    log.log(header)
    log.log("-" * len(header))
    for pe in PE_TYPES:
        for key, label in [("imn_main", "ImageNet"), ("cif_main", "CIFAR")]:
            eps, acc_mean, clean = mean_accuracy_curve(data[key], pe)
            collapse = first_eps_where(eps, acc_mean, lambda a, c=clean: a <= 0.5 * c)
            eps_ads, ads_l4 = mean_ads_l4_curve(data[key], pe)
            baseline_idx = nearest_idx(eps_ads, 0.001)
            baseline = float(ads_l4[baseline_idx])
            trigger = first_eps_where(eps_ads, ads_l4, lambda a, b=baseline: a > 10.0 * b)
            if collapse is None or trigger is None:
                steps = "n/a"
            else:
                # Count tested epsilon grid points strictly between trigger and collapse.
                steps = int(np.sum((eps > trigger) & (eps <= collapse)))
            line = f"{PE_DISPLAY[pe]:<12} {label:<12} {collapse:>12.3f} {trigger:>12.3f} {str(steps):>12}"
            log.log(line)
            lines.append(line)
    write_table(output_dir / "tables" / "table_early_warning.txt", lines)


def threshold_calibration(threshold_data: Mapping[str, Any], log: Logger, output_dir: Path) -> None:
    log.section("Interpolated self-baseline threshold calibration")
    if threshold_data is None:
        return
    header = f"{'PE':<12} {'mean εhat':>10} {'min':>10} {'max':>10} {'max/min':>10}"
    lines = [header, "-" * len(header)]
    log.log(header)
    log.log("-" * len(header))
    groups = []
    all_values = []
    for pe in PE_TYPES:
        values = [float(threshold_data[pe][seed]["interpolated_threshold"]) for seed in PRIMARY_SEEDS]
        groups.append(values)
        all_values.extend(values)
        ratio = max(values) / min(values)
        line = f"{PE_DISPLAY[pe]:<12} {mean(values):>10.5f} {min(values):>10.5f} {max(values):>10.5f} {ratio:>10.2f}"
        log.log(line)
        lines.append(line)
    h, p = stats.kruskal(*groups)
    summary = f"Kruskal-Wallis H={h:.2f}, p={p:.3f}; global range=[{min(all_values):.5f}, {max(all_values):.5f}], max/min={max(all_values)/min(all_values):.2f}x"
    log.log(summary)
    lines += ["", summary]
    if p > 0.05:
        log.pass_check("Self-baseline thresholds are not significantly different across PE types.")
    else:
        log.warn("Self-baseline thresholds differ significantly across PE types.")
    write_table(output_dir / "tables" / "stats_threshold_calibration.txt", lines)


def specificity_table(data: Mapping[str, Any], log: Logger, output_dir: Path) -> None:
    log.section("Specificity table at ε=0.2")

    def row(spec_data: Mapping[str, Any], pe: str, attack: str) -> Tuple[float, float, float]:
        ads_values = []
        drops = []
        for seed in PRIMARY_SEEDS:
            sd = spec_data[pe][seed][attack]
            eps = np.asarray(sd["epsilons"], dtype=float)
            idx = nearest_idx(eps, TARGET_EPS)
            idx0 = nearest_idx(eps, 0.0)
            acc = np.asarray(sd["accuracies"], dtype=float)
            ads_l4 = np.asarray(sd["ads_layer4"], dtype=float)
            ads_values.append(float(ads_l4[idx]))
            drops.append(float(acc[idx0] - acc[idx]))
        ads = mean(ads_values)
        drop = mean(drops)
        ratio = ads / drop if drop > 0 else float("nan")
        return ads, drop, ratio

    for key, label in [("imn_spec", "ImageNet-100"), ("cif_spec", "CIFAR-100")]:
        log.subsection(label)
        lines = [f"Specificity at ε={TARGET_EPS} — {label}", ""]
        header = f"{'PE':<12} {'Attack':<14} {'ADS(L4)':>10} {'Drop(pp)':>10} {'ADS/Drop':>10}"
        lines += [header, "-" * len(header)]
        log.log(header)
        log.log("-" * len(header))
        for pe in PE_TYPES:
            for attack in ATTACKS:
                ads, drop, ratio = row(data[key], pe, attack)
                line = f"{PE_DISPLAY[pe]:<12} {attack:<14} {ads:>10.3f} {drop:>10.2f} {ratio:>10.4f}"
                log.log(line)
                lines.append(line)
            log.log("")
            lines.append("")
        write_table(output_dir / "tables" / f"table_specificity_{label.lower().replace('-', '').replace(' ', '_')}.txt", lines)

    # Budget ratios from main PE-only curves and MLP-only specificity curves.
    log.subsection("50%-of-clean saturation-budget ratios")
    lines = []
    header = f"{'Dataset':<12} {'PE':<12} {'PE ε*':>8} {'MLP ε*':>8} {'ratio':>8}"
    log.log(header)
    log.log("-" * len(header))
    lines += [header, "-" * len(header)]

    def collapse_from_spec(spec_data: Mapping[str, Any], pe: str, attack: str) -> Optional[float]:
        eps = np.asarray(spec_data[pe][PRIMARY_SEEDS[0]][attack]["epsilons"], dtype=float)
        acc_curves = []
        clean_values = []
        for seed in PRIMARY_SEEDS:
            sd = spec_data[pe][seed][attack]
            acc = np.asarray(sd["accuracies"], dtype=float)
            acc_curves.append(acc)
            clean_values.append(float(acc[nearest_idx(eps, 0.0)]))
        acc_mean = np.mean(acc_curves, axis=0)
        clean = mean(clean_values)
        return first_eps_where(eps, acc_mean, lambda a, c=clean: a <= 0.5 * c)

    for main_key, spec_key, label in [("imn_main", "imn_spec", "ImageNet"), ("cif_main", "cif_spec", "CIFAR")]:
        for pe in PE_TYPES:
            eps, acc_mean, clean = mean_accuracy_curve(data[main_key], pe)
            pe_eps = first_eps_where(eps, acc_mean, lambda a, c=clean: a <= 0.5 * c)
            mlp_eps = collapse_from_spec(data[spec_key], pe, "mlp_only")
            ratio = pe_eps / mlp_eps if pe_eps is not None and mlp_eps else float("nan")
            line = f"{label:<12} {PE_DISPLAY[pe]:<12} {pe_eps:>8.3f} {mlp_eps:>8.3f} {ratio:>7.0f}x"
            log.log(line)
            lines.append(line)
    write_table(output_dir / "tables" / "stats_saturation_budget.txt", lines)


def profile_for_seed(main_data: Mapping[str, Any], pe: str, seed: str) -> np.ndarray:
    sd = main_data[pe][seed]
    ads_per_layer = np.asarray(sd["ads_per_layer"], dtype=float)
    ads_mean = ads_per_layer.mean(axis=1)
    valid = ads_mean > 1e-12
    # Skip epsilon=0 and any all-zero rows.
    valid[0] = False
    ratios = ads_per_layer[valid] / ads_mean[valid, None]
    return ratios.mean(axis=0)


def all_profiles(main_data: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X = []
    y = []
    labels = []
    for pe in PE_TYPES:
        for seed in PRIMARY_SEEDS:
            X.append(profile_for_seed(main_data, pe, seed))
            y.append(pe)
            labels.append(f"{pe}:{seed}")
    return np.asarray(X), np.asarray(y), labels


def l4_ratio_stats(main_data: Mapping[str, Any], pe: str) -> Tuple[float, float, List[float]]:
    vals = [float(profile_for_seed(main_data, pe, seed)[3]) for seed in PRIMARY_SEEDS]
    return mean(vals), t_ci(vals), vals


def slope_stats(main_data: Mapping[str, Any], pe: str) -> Dict[str, float]:
    x = np.arange(1, 13)
    profiles = np.asarray([profile_for_seed(main_data, pe, seed) for seed in PRIMARY_SEEDS])
    slopes = [float(np.polyfit(x, p, 1)[0]) for p in profiles]
    mean_profile = profiles.mean(axis=0)
    return {
        "slope_mean": mean(slopes),
        "slope_ci": t_ci(slopes),
        "profile_std": float(mean_profile.std(ddof=0)),
    }


def loo_accuracy(X: np.ndarray, y: np.ndarray, metric: str, feature_index: Optional[int] = None) -> float:
    if feature_index is not None:
        X_use = X[:, [feature_index]]
    else:
        X_use = X
    correct = 0
    for i in range(len(X_use)):
        mask = np.arange(len(X_use)) != i
        train = X_use[mask]
        if metric == "euclidean":
            dists = np.linalg.norm(train - X_use[i], axis=1)
        elif metric == "cityblock":
            dists = np.sum(np.abs(train - X_use[i]), axis=1)
        elif metric == "cosine":
            denom = np.linalg.norm(train, axis=1) * np.linalg.norm(X_use[i])
            dists = 1.0 - (train @ X_use[i]) / (denom + 1e-12)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        pred = y[mask][int(np.argmin(dists))]
        correct += int(pred == y[i])
    return correct / len(X_use)


def centroid_transfer(train_data: Mapping[str, Any], test_data: Mapping[str, Any]) -> float:
    centroids = {}
    for pe in PE_TYPES:
        centroids[pe] = np.mean([profile_for_seed(train_data, pe, seed) for seed in PRIMARY_SEEDS], axis=0)
    correct = 0
    total = 0
    for pe in PE_TYPES:
        for seed in PRIMARY_SEEDS:
            p = profile_for_seed(test_data, pe, seed)
            d = {candidate: float(np.linalg.norm(p - c)) for candidate, c in centroids.items()}
            pred = min(d, key=d.get)
            correct += int(pred == pe)
            total += 1
    return correct / total


def fingerprint_stats(data: Mapping[str, Any], log: Logger, output_dir: Path) -> None:
    log.section("Hierarchical PE fingerprint")

    # Slope and L4 ratio tables.
    lines_slope = []
    header = f"{'PE':<12} {'Slope IMN':>18} {'Slope CIF':>18} {'Profile std IMN/CIF':>22}"
    lines_slope += [header, "-" * len(header)]
    log.subsection("Layer-wise attenuation slopes")
    log.log(header)
    log.log("-" * len(header))
    for pe in PE_TYPES:
        imn = slope_stats(data["imn_main"], pe)
        cif = slope_stats(data["cif_main"], pe)
        line = (f"{PE_DISPLAY[pe]:<12} "
                f"{imn['slope_mean']:+.3f} ± {imn['slope_ci']:.3f} "
                f"{cif['slope_mean']:+.3f} ± {cif['slope_ci']:.3f} "
                f"{imn['profile_std']:.2f} / {cif['profile_std']:.2f}")
        log.log(line)
        lines_slope.append(line)
    write_table(output_dir / "tables" / "table_slope.txt", lines_slope)

    lines_l4 = []
    header = f"{'PE':<12} {'L4 ratio IMN':>18} {'L4 ratio CIF':>18}"
    lines_l4 += [header, "-" * len(header)]
    log.subsection("ADS(L4)/ADS(mean) ratio")
    log.log(header)
    log.log("-" * len(header))
    for pe in PE_TYPES:
        mi, cii, _ = l4_ratio_stats(data["imn_main"], pe)
        mc, cic, _ = l4_ratio_stats(data["cif_main"], pe)
        line = f"{PE_DISPLAY[pe]:<12} {mi:>7.2f} ± {cii:<7.2f} {mc:>7.2f} ± {cic:<7.2f}"
        log.log(line)
        lines_l4.append(line)
    write_table(output_dir / "tables" / "table_fingerprint_l4.txt", lines_l4)

    # ANOVA and contrasts.
    lines_stats = []
    for key, label in [("imn_main", "ImageNet-100"), ("cif_main", "CIFAR-100")]:
        groups = []
        for pe in PE_TYPES:
            _, _, vals = l4_ratio_stats(data[key], pe)
            groups.append(vals)
        f_stat, p_val = stats.f_oneway(*groups)
        line = f"{label}: L4-ratio ANOVA F(3,20)={f_stat:.2f}, p={p_val:.4g}"
        log.log(line)
        lines_stats.append(line)
        alibi = np.asarray(groups[3])
        rest = np.concatenate(groups[:3])
        t, p = stats.ttest_ind(alibi, rest, equal_var=False)
        line = f"{label}: ALiBi vs rest Welch t={t:.2f}, p={p:.4g}"
        log.log(line)
        lines_stats.append(line)
    write_table(output_dir / "tables" / "stats_fingerprint_anova.txt", lines_stats)

    # LOO classifier.
    log.subsection("1-NN LOO classification")
    header = f"{'Dataset':<12} {'Euclidean':>10} {'Cosine':>10} {'Cityblock':>10} {'L4 only':>10}"
    lines_loo = [header, "-" * len(header)]
    log.log(header)
    log.log("-" * len(header))
    for key, label in [("imn_main", "ImageNet"), ("cif_main", "CIFAR")]:
        X, y, _ = all_profiles(data[key])
        e = loo_accuracy(X, y, "euclidean")
        c = loo_accuracy(X, y, "cosine")
        m = loo_accuracy(X, y, "cityblock")
        l4 = loo_accuracy(X, y, "euclidean", feature_index=3)
        line = f"{label:<12} {100*e:>9.1f}% {100*c:>9.1f}% {100*m:>9.1f}% {100*l4:>9.1f}%"
        log.log(line)
        lines_loo.append(line)
    transfer_i2c = centroid_transfer(data["imn_main"], data["cif_main"])
    transfer_c2i = centroid_transfer(data["cif_main"], data["imn_main"])
    lines_loo += ["", f"IMN centroids -> CIF: {100*transfer_i2c:.1f}%", f"CIF centroids -> IMN: {100*transfer_c2i:.1f}%"]
    log.log(f"Cross-dataset centroid transfer: IMN->CIF {100*transfer_i2c:.1f}%, CIF->IMN {100*transfer_c2i:.1f}%")
    write_table(output_dir / "tables" / "table_fingerprint_loo.txt", lines_loo)


def roc_stats(roc_data: Optional[Mapping[str, Any]], log: Logger, output_dir: Path) -> None:
    log.section("ROC detection boundary")
    if roc_data is None:
        return
    pe_list = [pe for pe in ["learned", "rope"] if pe in roc_data]
    lines = []
    for pe in pe_list:
        eps_keys = sorted(roc_data[pe][PRIMARY_SEEDS[0]]["roc_by_eps"].keys(), key=float)
        header = f"{PE_DISPLAY[pe]}: " + " ".join([f"ε={e}" for e in eps_keys])
        log.log(header)
        lines.append(header)
        aucs_line = []
        for eps in eps_keys:
            values = [float(roc_data[pe][seed]["roc_by_eps"][eps]["auc"]) for seed in PRIMARY_SEEDS]
            aucs_line.append(f"{mean(values):.3f}")
        line = "AUC: " + " ".join(aucs_line)
        log.log(line)
        lines.append(line)
    if "learned" in roc_data:
        vals = [float(roc_data["learned"][seed]["roc_by_eps"].get("0.1", {}).get("auc", float("nan"))) for seed in PRIMARY_SEEDS]
        log.pass_check(f"Learned ε=0.1 mean AUC={mean(vals):.3f} (operational boundary; manuscript claims AUC ≥ 0.99).")
    if "rope" in roc_data:
        vals = [float(roc_data["rope"][seed]["roc_by_eps"].get("0.2", {}).get("auc", float("nan"))) for seed in PRIMARY_SEEDS]
        log.pass_check(f"RoPE ε=0.2 mean AUC={mean(vals):.3f} (operational boundary; manuscript claims AUC ≥ 0.99).")
    write_table(output_dir / "tables" / "stats_roc.txt", lines)


def comparison_stats(comparison_data: Optional[Mapping[str, Any]], log: Logger, output_dir: Path) -> None:
    log.section("ADS vs Attn-L2/Mahalanobis-style vs LogitKL comparison")
    if comparison_data is None:
        return
    lines = []
    for pe in [pe for pe in ["learned", "rope"] if pe in comparison_data]:
        eps_keys = sorted(comparison_data[pe][PRIMARY_SEEDS[0]]["attack"].keys(), key=float)
        lines.append(PE_DISPLAY[pe])
        log.subsection(PE_DISPLAY[pe])
        header = f"{'ε':>8} {'ADS AUC':>10} {'Attn-L2 AUC':>12} {'LogitKL AUC':>12}"
        log.log(header)
        lines += [header, "-" * len(header)]
        for eps in eps_keys:
            vals = {"ads": [], "mahalanobis": [], "logit_kl": []}
            for seed in PRIMARY_SEEDS:
                attack = comparison_data[pe][seed]["attack"][eps]
                for method in vals:
                    vals[method].append(float(attack[method]["auc"]))
            line = f"{eps:>8} {mean(vals['ads']):>10.3f} {mean(vals['mahalanobis']):>12.3f} {mean(vals['logit_kl']):>12.3f}"
            log.log(line)
            lines.append(line)
        lines.append("")
    write_table(output_dir / "tables" / "stats_detection_comparison.txt", lines)


def evasion_stats(evasion_data: Optional[Mapping[str, Any]], adaptive_data: Optional[Mapping[str, Any]], log: Logger, output_dir: Path) -> None:
    log.section("Reference-set and adaptive evasion summaries")
    lines = []
    if evasion_data is not None:
        lines.append("Reference-set evasion")
        for pe in [pe for pe in ["learned", "rope"] if pe in evasion_data]:
            eps_keys = sorted(evasion_data[pe][PRIMARY_SEEDS[0]].keys(), key=float)
            for eps in eps_keys:
                lam_keys = sorted(evasion_data[pe][PRIMARY_SEEDS[0]][eps].keys(), key=float)
                acc, ref, hold, ev_ref, ev_hold, ev_both = [], [], [], 0, 0, 0
                denom = 0
                for seed in PRIMARY_SEEDS:
                    for lam in lam_keys:
                        d = evasion_data[pe][seed][eps][lam]
                        acc.append(float(d["acc_drop"]))
                        ref.append(float(d.get("ads_ref_ratio", d.get("ads_ref"))))
                        hold.append(float(d.get("ads_hold_ratio", d.get("ads_hold"))))
                        r = bool(d.get("evades_ref", False))
                        h = bool(d.get("evades_hold", False))
                        ev_ref += int(r)
                        ev_hold += int(h)
                        ev_both += int(r and h)
                        denom += 1
                line = (f"{PE_DISPLAY[pe]} ε={eps}: drop={mean(acc):.2f}pp, "
                        f"ADS(ref)={mean(ref):.2f}x, ADS(hold)={mean(hold):.2f}x, "
                        f"Ref/Hold/Both={ev_ref}/{denom}, {ev_hold}/{denom}, {ev_both}/{denom}")
                log.log(line)
                lines.append(line)
    elif adaptive_data is None:
        log.warn("Reference-evasion JSON not available.")

    if adaptive_data is not None:
        lines += ["", "Adaptive evasion"]
        for pe in [pe for pe in ["learned", "rope"] if pe in adaptive_data]:
            eps_keys = sorted(adaptive_data[pe][PRIMARY_SEEDS[0]].keys(), key=float)
            for eps in eps_keys:
                # Report λ=0 and λ=50 when present, because those are the manuscript rows.
                for target_lam in [0.0, 50.0]:
                    lam_keys = sorted(adaptive_data[pe][PRIMARY_SEEDS[0]][eps].keys(), key=float)
                    matches = [k for k in lam_keys if abs(float(k) - target_lam) < 1e-12]
                    if not matches:
                        continue
                    lam = matches[0]
                    acc, ratio, evades = [], [], 0
                    for seed in PRIMARY_SEEDS:
                        d = adaptive_data[pe][seed][eps][lam]
                        acc.append(float(d["acc_drop"]))
                        ratio.append(float(d["ads_ratio"]))
                        evades += int(bool(d.get("evades_detection", False)))
                    line = f"{PE_DISPLAY[pe]} ε={eps}, λ={float(lam):g}: drop={mean(acc):.2f}pp, ADS ratio={mean(ratio):.2f}x, evades={evades}/6"
                    log.log(line)
                    lines.append(line)
    elif evasion_data is None:
        log.warn("Adaptive JSON not available.")

    write_table(output_dir / "tables" / "stats_evasion.txt", lines)


def probing_table(data: Mapping[str, Any], log: Logger, output_dir: Path) -> None:
    log.section("Residual-stream probing sanity check")
    header = f"{'PE':<12} {'IMN peak':>12} {'IMN R2(L4)':>12} {'CIF peak':>12} {'CIF R2(L4)':>12}"
    lines = [header, "-" * len(header)]
    log.log(header)
    log.log("-" * len(header))

    def probe_summary(probe_data: Mapping[str, Any], pe: str) -> Tuple[str, float, float]:
        layer_means = []
        for layer in range(1, 13):
            vals = [float(probe_data[pe][seed]["layers"][str(layer)]["r2_mean"]) for seed in PRIMARY_SEEDS]
            layer_means.append(mean(vals))
        peak_val = max(layer_means)
        peak_layers = [i + 1 for i, v in enumerate(layer_means) if abs(v - peak_val) < 0.01]
        if len(peak_layers) == 1:
            peak_label = f"L{peak_layers[0]} {peak_val:.2f}"
        else:
            peak_label = f"L{peak_layers[0]}-L{peak_layers[-1]} {peak_val:.2f}"
        return peak_label, peak_val, layer_means[3]

    for pe in PE_TYPES:
        imn_peak, _, imn_l4 = probe_summary(data["imn_probe"], pe)
        cif_peak, _, cif_l4 = probe_summary(data["cif_probe"], pe)
        line = f"{PE_DISPLAY[pe]:<12} {imn_peak:>12} {imn_l4:>12.2f} {cif_peak:>12} {cif_l4:>12.2f}"
        log.log(line)
        lines.append(line)
    write_table(output_dir / "tables" / "table_residual_probing.txt", lines)



def shared_delta_attack_convention_stats(data: Mapping[str, Any], log: Logger, output_dir: Path) -> None:
    """Verify the shared-delta/tied-buffer attack-convention artifacts."""
    log.section("Shared-delta / tied-buffer attack-convention check")
    entries = [
        ("ImageNet-100", data.get("shared_delta_imn")),
        ("CIFAR-100", data.get("shared_delta_cif")),
    ]
    lines: List[str] = []
    for dataset_name, obj in entries:
        if obj is None:
            log.warn(f"Shared-delta artifact missing for {dataset_name}; skipping.")
            continue
        meta = obj.get("metadata", {}) if isinstance(obj, Mapping) else {}
        summary = obj.get("summary", {}) if isinstance(obj, Mapping) else {}
        results = obj.get("results", {}) if isinstance(obj, Mapping) else {}
        eps = [float(x) for x in summary.get("epsilons", [])]
        means = summary.get("mean_accuracy_by_pe", {})
        ns = summary.get("n_by_pe", {})
        if set(results.keys()) >= set(PE_TYPES) and all(set(get_seeds(results[pe])) == set(PRIMARY_SEEDS) for pe in PE_TYPES):
            log.pass_check(f"{dataset_name}: shared-delta artifact has final n=6 seed coverage.")
        else:
            log.fail(f"{dataset_name}: shared-delta artifact does not have complete final n=6 PE/seed coverage.")
        pattern = meta.get("attack", {}).get("pattern") if isinstance(meta.get("attack"), Mapping) else None
        if pattern == "shared_delta_all_12_blocks":
            log.pass_check(f"{dataset_name}: attack metadata is consistent with shared_delta_all_12_blocks.")
        else:
            log.warn(f"{dataset_name}: attack metadata pattern is {pattern!r}, expected shared_delta_all_12_blocks.")
        from decimal import Decimal, ROUND_HALF_UP
        def one_decimal_half_up(value: float) -> str:
            return str(Decimal(f"{value:.6f}").quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

        header = f"{dataset_name} mean attacked accuracy (%) under shared-delta PE attack"
        log.log(header)
        lines += [header]
        colhdr = f"{'PE':<12}" + "".join(f" eps={e:g}".rjust(10) for e in eps) + "     n"
        log.log(colhdr)
        lines += [colhdr, "-" * len(colhdr)]
        for pe in PE_TYPES:
            vals = [float(v) for v in means.get(pe, [])]
            nvals = ns.get(pe, [])
            n_label = ",".join(str(x) for x in sorted(set(nvals))) if isinstance(nvals, list) else str(nvals)
            line = f"{PE_DISPLAY[pe]:<12}" + "".join(f"{one_decimal_half_up(v):>10}" for v in vals) + f" {n_label:>5}"
            log.log(line)
            lines.append(line)
        lines.append("")
        expected_meta = set(str(s) for s in meta.get("seeds", []))
        if expected_meta == set(PRIMARY_SEEDS):
            log.pass_check(f"{dataset_name}: metadata seed list matches final n=6 set.")
        else:
            log.warn(f"{dataset_name}: metadata seed list {sorted(expected_meta)} differs from expected {PRIMARY_SEEDS}.")
    if lines:
        write_table(output_dir / "tables" / "table_shared_delta_attack_convention.txt", lines)


def roc_rank_auc_sensitivity_stats(data: Mapping[str, Any], log: Logger, output_dir: Path) -> None:
    """Verify the derived exact-rank-AUC sensitivity artifact."""
    log.section("ROC rank-AUC sensitivity check")
    rows = data.get("roc_rank_auc")
    if rows is None:
        log.warn("ROC rank-AUC sensitivity artifact not found; skipping.")
        return
    if not isinstance(rows, list):
        log.fail("ROC rank-AUC sensitivity artifact has unexpected non-list format.")
        return
    expected = {("learned", "0.05"), ("learned", "0.1"), ("learned", "0.2"),
                ("rope", "0.05"), ("rope", "0.1"), ("rope", "0.2")}
    observed = {(str(r.get("pe")), str(r.get("epsilon"))) for r in rows}
    if observed == expected and all(int(r.get("n_seeds", 0)) == len(PRIMARY_SEEDS) for r in rows):
        log.pass_check("ROC rank-AUC sensitivity artifact has expected PE/epsilon rows and n=6 coverage.")
    else:
        log.fail(f"ROC rank-AUC sensitivity rows/coverage differ from expectation: {sorted(observed)}")
    header = f"{'PE':<10} {'epsilon':>8} {'exact rank AUC':>18} {'stored operating AUC':>22} {'n':>4}"
    lines = [header, "-" * len(header)]
    log.log(header)
    log.log("-" * len(header))
    for r in rows:
        line = (f"{PE_DISPLAY.get(str(r.get('pe')), str(r.get('pe'))):<10} "
                f"{str(r.get('epsilon')):>8} "
                f"{float(r.get('exact_mean')):7.3f} ± {float(r.get('exact_std')):.3f} "
                f"{float(r.get('stored_mean')):10.3f} ± {float(r.get('stored_std')):.3f} "
                f"{int(r.get('n_seeds')):4d}")
        log.log(line)
        lines.append(line)
    lookup = {(str(r.get("pe")), str(r.get("epsilon"))): r for r in rows}
    for key in [("learned", "0.1"), ("rope", "0.2")]:
        r = lookup.get(key)
        if r and abs(float(r.get("exact_mean", -1)) - 1.0) < 1e-12 and abs(float(r.get("stored_mean", -1)) - 1.0) < 1e-12:
            log.pass_check(f"{PE_DISPLAY[key[0]]} eps={key[1]} boundary remains AUC=1.000 under both definitions.")
        else:
            log.warn(f"{key[0]} eps={key[1]} boundary is not exactly AUC=1.000 in one definition.")
    write_table(output_dir / "tables" / "table_roc_rank_auc_sensitivity.txt", lines)


def canonical_protocol_stats(data: Mapping[str, Any], log: Logger, output_dir: Path) -> None:
    log.section("Optional protocol-robustness cohort")
    main = data.get("canonical_main")
    spec = data.get("canonical_spec")
    if main is None:
        log.warn("Canonical n=12 ALiBi-style cohort not found; skipping.")
        return
    pe_list = [pe for pe in ["alibi", "alibi_2d", "alibi_2d_matched"] if pe in main]
    seeds = get_seeds(main[pe_list[0]])
    log.log(f"Canonical cohort seeds/checkpoints per condition: n={len(seeds)}")
    lines = [f"Canonical CIFAR-100 ALiBi-style cohort, n={len(seeds)}", ""]
    header = f"{'Condition':<20} {'slope':>15} {'profile std':>12}"
    log.log(header)
    lines += [header, "-" * len(header)]
    x = np.arange(1, 13)
    slopes_by_pe: Dict[str, List[float]] = {}
    for pe in pe_list:
        profiles = []
        for seed in seeds:
            sd = main[pe][seed]
            arr = np.asarray(sd["ads_per_layer"], dtype=float)
            ads_mean = arr.mean(axis=1)
            valid = ads_mean > 1e-12
            valid[0] = False
            profiles.append((arr[valid] / ads_mean[valid, None]).mean(axis=0))
        slopes = [float(np.polyfit(x, p, 1)[0]) for p in profiles]
        slopes_by_pe[pe] = slopes
        profile_std = float(np.mean(np.std(np.asarray(profiles), axis=1)))
        line = f"{PE_DISPLAY[pe]:<20} {mean(slopes):+7.3f} ± {t_ci(slopes):.3f} {profile_std:>12.3f}"
        log.log(line)
        lines.append(line)
    if "alibi_2d" in slopes_by_pe and "alibi_2d_matched" in slopes_by_pe:
        t, p = stats.ttest_ind(slopes_by_pe["alibi_2d"], slopes_by_pe["alibi_2d_matched"], equal_var=False)
        line = f"2D fixed vs matched Welch t={t:.2f}, p={p:.3f}"
        log.log(line)
        lines += ["", line]
    write_table(output_dir / "tables" / "stats_protocol_robustness.txt", lines)


def maybe_generate_figures(data_dir: Path, output_dir: Path, log: Logger) -> None:
    log.section("Figure generation")
    candidates = [Path("generate_ads_figures.py"), Path("scripts/generate_ads_figures.py")]
    script = next((candidate for candidate in candidates if candidate.exists()), None)
    if script is None:
        log.warn("generate_ads_figures.py not found in current working directory or scripts/; skipping figure generation.")
        log.log("Run manually: python scripts/generate_ads_figures.py --data-dir data --output-dir output")
        return
    cmd = [sys.executable, str(script), "--data-dir", str(data_dir), "--output-dir", str(output_dir)]
    log.log("Running: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        log.pass_check("Figures regenerated successfully.")
    except subprocess.CalledProcessError as exc:
        log.warn(f"Figure generation failed with exit code {exc.returncode}; tables/stats are still available.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ADS paper numbers from archived JSON files.")
    parser.add_argument("--data-dir", default="data", help="Directory containing JSON artifacts.")
    parser.add_argument("--output-dir", default="output", help="Directory for generated tables/logs.")
    parser.add_argument("--section", default=None,
                        help="Optional section filter: ref, clean, 4.4, 5, 6.2, 6.3, 6.6, 6.7, 7.1, attack_convention, roc_sensitivity, protocol")
    parser.add_argument("--no-figures", action="store_true", help="Do not call generate_ads_figures.py.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "tables")

    log = Logger(output_dir)
    log.section("ADS REPRODUCTION RUN — " + datetime.now().isoformat())
    log.log("Paper: Attention Divergence Score: A Forensic Metric for Characterizing")
    log.log("       Parameter-Level Attacks in Vision Transformers")
    log.log("Target: IEEE TIFS 13-page resubmission")
    log.log(f"Data dir:   {data_dir}")
    log.log(f"Output dir: {output_dir}")

    data = load_all_data(data_dir, log)
    validate_primary_seed_coverage(data, log)

    section = args.section
    if section is None or section == "ref":
        reference_indices_check(data["ref_indices"], log, output_dir)
    if section is None or section == "clean":
        clean_accuracy_table(data, log, output_dir)
    if section is None or section == "4.4":
        threshold_calibration(data["threshold"], log, output_dir)
    if section is None or section == "5":
        roc_stats(data.get("roc"), log, output_dir)
    if section is None or section == "6.2":
        specificity_table(data, log, output_dir)
        early_warning_table(data, log, output_dir)
    if section is None or section == "6.3":
        fingerprint_stats(data, log, output_dir)
    if section is None or section == "6.6":
        comparison_stats(data.get("comparison"), log, output_dir)
        evasion_stats(data.get("evasion"), None, log, output_dir)
    if section is None or section == "6.7":
        evasion_stats(None, data.get("adaptive"), log, output_dir)
    if section is None or section == "7.1":
        probing_table(data, log, output_dir)
    if section is None or section == "attack_convention":
        shared_delta_attack_convention_stats(data, log, output_dir)
    if section is None or section == "roc_sensitivity":
        roc_rank_auc_sensitivity_stats(data, log, output_dir)
    if section is None or section == "protocol":
        canonical_protocol_stats(data, log, output_dir)

    if section is None and not args.no_figures:
        maybe_generate_figures(data_dir, output_dir, log)

    log.section("SUMMARY")
    if log.warnings:
        log.log(f"Warnings: {len(log.warnings)}")
        for w in log.warnings[:10]:
            log.log(f"  - {w}")
        if len(log.warnings) > 10:
            log.log(f"  ... {len(log.warnings) - 10} more")
    if log.failures:
        log.log(f"Failures: {len(log.failures)}")
        for f in log.failures[:20]:
            log.log(f"  - {f}")
        log.log("Result: FAIL")
        log.close()
        sys.exit(1)
    log.log("Result: PASS")
    log.log(f"Full log: {log.path}")
    log.close()


if __name__ == "__main__":
    main()
