"""
generate_ads_figures_n6_compact_figurestar.py

Compact figure* versions of the ADS paper figures.

Design goals:
  - readable when inserted as two-column IEEE figure*;
  - reduced vertical whitespace;
  - no large in-figure suptitles (captions carry the title);
  - Fig. 2 and Fig. 3 optimized for 8-panel figure* placement;
  - robust Google Drive handling: write locally first, then copy to Drive.

Outputs both PNG (300 dpi) and PDF:
  ads_fig1_l4_vs_epsilon_compact_large.{png,pdf}
  ads_fig2_per_layer_heatmap_compact_large.{png,pdf}
  ads_fig3_layer_profile_compact_large.{png,pdf}
  ads_fig4_early_warning_combined_compact_large.{png,pdf}
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="Generate compact ADS figure* figures.")
parser.add_argument("--data-dir", default="data",
                    help="Directory containing ads_results.json and ads_results_cifar100.json")
parser.add_argument("--output-dir", default="output",
                    help="Directory where figures/ will be created")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
FINAL_FIG_DIR = Path(args.output_dir) / "figures"
FINAL_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colab/Drive FUSE can occasionally lose direct matplotlib writes.
# Write locally first, then copy to Drive with verification.
if str(FINAL_FIG_DIR).startswith("/content/drive/"):
    FIG_DIR = Path("/content/ads_figures_compact_tmp") / "figures"
    if FIG_DIR.exists():
        shutil.rmtree(FIG_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
else:
    FIG_DIR = FINAL_FIG_DIR


# ============================================================
# Style
# ============================================================
FONT_SIZE = 16
PANEL_TITLE_SIZE = 15
TICK_SIZE = 13
SMALL_TICK_SIZE = 12
SLOPE_BOX_SIZE = 14

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": PANEL_TITLE_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": FONT_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SAVEFIG_KWARGS = dict(dpi=300, bbox_inches="tight")
LEGEND_UPPER_LEFT = dict(
    loc="upper left",
    fontsize=FONT_SIZE,
    frameon=True,
    fancybox=True,
    framealpha=0.94,
    borderpad=0.35,
    labelspacing=0.35,
    handlelength=1.8,
)


# ============================================================
# Load data
# ============================================================
imn_path = DATA_DIR / "ads_results.json"
cif_path = DATA_DIR / "ads_results_cifar100.json"

if not imn_path.exists():
    raise FileNotFoundError(f"Missing {imn_path}")
if not cif_path.exists():
    raise FileNotFoundError(f"Missing {cif_path}")

with open(imn_path) as f:
    imagenet = json.load(f)
with open(cif_path) as f:
    cifar = json.load(f)

print(f"Loaded data from {DATA_DIR.resolve()}/")
if FIG_DIR != FINAL_FIG_DIR:
    print(f"Writing local temp figures to {FIG_DIR.resolve()}/")
    print(f"Final copy target: {FINAL_FIG_DIR.resolve()}/")
else:
    print(f"Saving figures to {FIG_DIR.resolve()}/")
print()

pe_types = ["learned", "sinusoidal", "rope", "alibi"]
PE_LABELS = {
    "learned": "Learned PE",
    "sinusoidal": "Sinusoidal PE",
    "rope": "RoPE",
    "alibi": "ALiBi",
}
PE_SHORT = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "rope": "RoPE",
    "alibi": "ALiBi",
}
DATA_SHORT = {
    "ImageNet-100": "IMN",
    "CIFAR-100": "CIF",
}
PE_COLORS = {
    "learned": "#7B68EE",
    "sinusoidal": "#00CED1",
    "rope": "#FF6347",
    "alibi": "#32CD32",
}
PE_MARKERS = {
    "learned": "o",
    "sinusoidal": "s",
    "rope": "^",
    "alibi": "D",
}


def get_seeds_for(dataset, pe):
    if pe not in dataset:
        raise KeyError(f"PE type '{pe}' not found. Available: {list(dataset.keys())}")
    return sorted(dataset[pe].keys(), key=lambda s: int(s))


def _print_seed_banner():
    seed_inventory = {}
    for ds_name, data in [("ImageNet-100", imagenet), ("CIFAR-100", cifar)]:
        for pe in pe_types:
            if pe in data:
                seed_inventory[(ds_name, pe)] = get_seeds_for(data, pe)
    all_unique = {tuple(v) for v in seed_inventory.values()}
    if len(all_unique) == 1:
        seeds_str = ", ".join(next(iter(all_unique)))
        print(f"Seeds detected (all PE×datasets): {seeds_str}")
    else:
        print("Seeds detected (varies by PE×dataset):")
        for (ds, pe), s in seed_inventory.items():
            print(f"  {ds:14s} {pe:11s} → {', '.join(s)}")
    print()


_print_seed_banner()

_first_ds = imagenet if "learned" in imagenet else cifar
_first_pe = next(iter(_first_ds.keys()))
_first_seed = next(iter(_first_ds[_first_pe].keys()))
epsilons = _first_ds[_first_pe][_first_seed]["epsilons"]


def fmt_eps(e):
    """Compact epsilon labels for crowded heatmaps."""
    if abs(e) < 1e-12:
        return "0"
    if e >= 1:
        return "1"
    s = f"{e:g}"
    return s[1:] if s.startswith("0.") else s


def get_mean_std(data, pe, metric):
    seeds_here = get_seeds_for(data, pe)
    vals = np.array([[data[pe][s][metric][i] for s in seeds_here]
                     for i in range(len(epsilons))])
    if vals.shape[1] >= 2:
        return vals.mean(1), vals.std(1, ddof=1)
    return vals.mean(1), np.zeros(vals.shape[0])


def save_current_figure(base_name):
    """Save current matplotlib figure as PNG and PDF."""
    png = FIG_DIR / f"{base_name}.png"
    pdf = FIG_DIR / f"{base_name}.pdf"
    plt.savefig(png, **SAVEFIG_KWARGS)
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"✓ saved: {png.name} and {pdf.name}")


# ============================================================
# FIG 1 COMPACT: ADS(L4) vs ε — figure*
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.45))

for ax, (ds_name, data) in zip(axes, [("ImageNet-100", imagenet), ("CIFAR-100", cifar)]):
    for pe in pe_types:
        mean, std = get_mean_std(data, pe, "ads_layer4")
        ax.plot(epsilons, mean, marker=PE_MARKERS[pe], color=PE_COLORS[pe],
                linewidth=2.25, markersize=7.0, label=PE_LABELS[pe])
        ax.fill_between(epsilons, mean - std, mean + std,
                        alpha=0.14, color=PE_COLORS[pe])

    ax.set_title(ds_name, fontweight="bold")
    ax.set_xlabel("Perturbation budget $\\varepsilon$ ($L_\\infty$)")
    ax.set_ylabel("ADS(L4) KL divergence")
    ax.legend(**LEGEND_UPPER_LEFT)
    ax.grid(True, alpha=0.28)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(0.0009, 1.1)

fig.subplots_adjust(left=0.070, right=0.995, bottom=0.17, top=0.89, wspace=0.22)
save_current_figure("ads_fig1_l4_vs_epsilon_compact_large")


# ============================================================
# FIG 2 COMPACT: Per-layer ADS heatmap — shared colorbar figure*
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.65))

# Global log norm makes panels comparable and removes per-panel colorbars.
all_mats = []
for pe in pe_types:
    for _, data in [("ImageNet-100", imagenet), ("CIFAR-100", cifar)]:
        seeds_here = get_seeds_for(data, pe)
        matrix = np.array([
            np.mean([data[pe][s]["ads_per_layer"][i] for s in seeds_here], axis=0)
            for i in range(len(epsilons))
        ])
        all_mats.append(matrix)
global_min = min(m[m > 0].min() for m in all_mats if (m > 0).any())
global_max = max(m.max() for m in all_mats)
norm = mcolors.LogNorm(vmin=max(global_min, 1e-5), vmax=max(global_max, 1e-4))

# The first epsilon column is the clean reference (epsilon = 0), so ADS is
# exactly zero there. LogNorm cannot display non-positive values; if passed
# directly, matplotlib masks them and renders them with the colormap's "bad"
# color, which can look white/high. Clip non-positive entries to the lower
# log-scale bound so the clean column is shown as the darkest/lowest value.
heatmap_cmap = plt.get_cmap("hot").copy()
heatmap_cmap.set_under("black")
heatmap_cmap.set_bad("black")

im_for_cbar = None
for col, pe in enumerate(pe_types):
    for row, (ds_name, data) in enumerate([("ImageNet-100", imagenet),
                                           ("CIFAR-100", cifar)]):
        ax = axes[row, col]
        seeds_here = get_seeds_for(data, pe)
        matrix = np.array([
            np.mean([data[pe][s]["ads_per_layer"][i] for s in seeds_here], axis=0)
            for i in range(len(epsilons))
        ])

        # Use a display-only copy: keep the data unchanged, but render
        # epsilon=0 / exact-zero ADS as the lower end of the log scale.
        matrix_plot = np.where(matrix <= 0, norm.vmin, matrix)
        im = ax.imshow(matrix_plot.T, aspect="auto", cmap=heatmap_cmap,
                       interpolation="nearest", norm=norm)
        im_for_cbar = im

        ax.set_title(f"{PE_SHORT[pe]} — {DATA_SHORT[ds_name]}",
                     color=PE_COLORS[pe], fontweight="bold",
                     fontsize=PANEL_TITLE_SIZE)

        ax.set_xticks(range(len(epsilons)))
        if row == 1:
            ax.set_xticklabels([fmt_eps(e) for e in epsilons],
                               rotation=55, ha="right", fontsize=SMALL_TICK_SIZE)
            ax.set_xlabel("$\\varepsilon$", fontsize=PANEL_TITLE_SIZE)
        else:
            ax.set_xticklabels([])

        layer_ticks = [0, 3, 7, 11]
        ax.set_yticks(layer_ticks)
        if col == 0:
            ax.set_yticklabels([f"L{i+1}" for i in layer_ticks],
                               fontsize=TICK_SIZE)
            ax.set_ylabel("Layer", fontsize=PANEL_TITLE_SIZE)
        else:
            ax.set_yticklabels([])

fig.subplots_adjust(left=0.048, right=0.932, bottom=0.12,
                    top=0.93, wspace=0.085, hspace=0.22)
cbar = fig.colorbar(im_for_cbar, ax=axes.ravel().tolist(),
                    fraction=0.018, pad=0.012)
cbar.set_label("KL div", fontsize=PANEL_TITLE_SIZE)
cbar.ax.tick_params(labelsize=SMALL_TICK_SIZE)

save_current_figure("ads_fig2_per_layer_heatmap_compact_large")


# ============================================================
# FIG 3 COMPACT: Layer-wise ratio profiles — figure*
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.7))
LAYERS = np.arange(1, 13)

for col, pe in enumerate(pe_types):
    for row, (ds_name, data) in enumerate([("ImageNet-100", imagenet),
                                           ("CIFAR-100", cifar)]):
        ax = axes[row, col]
        seeds_here = get_seeds_for(data, pe)

        per_seed_profiles = []
        for s in seeds_here:
            ads_per_layer = np.array(data[pe][s]["ads_per_layer"])
            ads_mean_layers = ads_per_layer.mean(axis=1)
            profile = []
            for layer in range(12):
                ratios = []
                for ei in range(1, ads_per_layer.shape[0]):
                    if ads_mean_layers[ei] > 1e-10:
                        ratios.append(ads_per_layer[ei, layer] / ads_mean_layers[ei])
                profile.append(np.mean(ratios) if ratios else 0)
            per_seed_profiles.append(profile)

        per_seed_profiles = np.array(per_seed_profiles)
        mean_profile = per_seed_profiles.mean(axis=0)
        std_profile = (per_seed_profiles.std(axis=0, ddof=1)
                       if per_seed_profiles.shape[0] >= 2
                       else np.zeros(12))

        color = PE_COLORS[pe]

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.fill_between(LAYERS, mean_profile - std_profile,
                        mean_profile + std_profile,
                        alpha=0.16, color=color)
        for si in range(per_seed_profiles.shape[0]):
            ax.plot(LAYERS, per_seed_profiles[si], "-",
                    color=color, alpha=0.25, linewidth=0.95)
        ax.plot(LAYERS, mean_profile, marker=PE_MARKERS[pe], color=color,
                linewidth=2.25, markersize=6.0)

        slope, intercept = np.polyfit(LAYERS, mean_profile, 1)
        fit_line = slope * LAYERS + intercept
        ax.plot(LAYERS, fit_line, "--", color=color, alpha=0.68, linewidth=1.15)

        ax.text(0.965, 0.93, f"{slope:+.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=SLOPE_BOX_SIZE, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor=color, alpha=0.9))

        ax.set_title(f"{PE_SHORT[pe]} — {DATA_SHORT[ds_name]}",
                     fontweight="bold", color=color, fontsize=PANEL_TITLE_SIZE)
        ax.grid(True, alpha=0.25)
        ax.set_xticks([1, 3, 6, 9, 12])

        if row == 1:
            ax.set_xlabel("Layer index", fontsize=PANEL_TITLE_SIZE)
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel("ADS(L$_\\ell$)/mean", fontsize=PANEL_TITLE_SIZE)
        else:
            ax.set_yticklabels([])

        # Use a common y-axis across all PE types so normalized profiles are
        # visually comparable and ALiBi is not mistaken for an offset curve.
        ax.set_ylim(-0.2, 4.7)

fig.subplots_adjust(left=0.052, right=0.995, bottom=0.105,
                    top=0.93, wspace=0.14, hspace=0.23)
save_current_figure("ads_fig3_layer_profile_compact_large")


# ============================================================
# FIG 4 COMPACT: ADS-vs-accuracy trajectories — figure*
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.45))

for ax, (ds_name, data) in zip(axes, [("ImageNet-100", imagenet), ("CIFAR-100", cifar)]):
    for pe in pe_types:
        mean_l4 = get_mean_std(data, pe, "ads_layer4")[0]
        mean_acc = get_mean_std(data, pe, "accuracies")[0]
        clean = mean_acc[0]

        acc_loss_norm = (clean - mean_acc) / clean
        ads_norm = mean_l4 / (mean_l4.max() + 1e-10)

        ax.plot(epsilons, ads_norm, marker=PE_MARKERS[pe], color=PE_COLORS[pe],
                linewidth=2.25, markersize=6.3, label=f"{PE_LABELS[pe]} ADS",
                linestyle="-")
        ax.plot(epsilons, acc_loss_norm, color=PE_COLORS[pe],
                linewidth=1.6, linestyle="--", alpha=0.58)

    ax.set_title(f"{ds_name}\nsolid=ADS, dashed=accuracy loss",
                 fontweight="bold", fontsize=PANEL_TITLE_SIZE)
    ax.set_xlabel("Perturbation budget $\\varepsilon$ ($L_\\infty$)")
    ax.set_ylabel("Normalized value")
    ax.legend(**LEGEND_UPPER_LEFT)
    ax.grid(True, alpha=0.28)
    ax.set_xscale("log")
    ax.set_xlim(0.0009, 1.1)

fig.subplots_adjust(left=0.070, right=0.995, bottom=0.17, top=0.87, wspace=0.22)
save_current_figure("ads_fig4_early_warning_combined_compact_large")


# ============================================================
# Robust finalization
# ============================================================
expected_bases = [
    "ads_fig1_l4_vs_epsilon_compact_large",
    "ads_fig2_per_layer_heatmap_compact_large",
    "ads_fig3_layer_profile_compact_large",
    "ads_fig4_early_warning_combined_compact_large",
]
expected_files = [f"{b}.{ext}" for b in expected_bases for ext in ("png", "pdf")]

if FIG_DIR != FINAL_FIG_DIR:
    print()
    print(f"Copying figures to final directory: {FINAL_FIG_DIR.resolve()}/")
    for name in expected_files:
        src_path = FIG_DIR / name
        dst_path = FINAL_FIG_DIR / name
        if not src_path.exists() or src_path.stat().st_size == 0:
            raise RuntimeError(f"Expected local figure missing or empty: {src_path}")
        shutil.copy2(src_path, dst_path)
        if not dst_path.exists() or dst_path.stat().st_size == 0:
            raise RuntimeError(f"Copy failed: {dst_path}")
        print(f"  ✓ {name} ({dst_path.stat().st_size/1024:.1f} KB)")
    try:
        os.sync()
    except Exception:
        pass

print()
print(f"✅ Large compact ADS figures generated in {FINAL_FIG_DIR.resolve()}/")
print("Final directory listing:")
for name in expected_files:
    p = FINAL_FIG_DIR / name
    if p.exists():
        print(f"  {name}: {p.stat().st_size/1024:.1f} KB")
    else:
        print(f"  {name}: MISSING")
