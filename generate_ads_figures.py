"""Generate ADS figures for paper."""
import json, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['legend.fontsize'] = 10

with open('/content/ads_results.json') as f:
    imagenet = json.load(f)
with open('/content/ads_results_cifar100.json') as f:
    cifar = json.load(f)

seeds = ['42', '123', '456']
pe_types = ['learned', 'sinusoidal', 'rope', 'alibi']
PE_LABELS = {'learned': 'Learned PE', 'sinusoidal': 'Sinusoidal PE',
             'rope': 'RoPE', 'alibi': 'ALiBi'}
PE_COLORS = {'learned': '#7B68EE', 'sinusoidal': '#00CED1',
             'rope': '#FF6347', 'alibi': '#32CD32'}
PE_MARKERS = {'learned': 'o', 'sinusoidal': 's', 'rope': '^', 'alibi': 'D'}

epsilons = imagenet['learned']['42']['epsilons']

def get_mean_std(data, pe, metric):
    vals = np.array([[data[pe][s][metric][i] for s in seeds]
                     for i in range(len(epsilons))])
    return vals.mean(1), vals.std(1, ddof=1)

# ============================================================
# FIG 1: ADS(L4) vs ε — 2 datasets side by side
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Attention Divergence Score (Layer 4) vs Perturbation Budget',
             fontsize=14, fontweight='bold', y=1.02)

for ax, (ds_name, data) in zip(axes, [("ImageNet-100", imagenet), ("CIFAR-100", cifar)]):
    for pe in pe_types:
        mean, std = get_mean_std(data, pe, 'ads_layer4')
        ax.plot(epsilons, mean, marker=PE_MARKERS[pe], color=PE_COLORS[pe],
                linewidth=2, markersize=7, label=PE_LABELS[pe])
        ax.fill_between(epsilons, mean-std, mean+std, alpha=0.15, color=PE_COLORS[pe])

    ax.set_xlabel('Perturbation budget ε (L∞)')
    ax.set_ylabel('ADS (Layer 4 KL divergence)')
    ax.set_title(ds_name, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(0.0009, 1.1)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/pe_experiment/results/ads/ads_fig1_l4_vs_epsilon.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Fig 1 saved")


# ============================================================
# FIG 2: Per-layer ADS heatmap — all 4 PE types, both datasets
# ============================================================
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Per-Layer ADS Heatmap (mean over seeds)',
             fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.3)

for col, pe in enumerate(pe_types):
    for row, (ds_name, data) in enumerate([("ImageNet-100", imagenet), ("CIFAR-100", cifar)]):
        ax = fig.add_subplot(gs[row, col])

        # Build matrix: rows=epsilons, cols=layers
        matrix = np.array([
            np.mean([data[pe][s]['ads_per_layer'][i] for s in seeds], axis=0)
            for i in range(len(epsilons))
        ])  # (n_eps, 12)

        data_floor = max(matrix[matrix > 0].min() if (matrix > 0).any() else 1e-5, 1e-5)
        im = ax.imshow(matrix.T, aspect='auto', cmap='hot',
                   interpolation='nearest',
                   norm=matplotlib.colors.LogNorm(
                       vmin=data_floor,
                       vmax=max(matrix.max(), 1e-4)))

        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([str(e) for e in epsilons], rotation=45, fontsize=7)
        ax.set_yticks(range(12))
        ax.set_yticklabels([f'L{i+1}' for i in range(12)], fontsize=8)
        ax.set_xlabel('ε', fontsize=10)
        ax.set_ylabel('Layer', fontsize=10)
        title = f"{PE_LABELS[pe]} — {ds_name}"
        ax.set_title(title, color=PE_COLORS[pe], fontweight='bold', fontsize=10)
        plt.colorbar(im, ax=ax, label='KL div')

plt.savefig('/content/drive/MyDrive/pe_experiment/results/ads/ads_fig2_per_layer_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Fig 2 saved")


# ============================================================
# FIG 3: Layer-wise ADS ratio profile — Hierarchical Fingerprint Level 1
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Layer-wise ADS Ratio Profile by PE Operation Space',
             fontsize=14, fontweight='bold', y=1.00)

LAYERS = np.arange(1, 13)  # L1..L12

for col, pe in enumerate(pe_types):
    for row, (ds_name, data) in enumerate([("ImageNet-100", imagenet),
                                            ("CIFAR-100", cifar)]):
        ax = axes[row, col]

        # Compute per-seed 12-d ratio profile (averaged over ε > 0)
        per_seed_profiles = []
        for s in seeds:
            ads_per_layer = np.array(data[pe][s]['ads_per_layer'])  # (n_eps, 12)
            ads_mean_layers = ads_per_layer.mean(axis=1)            # (n_eps,)
            profile = []
            for layer in range(12):
                ratios = []
                for ei in range(1, ads_per_layer.shape[0]):  # skip ε=0
                    if ads_mean_layers[ei] > 1e-10:
                        ratios.append(ads_per_layer[ei, layer] / ads_mean_layers[ei])
                profile.append(np.mean(ratios) if ratios else 0)
            per_seed_profiles.append(profile)
        per_seed_profiles = np.array(per_seed_profiles)  # (3, 12)

        mean_profile = per_seed_profiles.mean(axis=0)
        std_profile = per_seed_profiles.std(axis=0, ddof=1)

        color = PE_COLORS[pe]

        # (1) Reference line at ratio = 1
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.6,
                   zorder=1)

        # (2) Shaded ±1σ band
        ax.fill_between(LAYERS,
                        mean_profile - std_profile,
                        mean_profile + std_profile,
                        alpha=0.18, color=color, zorder=2)

        # (3) Per-seed thin lines
        for si in range(per_seed_profiles.shape[0]):
            ax.plot(LAYERS, per_seed_profiles[si], '-',
                    color=color, alpha=0.35, linewidth=1, zorder=3)

        # (4) Mean profile with markers
        ax.plot(LAYERS, mean_profile, marker=PE_MARKERS[pe], color=color,
                linewidth=2.2, markersize=6, zorder=4)

        # (5) Linear-fit slope overlay (dashed) and annotation
        slope, intercept = np.polyfit(LAYERS, mean_profile, 1)
        fit_line = slope * LAYERS + intercept
        ax.plot(LAYERS, fit_line, '--', color=color, alpha=0.7, linewidth=1.2,
                zorder=3)

        # Annotate slope value in upper-right corner
        ax.text(0.97, 0.95, f'slope = {slope:+.3f}',
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=10, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9))

        ax.set_xlabel('Layer index', fontsize=10)
        ax.set_ylabel('ADS(L$_\\ell$) / ADS(mean)', fontsize=10)
        ax.set_xticks(LAYERS)
        ax.set_xticklabels([str(l) for l in LAYERS], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{PE_LABELS[pe]} — {ds_name}',
                     fontweight='bold', color=color, fontsize=11)

        # Set common y-limits per PE column for direct visual comparison
        # (ALiBi has narrow range ~[0.5, 1.3]; others wider [0, 4])
        if pe == 'alibi':
            ax.set_ylim(0.4, 1.4)
        else:
            ax.set_ylim(-0.2, 4.7)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/pe_experiment/results/ads/ads_fig3_layer_profile.png', dpi=200,
            bbox_inches='tight')
plt.close()
print("✓ Fig 3 saved")


# ============================================================
# FIG 4: ADS early warning — combined view
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ADS vs. Accuracy Loss Trajectories by PE Operation Space',
                fontsize=14, fontweight='bold', y=1.02)

for ax, (ds_name, data) in zip(axes, [("ImageNet-100", imagenet), ("CIFAR-100", cifar)]):
    for pe in pe_types:
        mean_l4  = get_mean_std(data, pe, 'ads_layer4')[0]
        mean_acc = get_mean_std(data, pe, 'accuracies')[0]
        clean = mean_acc[0]

        # Normalize accuracy loss (0=clean, 1=fully collapsed)
        acc_loss_norm = (clean - mean_acc) / clean

        # Normalize ADS to [0,1]
        ads_norm = mean_l4 / (mean_l4.max() + 1e-10)

        ax.plot(epsilons, ads_norm, marker=PE_MARKERS[pe], color=PE_COLORS[pe],
                linewidth=2, markersize=6, label=f'{PE_LABELS[pe]} ADS', linestyle='-')
        ax.plot(epsilons, acc_loss_norm, color=PE_COLORS[pe],
                linewidth=1.5, markersize=4, linestyle='--', alpha=0.6)

    ax.set_xlabel('Perturbation budget ε (L∞)')
    ax.set_ylabel('Normalized value (0=clean, 1=max)')
    ax.set_title(f'{ds_name}\n(solid=ADS norm, dashed=accuracy loss norm)',
                 fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlim(0.0009, 1.1)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/pe_experiment/results/ads/ads_fig4_early_warning_combined.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Fig 4 saved")


print("\n✅ All 4 ADS figures generated!")