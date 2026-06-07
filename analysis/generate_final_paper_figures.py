#!/usr/bin/env python3
"""Generate final publication-quality figures for the Anamnesis paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    # savefig.bbox_inches set per-figure
})

# Color palette
C_ANAM = '#2563EB'   # blue
C_RET  = '#059669'   # green
C_TF   = '#DC2626'   # red

# ============================================================================
# Figure 1: Complete Scaling Ablation (d=128 + d=256, all models, all steps)
# ============================================================================
def fig_scaling_ablation():
    steps = np.array([5, 10, 20])

    # d=128 data
    anam_128 = [3.12, 1.88, 1.51]
    retnet_128 = [4.11, 2.62, 2.07]
    tf_128 = [3.94, 2.93, 2.62]

    # d=256 data
    anam_256 = [2.42, 1.15, 1.08]
    _retnet_256 = [None, None, 1.114]  # only 20K available (used inline below)
    tf_256 = [3.38, 1.84, 1.30]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # d=128
    ax = axes[0]
    ax.plot(steps, tf_128, 'o-', color=C_TF, linewidth=2, markersize=7, label='Transformer')
    ax.plot(steps, retnet_128, 's--', color=C_RET, linewidth=2, markersize=7, label='RetNet 8h+lw')
    ax.plot(steps, anam_128, 'D-', color=C_ANAM, linewidth=2, markersize=7, label='Anamnesis')
    ax.set_xlabel('Training Steps (K)')
    ax.set_ylabel('Validation PPL')
    ax.set_title('d=128')
    ax.set_xticks(steps)
    ax.set_xticklabels(['5K', '10K', '20K'])
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # d=256
    ax = axes[1]
    ax.plot(steps, tf_256, 'o-', color=C_TF, linewidth=2, markersize=7, label='Transformer')
    # Bare RetNet only at 20K — plot as single point
    ax.plot([20], [1.114], 's', color=C_RET, markersize=8, label='RetNet 8h+lw (20K)')
    ax.plot(steps, anam_256, 'D-', color=C_ANAM, linewidth=2, markersize=7, label='Anamnesis')
    ax.set_xlabel('Training Steps (K)')
    ax.set_ylabel('Validation PPL')
    ax.set_title('d=256')
    ax.set_xticks(steps)
    ax.set_xticklabels(['5K', '10K', '20K'])
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Scaling with Training Duration (Shakespeare char-level, lr=$10^{-3}$, seed=42)', fontsize=13)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT, f'fig_paper_scaling_ablation.{ext}'), bbox_inches='tight')
    plt.close()
    print('fig_paper_scaling_ablation done')

# ============================================================================
# Figure 2: Decomposition — RetNet vs Engram contribution across training
# ============================================================================
def fig_decomposition():
    steps = [5, 10, 20]

    # d=128 decomposition (as % PPL change vs previous model)
    retnet_contribution = [4.3, -10.6, -20.9]
    engram_contribution = [-24.1, -28.2, -27.1]
    total = [r + e for r, e in zip(retnet_contribution, engram_contribution)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(steps))
    width = 0.25

    bars1 = ax.bar(x - width, retnet_contribution, width, label='RetNet+layerwise vs TF',
                   color=C_RET, alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x, engram_contribution, width, label='Engram (additive)',
                   color='#F59E0B', alpha=0.85, edgecolor='white')
    bars3 = ax.bar(x + width, total, width, label='Total Anamnesis vs TF',
                   color=C_ANAM, alpha=0.85, edgecolor='white')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('PPL Change vs Transformer (%)')
    ax.set_title('Advantage Decomposition (Shakespeare d=128, seed=42)')
    ax.set_xticks(x)
    ax.set_xticklabels(['5K', '10K', '20K'])
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            offset = -1.5 if h < 0 else 1.0
            ax.annotate(f'{h:+.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, offset), textcoords='offset points',
                       ha='center', va='top' if h < 0 else 'bottom',
                       fontsize=8)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT, f'fig_paper_decomposition.{ext}'), bbox_inches='tight')
    plt.close()
    print('fig_paper_decomposition done')

# ============================================================================
# Figure 3: Cross-Dataset Diversity Scaling
# ============================================================================
def fig_cross_dataset():
    datasets = ['Shakespeare\n(diverse)', 'WikiText-2\n(moderate)', 'TinyStories\n(repetitive)']
    anam_10k = [1.88, 3.94, 2.38]
    tf_10k = [2.93, 4.00, 2.07]
    advantage = [(a - t) / t * 100 for a, t in zip(anam_10k, tf_10k)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: PPL comparison
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.35
    ax.bar(x - width/2, anam_10k, width, label='Anamnesis', color=C_ANAM, alpha=0.85)
    ax.bar(x + width/2, tf_10k, width, label='Transformer', color=C_TF, alpha=0.85)
    ax.set_ylabel('Validation PPL (10K steps)')
    ax.set_title('10K PPL by Dataset (d=128)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Right: advantage %
    ax = axes[1]
    colors = [C_ANAM if a < 0 else C_TF for a in advantage]
    bars = ax.bar(x, advantage, 0.5, color=colors, alpha=0.85)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Anamnesis vs Transformer (%)')
    ax.set_title('Advantage vs Data Diversity')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, advantage):
        ax.annotate(f'{val:+.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, -12 if val < 0 else 5), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('Advantage Scales with Data Diversity (d=128, 10K steps, lr=$10^{-3}$)', fontsize=12)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT, f'fig_paper_cross_dataset.{ext}'), bbox_inches='tight')
    plt.close()
    print('fig_paper_cross_dataset done')

# ============================================================================
# Figure 4: d=256 20K decomposition — entropy floor
# ============================================================================
def fig_d256_entropy_floor():
    models = ['Transformer', 'RetNet 8h+lw', 'Anamnesis']
    ppls = [1.30, 1.114, 1.08]
    colors = [C_TF, C_RET, C_ANAM]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(models, ppls, color=colors, alpha=0.85, width=0.5, edgecolor='white')

    for bar, val in zip(bars, ppls):
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel('Validation PPL')
    ax.set_title('d=256 20K Decomposition\n(Shakespeare char-level, seed=42)')
    ax.set_ylim(0.9, 1.4)
    ax.grid(True, alpha=0.3, axis='y')

    ax.annotate('', xy=(1, 1.114), xytext=(0, 1.30),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(0.5, 1.21, '-14.3%\n(RetNet)', ha='center', fontsize=9, color='gray')

    ax.annotate('', xy=(2, 1.08), xytext=(1, 1.114),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(1.5, 1.10, '-3.1%\n(Engram)', ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT, f'fig_paper_d256_decomposition.{ext}'), bbox_inches='tight')
    plt.close()
    print('fig_paper_d256_decomposition done')


if __name__ == '__main__':
    fig_scaling_ablation()
    fig_decomposition()
    fig_cross_dataset()
    fig_d256_entropy_floor()
    print('All figures generated.')
