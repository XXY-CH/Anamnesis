#!/usr/bin/env python3
"""Generate scaling law exponent figure for the Anamnesis paper."""
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
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

C_ANAM = '#2563EB'
C_RET  = '#059669'
C_TF   = '#DC2626'

# Data: PPL at 5K, 10K, 20K (Shakespeare char-level, d=128, seed=42)
steps = np.array([5, 10, 20])
anam = np.array([3.12, 1.88, 1.51])
retnet = np.array([4.11, 2.62, 2.07])
tf = np.array([3.94, 2.93, 2.62])

def power_law(steps_arr, a, b):
    return a * steps_arr ** b

# Fit a using the first data point
a_anam = anam[0] / steps[0] ** (-0.523)
a_ret = retnet[0] / steps[0] ** (-0.495)
a_tf = tf[0] / steps[0] ** (-0.294)

s_fine = np.linspace(4, 22, 100)

fig, ax = plt.subplots(figsize=(7, 5))

# Data points
ax.plot(steps, tf, 'o', color=C_TF, markersize=10, zorder=5, label='Transformer')
ax.plot(steps, retnet, 's', color=C_RET, markersize=10, zorder=5, label='RetNet 8h+lw')
ax.plot(steps, anam, 'D', color=C_ANAM, markersize=10, zorder=5, label='Anamnesis')

# Power law fits
ax.plot(s_fine, power_law(s_fine, a_tf, -0.294), '--', color=C_TF, alpha=0.6, linewidth=1.5,
        label=f'Transformer fit ($b$={-0.294:.3f})')
ax.plot(s_fine, power_law(s_fine, a_ret, -0.495), '--', color=C_RET, alpha=0.6, linewidth=1.5,
        label=f'RetNet fit ($b$={-0.495:.3f})')
ax.plot(s_fine, power_law(s_fine, a_anam, -0.523), '--', color=C_ANAM, alpha=0.6, linewidth=1.5,
        label=f'Anamnesis fit ($b$={-0.523:.3f})')

ax.set_xlabel('Training Steps (K)')
ax.set_ylabel('Validation PPL')
ax.set_title('Scaling Law: PPL $\\propto$ steps$^b$\n(Shakespeare char-level, d=128, lr=$10^{-3}$, seed=42)')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, which='both')

# Annotate exponents
ax.annotate('RetNet $\\sim$1.7$\\times$\nsteeper than Transformer',
           xy=(12, power_law(12, a_ret, -0.495)),
           xytext=(15, 3.0),
           fontsize=10, color=C_RET,
           arrowprops=dict(arrowstyle='->', color=C_RET, lw=1.5))

plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(OUT, f'fig_paper_scaling_law.{ext}'), bbox_inches='tight')
plt.close()
print('fig_paper_scaling_law done')
