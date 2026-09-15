"""DF-Wild-Cup benchmark figures: accuracy-vs-cost trade-off (current paper) and
the benchmark bar chart and embedding-geometry figure used by the v2 manuscript.
Run from the repository root:  python scripts/plot_dfwildcup_benchmarks.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 400, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})
GREY = '#9aa0a6'
BLUE = '#2f6fb2'
ORANGE = '#d1663a'
DARK = '#22303f'

# ------------------------------------------------------------------
# Figure 1: SP-Cup benchmark accuracy bar chart
# ------------------------------------------------------------------
data = [
    ("VGG-19", 62.25, 'cnn'), ("ResNet-50", 69.00, 'cnn'),
    ("ResNet-34 + FFT", 66.55, 'freq'), ("ResNet-34 + DCT", 71.48, 'freq'),
    ("MobileNet-V2", 78.12, 'cnn'), ("ViT-B", 78.12, 'vit'),
    ("ResNet-34", 80.32, 'cnn'), ("ResNet-34 + Wavelet", 80.44, 'freq'),
    ("RegNet-400mF", 80.92, 'cnn'), ("EfficientNet-B0 + Wavelet", 82.48, 'freq'),
    ("Inception-V3", 84.38, 'cnn'), ("ResNeXt-50", 85.19, 'cnn'),
    ("ResNet-34 + CBAM", 85.19, 'cnn'), ("RegNet-Y + Wavelet", 86.04, 'freq'),
    ("Swin Transformer", 86.52, 'vit'), ("MobileViT", 87.04, 'vit'),
    ("EfficientNet-B4", 87.27, 'cnn'), ("Xception + Wavelet", 87.76, 'freq'),
    ("ConvNeXt", 88.18, 'cnn'), ("EfficientNet-B5", 88.31, 'cnn'),
    ("ResNet-34 + SE", 88.90, 'cnn'), ("DeiT-B", 89.32, 'vit'),
    ("ConvNeXt + Wavelet", 89.49, 'freq'), ("DenseNet-121", 90.01, 'cnn'),
    ("EfficientNet-B0", 90.79, 'cnn'),
    ("CAE-Net (ensemble)", 94.46, 'ref'),
    ("PairCLIP-SWA (ours)", 97.04, 'ours'),
]
data.sort(key=lambda t: t[1])
labels = [d[0] for d in data]
vals = [d[1] for d in data]
kind = [d[2] for d in data]
cmap = {'cnn': GREY, 'vit': '#b9c2cc', 'freq': '#c9b48c', 'ref': ORANGE, 'ours': BLUE}
colors = [cmap[k] for k in kind]

fig, ax = plt.subplots(figsize=(3.42, 4.6))
y = np.arange(len(vals))
ax.barh(y, vals, color=colors, height=0.72, edgecolor='none')
ax.set_yticks(y)
ax.set_yticklabels(labels)
for lbl, k in zip(ax.get_yticklabels(), kind):
    if k in ('ours', 'ref'):
        lbl.set_fontweight('bold')
ax.set_ylim(-0.7, len(vals) - 0.3)
ax.set_xlim(55, 103.5)
ax.set_xlabel('Validation accuracy on SP-Cup 2025 (DF-Wild-Cup), %')
for i, (v, k) in enumerate(zip(vals, kind)):
    ax.text(v + 0.5, i, '%.2f' % v, va='center', ha='left', fontsize=6.2,
            color=DARK if k in ('ours', 'ref') else '#5f6771',
            fontweight='bold' if k in ('ours', 'ref') else 'normal')
ax.axvline(94.46, color=ORANGE, lw=0.7, ls=(0, (3, 2)), zorder=0)
ax.grid(axis='x', lw=0.4, color='#e3e6e8', zorder=0)
ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k])
           for k in ['cnn', 'vit', 'freq', 'ref', 'ours']]
ax.legend(handles,
          ['CNN backbone', 'Transformer backbone', 'Frequency-domain hybrid',
           'CAE-Net ensemble', 'Proposed'],
          loc='upper center', bbox_to_anchor=(0.5, -0.105), ncol=2,
          frameon=False, handlelength=1.1, handleheight=0.9,
          labelspacing=0.3, columnspacing=1.1, borderpad=0.1, fontsize=6.6)
fig.savefig('paper/archive/v2-before-audit/figures/benchmark_bars.pdf')
plt.close(fig)
print('benchmark_bars ok')

# ------------------------------------------------------------------
# Figure 2: accuracy vs. inference cost
# ------------------------------------------------------------------
pts = [
    ("EfficientNet-B0", 3.96, 0.386, 86.11),
    ("DeiT-B", 85.64, 4.608, 89.16),
    ("ConvNeXt+Wavelet", 27.81, 16.863, 91.24),
    ("Eff+DeiT", 89.61, 7.521, 92.59),
    ("Eff+ConvNeXt", 31.77, 17.247, 93.14),
    ("DeiT+ConvNeXt", 113.46, 24.384, 91.87),
    ("CAE-Net", 117.43, 24.768, 94.46),
]
fig, ax = plt.subplots(figsize=(3.42, 2.72))
for n, p_, f, a in pts:
    ax.scatter(f, a, s=14 + p_ * 0.28, color=GREY, edgecolor='#6f7781',
               linewidth=0.5, zorder=3)
ax.scatter(81.01, 97.04, s=14 + 303.97 * 0.28, color=BLUE, edgecolor=DARK,
           linewidth=0.6, zorder=4)
ann = {"EfficientNet-B0": (7, -2, 'left'), "DeiT-B": (7, -2, 'left'),
       "Eff+DeiT": (-6, 5, 'right'), "ConvNeXt+Wavelet": (-6, -4, 'right'),
       "DeiT+ConvNeXt": (8, -8, 'left'), "Eff+ConvNeXt": (8, 0, 'left'),
       "CAE-Net": (8, 4, 'left')}
for n, p_, f, a in pts:
    dx, dy, ha = ann[n]
    ax.annotate(n, (f, a), textcoords='offset points', xytext=(dx, dy),
                ha=ha, fontsize=6, color='#4b535c')
ax.annotate('PairCLIP-SWA (ours)', (81.01, 97.04), textcoords='offset points',
            xytext=(-13, 14), ha='right', fontsize=6.5, color=DARK,
            fontweight='bold')
ax.set_xlabel('Inference cost (GMACs at 224x224 input)')
ax.set_ylabel('SP-Cup accuracy (%)')
ax.set_xscale('log')
ax.set_xlim(0.25, 220)
ax.set_ylim(84, 99.5)
ax.grid(lw=0.4, color='#e3e6e8')
ax.set_axisbelow(True)
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100])
ax.set_xticklabels(['0.5', '1', '2', '5', '10', '20', '50', '100'])
fig.savefig('paper/figures/tradeoff.pdf')
plt.close(fig)
print('tradeoff ok')

# ------------------------------------------------------------------
# Figure 3: embedding geometry and probe agreement
# ------------------------------------------------------------------
ck = ['Best regular', 'Best SWA', 'Final SWA']
sep = [2.033, 1.927, 1.917]
dist = [43.895, 40.726, 39.081]
sig = [10.697 + 10.897, 10.412 + 10.721, 10.318 + 10.071]
knn = [96.00, 97.20, 97.20]
f1 = [95.82, 97.11, 97.12]
heldout = [95.67, 97.04, 97.04]

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35))
fig.subplots_adjust(wspace=0.46)
ax = axes[0]
x = np.arange(3)
ax.plot(x, dist, 's-', color=BLUE, lw=1.2, ms=3.6,
        label=r'Inter-centroid distance $\Delta$')
ax.plot(x, sig, '^-', color='#8d99a6', lw=1.2, ms=3.6,
        label=r'Total dispersion $\sigma_f+\sigma_r$')
for xi, v in zip(x, dist):
    ax.annotate('%.2f' % v, (xi, v), textcoords='offset points', xytext=(0, 6),
                ha='center', fontsize=6.3, color=BLUE)
for xi, v in zip(x, sig):
    ax.annotate('%.2f' % v, (xi, v), textcoords='offset points', xytext=(0, -11),
                ha='center', fontsize=6.3, color='#5f6771')
ax2 = ax.twinx()
ax2.spines['top'].set_visible(False)
ax2.plot(x, sep, 'o--', color=ORANGE, lw=1.1, ms=3.2,
         label=r'Separation score $S$')
for xi, v in zip(x, sep):
    ax2.annotate('%.3f' % v, (xi, v), textcoords='offset points', xytext=(0, 6),
                 ha='center', fontsize=6.3, color=ORANGE)
ax2.set_ylim(1.15, 2.65)
ax2.set_ylabel(r'Separation score $S$', color=ORANGE, labelpad=1)
ax2.tick_params(axis='y', colors=ORANGE, width=0.7)
ax2.set_yticks([1.2, 1.6, 2.0, 2.4])
ax.set_xticks(x)
ax.set_xticklabels(ck)
ax.set_xlim(-0.32, 2.32)
ax.set_ylim(0, 52)
ax.set_ylabel('t-SNE units')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='lower left', frameon=False, handlelength=1.6,
          labelspacing=0.28, fontsize=6.4)
ax.set_title('(a) Embedding geometry', loc='left')
ax.grid(axis='y', lw=0.4, color='#e3e6e8')
ax.set_axisbelow(True)

ax = axes[1]
w = 0.26
ax.bar(x - w, knn, w, color=BLUE, label='k-NN probe accuracy (t-SNE, n=1000)')
ax.bar(x, f1, w, color='#7fa8cf', label='k-NN probe F1 (fake class)')
ax.bar(x + w, heldout, w, color='#bfc7cf', label='Classifier accuracy (n=3072)')
for xs, vs, col in ((x - w, knn, DARK), (x, f1, '#3d5a78'), (x + w, heldout, '#5f6771')):
    for xi, v in zip(xs, vs):
        ax.annotate('%.2f' % v, (xi, v), textcoords='offset points', xytext=(0, 2.5),
                    ha='center', va='bottom', rotation=90, fontsize=5.9, color=col)
ax.set_xticks(x)
ax.set_xticklabels(ck)
ax.set_ylim(93, 99.6)
ax.set_ylabel('%')
ax.legend(loc='lower right', frameon=False, handlelength=1.1, labelspacing=0.25,
          fontsize=6.4)
ax.set_title('(b) Probe agreement with the classifier head', loc='left')
ax.grid(axis='y', lw=0.4, color='#e3e6e8')
ax.set_axisbelow(True)
fig.savefig('paper/archive/v2-before-audit/figures/embedding_geometry.pdf')
plt.close(fig)
print('embedding_geometry ok')
