"""Figures built from the Kaggle execution logs summarized in docs/evidence_audit.pdf:
SP-Cup training trajectory (epochs 12-19) and the in-house design-space study.
Run from the repository root:  python scripts/plot_training_dynamics.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.6,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 400, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})
BLUE = '#2f6fb2'
ORANGE = '#d1663a'
DARK = '#22303f'
GREY = '#9aa0a6'
GREEN = '#2e8b6f'

# ------------------------------------------------------------------
# Figure: SP-Cup training trajectory, epochs 12-19 (clip-vit-l-jan-09 log)
# ------------------------------------------------------------------
ep = np.array([12, 13, 14, 15, 16, 17, 18, 19])
blocks = [8, 8, 9, 9, 10, 10, 11, 11]
tr_loss = [0.1628, 0.0802, 0.1060, 0.0669, 0.0885, 0.0538, 0.0697, 0.0351]
tr_acc = [93.58, 97.01, 95.93, 97.44, 96.65, 97.89, 97.29, 98.48]
ck_loss = [0.1677, 0.1684, 0.2552, 0.2005, 0.2344, 0.3603, 0.2307, 0.2226]
ck_acc = [95.70, 95.61, 91.05, 93.88, 92.22, 90.36, 93.49, 95.67]
sw_loss = [0.1677, 0.1684, 0.1383, 0.1426, 0.1386, 0.1334, 0.1178, 0.1227]
sw_acc = [95.70, 95.61, 95.61, 95.31, 95.80, 96.35, 97.17, 97.04]

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
fig.subplots_adjust(wspace=0.28)


def shade_blocks(ax, ymax):
    spans = [(11.5, 13.5, '8 blocks'), (13.5, 15.5, '9'), (15.5, 17.5, '10'),
             (17.5, 19.5, '11')]
    for i, (a, b, lab) in enumerate(spans):
        if i % 2 == 0:
            ax.axvspan(a, b, color='#eef1f4', zorder=0, lw=0)
        ax.text((a + b) / 2, ymax, lab, ha='center', va='top', fontsize=6.2,
                color='#7b838c')


ax = axes[0]
shade_blocks(ax, 99.3)
ax.plot(ep, tr_acc, ':', color=GREY, lw=1.0, marker='.', ms=3,
        label='Training accuracy (running)')
ax.plot(ep, ck_acc, 'o-', color=ORANGE, lw=1.2, ms=3.2,
        label='Validation, non-averaged checkpoint')
ax.plot(ep, sw_acc, 's-', color=BLUE, lw=1.3, ms=3.2,
        label='Validation, SWA running average')
ax.annotate('90.36', (17, 90.36), textcoords='offset points', xytext=(0, -9),
            ha='center', fontsize=6.3, color=ORANGE)
ax.annotate('', xy=(17, 96.35), xytext=(17, 90.36),
            arrowprops=dict(arrowstyle='<->', lw=0.6, color=DARK))
ax.text(17.12, 93.35, '+5.99', fontsize=6.3, color=DARK, va='center')
ax.set_xlim(11.5, 19.5)
ax.set_ylim(88.5, 99.6)
ax.set_xticks(ep)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('(a) Accuracy on SP-Cup validation ($n{=}3{,}072$)', loc='left')

ax = axes[1]
shade_blocks(ax, 0.435)
ax.plot(ep, tr_loss, ':', color=GREY, lw=1.0, marker='.', ms=3,
        label='Training loss')
ax.plot(ep, ck_loss, 'o-', color=ORANGE, lw=1.2, ms=3.2,
        label='Validation loss, non-averaged')
ax.plot(ep, sw_loss, 's-', color=BLUE, lw=1.3, ms=3.2,
        label='Validation loss, SWA average')
ax.annotate('0.3603', (17, 0.3603), textcoords='offset points', xytext=(6, -1),
            ha='left', va='center', fontsize=6.3, color=ORANGE)
ax.annotate('0.1178', (18, 0.1178), textcoords='offset points', xytext=(0, -10),
            ha='center', fontsize=6.3, color=BLUE)
ax.set_xlim(11.5, 19.5)
ax.set_ylim(0, 0.44)
ax.set_xticks(ep)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-entropy')
ax.set_title('(b) Loss: the running iterate overfits, the average does not',
             loc='left')
h, _ = ax.get_legend_handles_labels()
fig.legend(h, ['Training (running iterate)', 'Validation, non-averaged checkpoint',
               'Validation, SWA running average'],
           loc='lower center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
           handlelength=2.2, columnspacing=1.6)
fig.savefig('paper/figures/trajectory.pdf')
plt.close(fig)
print('trajectory ok')

# ------------------------------------------------------------------
# Figure: in-house design-space study on the SP-Cup validation split
# ------------------------------------------------------------------
runs = [
    # label, val acc, family
    ('ResNet-50, gradual unfreeze', 71.70, 'cnn'),
    ('Xception, undersampled', 75.03, 'cnn'),
    ('Xception, 5 ep', 75.79, 'cnn'),
    ('ViT, near-linear probe', 80.70, 'vit'),
    ('ViT + OpenCLIP + Xception fusion', 81.22, 'fuse'),
    ('ViT + LoRA', 86.75, 'vit'),
    ('ViT, full fine-tune', 89.58, 'vit'),
    ('OpenCLIP B/32, 20 ep', 90.20, 'b32'),
    ('ViT + LoRA + augmentation (peak)', 91.44, 'vit'),
    ('ViT + Swin', 93.03, 'fuse'),
    ('OpenCLIP B/32, 24 ep', 93.07, 'b32'),
    ('OpenCLIP L/14, undersampled', 93.36, 'l14'),
    ('OpenCLIP B/32, 30 ep', 93.62, 'b32'),
    ('OpenCLIP L/14, 8 ep', 94.11, 'l14'),
    ('5 x OpenCLIP B/32 ensemble', 94.95, 'fuse'),
    ('OpenCLIP L/14, earlier checkpoint', 95.51, 'l14'),
    ('PairCLIP, non-averaged', 95.67, 'l14'),
    ('PairCLIP-SWA', 97.04, 'ours'),
]
runs.sort(key=lambda r: r[1])
cmap = {'cnn': '#b9bec4', 'vit': '#c9b48c', 'fuse': '#a8bfd4', 'b32': '#7fa8cf',
        'l14': BLUE, 'ours': ORANGE}
names = {'cnn': 'ImageNet CNN', 'vit': 'Non-CLIP transformer',
         'fuse': 'Fusion / ensemble', 'b32': 'OpenCLIP ViT-B/32',
         'l14': 'OpenCLIP ViT-L/14', 'ours': 'PairCLIP-SWA (reported run)'}

reps = [  # input representation / objective: train, val
    ('FFT-domain input', 83.72, 49.61),
    ('Face crop (dlib) + OpenCLIP', 84.34, 56.05),
    ('Wavelet-coefficient input', 91.60, 73.20),
    ('Triplet metric learning', 98.70, 90.85),
    ('Full frame (reported run)', 98.48, 97.04),
]

fig = plt.figure(figsize=(7.16, 3.55))
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.86)

ax = fig.add_subplot(gs[0])
y = np.arange(len(runs))
ax.barh(y, [r[1] for r in runs], color=[cmap[r[2]] for r in runs], height=0.72,
        edgecolor='none')
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in runs], fontsize=6.4)
for lbl, r in zip(ax.get_yticklabels(), runs):
    if r[2] == 'ours':
        lbl.set_fontweight('bold')
for i, r in enumerate(runs):
    ax.text(r[1] + 0.4, i, '%.2f' % r[1], va='center', fontsize=5.9,
            color=DARK if r[2] == 'ours' else '#5f6771',
            fontweight='bold' if r[2] == 'ours' else 'normal', zorder=4,
            bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'))
ax.axvline(94.46, color=DARK, lw=0.7, ls=(0, (3, 2)), zorder=0)
ax.text(95.5, -0.35, 'CAE-Net 94.46', ha='left', va='bottom', fontsize=5.8,
        color=DARK, rotation=90)
ax.set_xlim(65, 102)
ax.set_ylim(-0.7, len(runs) - 0.3)
ax.set_xlabel('SP-Cup validation accuracy (%)')
ax.set_title('(a) In-house configurations, identical 3,072-image split',
             loc='left', fontsize=7.8)
ax.grid(axis='x', lw=0.4, color='#e3e6e8')
ax.set_axisbelow(True)
keys = ['cnn', 'vit', 'fuse', 'b32', 'l14', 'ours']
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in keys]
ax.legend(handles, [names[k] for k in keys], loc='upper center',
          bbox_to_anchor=(0.32, -0.15), ncol=3, frameon=False, fontsize=6.0,
          handlelength=1.0, handleheight=0.8, columnspacing=0.9,
          labelspacing=0.3)

ax = fig.add_subplot(gs[1])
yy = np.arange(len(reps))
for i, (lab, tr, va) in enumerate(reps):
    col = ORANGE if i == len(reps) - 1 else DARK
    ax.plot([va, tr], [i, i], '-', color='#c3c9cf', lw=2.2, zorder=1)
    ax.scatter([tr], [i], s=20, color='white', edgecolor=col, lw=1.0, zorder=3)
    ax.scatter([va], [i], s=22, color=col, zorder=3)
    ax.text(min(va, tr), i + 0.2, '%.2f' % va, ha='center', va='bottom',
            fontsize=5.9, color=col)
    ax.text(max(va, tr) + 3.2, i, 'gap %.1f' % (tr - va), ha='left',
            va='center', fontsize=5.9, color='#6f7781')
ax.set_yticks(yy)
ax.set_yticklabels([r[0] for r in reps], fontsize=6.4)
ax.get_yticklabels()[-1].set_fontweight('bold')
ax.axvline(50, color='#b9c2cc', lw=0.7, ls=(0, (2, 2)), zorder=0)
ax.text(50.8, -0.62, 'chance', fontsize=5.9, color='#7b838c')
ax.set_xlim(40, 116)
ax.set_ylim(-0.8, len(reps) - 0.35)
ax.set_xticks([50, 60, 70, 80, 90, 100])
ax.set_xlabel('Accuracy (%)')
ax.set_title('(b) Input representation / objective', loc='left', fontsize=7.8)
ax.grid(axis='x', lw=0.4, color='#e3e6e8')
ax.set_axisbelow(True)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], marker='o', ls='', mfc='white', mec=DARK, ms=4),
           Line2D([], [], marker='o', ls='', color=DARK, ms=4)],
          ['train', 'validation'], loc='upper center', bbox_to_anchor=(0.45, -0.15),
          ncol=2, frameon=False, fontsize=6.0, handletextpad=0.3)
fig.savefig('paper/figures/design_space.pdf')
plt.close(fig)
print('design_space ok')
