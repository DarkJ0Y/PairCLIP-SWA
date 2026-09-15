"""Figures for the OpenForensics validation / held-out test analysis.
Run from the repository root:  python scripts/plot_openforensics_selection.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 400, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})
BLUE = '#2f6fb2'
ORANGE = '#d1663a'
DARK = '#22303f'
GREY = '#9aa0a6'

# ------------------------------------------------------------------
# Figure: confusion matrices, validation and test, two checkpoints
# rows: true fake [TP, FN]; true real [FP, TN]
# ------------------------------------------------------------------
panels = [
    ('Best regular / Best SWA', 'Validation', [[19451, 190], [429, 19358]], 98.43),
    ('Final SWA', 'Validation', [[18802, 839], [80, 19707]], 97.67),
    ('Best regular / Best SWA', 'Test (held out)', [[5419, 73], [1143, 4270]], 88.85),
    ('Final SWA', 'Test (held out)', [[5196, 296], [275, 5138]], 94.76),
]

fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.55))
fig.subplots_adjust(wspace=0.30, hspace=0.52)
for ax, (ck, split, cm, acc) in zip(axes.ravel(), panels):
    cm = np.array(cm, dtype=float)
    frac = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(frac, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, '%d' % cm[i, j], ha='center', va='center',
                    fontsize=8.2, fontweight='bold' if i == j else 'normal',
                    color='white' if frac[i, j] > 0.55 else DARK)
            ax.text(j, i + 0.28, '(%.1f%%)' % (100 * frac[i, j]), ha='center',
                    va='center', fontsize=6.2,
                    color='#dbe6f2' if frac[i, j] > 0.55 else '#5f6771')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Fake', 'Real'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Fake', 'Real'])
    ax.set_xlabel('Predicted', labelpad=1)
    ax.set_ylabel('True', labelpad=1)
    ax.set_title('%s\n%s  --  accuracy %.2f%%' % (ck, split, acc),
                 fontsize=7.4, pad=4)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('#c8cdd2')
    ax.tick_params(length=0)
fig.savefig('paper/figures/confusion_of.pdf')
plt.close(fig)
print('confusion_of ok')

# ------------------------------------------------------------------
# Figure: validation / test trade-off across checkpoints
# ------------------------------------------------------------------
ck = ['Best\nregular', 'Best\nSWA', 'Final\nSWA']
val = [98.43, 98.43, 97.67]
tst = [88.85, 88.85, 94.76]
rec_f = [98.67, 98.67, 94.61]      # fake recall on test
rec_r = [78.88, 78.88, 94.92]      # real recall (specificity) on test

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
fig.subplots_adjust(wspace=0.30)

ax = axes[0]
x = np.arange(3)
w = 0.34
ax.bar(x - w / 2, val, w, color=GREY, label='Validation ($n{=}39{,}428$)')
ax.bar(x + w / 2, tst, w, color=BLUE, label='Held-out test ($n{=}10{,}905$)')
for xi, v in zip(x - w / 2, val):
    ax.annotate('%.2f' % v, (xi, v), textcoords='offset points', xytext=(0, 2.5),
                ha='center', fontsize=6.4, color='#5f6771')
for xi, v in zip(x + w / 2, tst):
    ax.annotate('%.2f' % v, (xi, v), textcoords='offset points', xytext=(0, 2.5),
                ha='center', fontsize=6.4, color=DARK, fontweight='bold')
for xi, a, b in zip(x, val, tst):
    ax.text(xi, max(a, b) + 3.1, 'gap %+.2f pp' % (b - a), ha='center',
            fontsize=6.4, color=ORANGE if b < a else '#2e8b6f')
ax.set_xticks(x)
ax.set_xticklabels(ck)
ax.set_ylim(84, 106)
ax.set_yticks([85, 90, 95, 100])
ax.set_ylabel('Accuracy (\\%)')
ax.set_title('(a) Validation-to-test generalization gap', loc='left')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2,
          frameon=False, handlelength=1.1, fontsize=6.4, columnspacing=1.2,
          labelspacing=0.25)
ax.grid(axis='y', lw=0.4, color='#e3e6e8')
ax.set_axisbelow(True)

ax = axes[1]
ax.bar(x - w / 2, rec_f, w, color='#7fa8cf', label='Fake-class recall')
ax.bar(x + w / 2, rec_r, w, color=ORANGE, label='Real-class recall (specificity)')
for xs, vs, col in ((x - w / 2, rec_f, '#3d5a78'), (x + w / 2, rec_r, '#8a3f1d')):
    for xi, v in zip(xs, vs):
        ax.annotate('%.2f' % v, (xi, v), textcoords='offset points', xytext=(0, 2.5),
                    ha='center', fontsize=6.4, color=col)
ax.set_xticks(x)
ax.set_xticklabels(ck)
ax.set_ylim(70, 106)
ax.set_yticks([70, 80, 90, 100])
ax.set_ylabel('Recall on held-out test (\\%)')
ax.set_title('(b) Per-class recall on the held-out test partition', loc='left')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2,
          frameon=False, handlelength=1.1, fontsize=6.4, columnspacing=1.2,
          labelspacing=0.25)
ax.grid(axis='y', lw=0.4, color='#e3e6e8')
ax.set_axisbelow(True)
fig.savefig('paper/figures/val_test_tradeoff.pdf')
plt.close(fig)
print('val_test_tradeoff ok')

# ------------------------------------------------------------------
# Qualitative test-sample panels (downscaled from the evaluation output)
# ------------------------------------------------------------------
for src, dst in [('results/openforensics/sample_predictions/test_results_Best_Regular.png',
                  'paper/figures/qual_best_regular.png'),
                 ('results/openforensics/sample_predictions/test_results_Final_SWA.png',
                  'paper/figures/qual_final_swa.png')]:
    im = Image.open(src).convert('RGB')
    tw = 2300
    im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS).save(
        dst, optimize=True)
    print(dst, 'ok')
