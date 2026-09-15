# Revision notes — evidence-audit revision (v3)

Previous version: [`paper/archive/v2-before-audit/`](../paper/archive/v2-before-audit/) (`main.tex`, `main.pdf`, `references.bib`).

## Corrections of fact (from [`evidence_audit.pdf`](evidence_audit.pdf) and one additional check)

| Was (v2) | Now (v3) | Source |
|---|---|---|
| 97.04% obtained "without a single gradient step on DF-Wild-Cup"; cross-corpus result | In-domain: Stage-1 corpus mined from the DF-Wild-Cup **training** partition (42,690 + 42,690 = 85,380 images). Like-for-like with CAE-Net. | Audit, Part 1 |
| Training corpus from HF v3; warm-started from an OpenForensics checkpoint | Starts from OpenAI CLIP weights + random head (45.80% before adaptation). HF v3 removed from the paper entirely. | Audit, Parts 1 and 5 |
| Single Tesla P100 | 2x T4 with `nn.DataParallel` for the reported run; some earlier runs on P100 | Audit, Part 1 |
| Unfreezing reached cap of 14 blocks; 41.98% frozen | Reached 11 blocks (epochs 18–19); 54.42% frozen. 41.98% kept as the design guarantee. | Audit, Part 1 |
| SWA from first epoch, constant LR 1e-5 | SWA starts at epoch 12 (average = epochs 13–19); SWALR cosine anneal to 1e-5 over 10 epochs | Audit trajectory table; notebook code |
| AU-ROC / EER / F1 "not available" | AU-ROC 98.64, EER 4.07, tau* = 0.6797 (97.30%), fake-positive P/R/F1 98.98 / 95.54 / 97.23 | Audit Part 2 (`clip-vit-l-inference-jan-13`) |
| "fluctuates by more than a percentage point" | 5.34-point range; full epoch 12–19 table and figure | Audit Part 3 |
| Stage 1 embeds images with the semantic CLIP descriptor | Mining code applied `ToTensor()` then `CLIPProcessor` (`do_rescale=True`): inputs rescaled by 1/255 **twice**. Verified locally with transformers 4.46.3: all inputs fall in [-1.792, -1.466]. Disclosed in Sec. IV-B; Stage-1 claims narrowed. | New check (not in audit — audit called it harmless) |

The fake-positive P/R/F1 at tau* were reconstructed from the logged real-positive values: with 1,548 real and 1,524 fake images, the only integer confusion matrix consistent with the logged accuracy, precision, recall and F1 is 1,533 real and 1,456 fake correct.

## Added

- Preprocessing pipeline with rationale and evidence for each step (Table I, Sec. III-B).
- Training-dynamics section, table and figure (Sec. IV-E).
- Complete head-to-head table incl. threshold-free metrics, latency, data usage (Table VIII).
- Ablation table: input representation, balancing, augmentation, objective, same-run controls (Table XI).
- 22-configuration design-space study (Table XII, Fig. 8).
- DBSCAN silhouette / Davies–Bouldin paragraph.
- Related work on model-selection bias; new references (Cawley & Talbot, Varma & Simon, Recht et al., Gulrajani & Lopez-Paz, Dong et al., Face X-ray, ...).
- Abstract and introduction reframed around checkpoint selection, per the audit's recommendation.
- Title kept in the v2 form; only "Semantic Pair Mining" became "Similarity-Guided Balancing",
  because the paper no longer claims that Stage 1 matches images by semantic content.

## Removed

- HF v3 results (no notebook uses it; n = 26).
- SP-Cup bar chart (duplicated Table VII), probe-metrics table and embedding-geometry figure (duplicated Table X), OpenForensics convergence table (moved to text).

## Must be verified before submission (needs Kaggle / co-author accounts)

1. **Provenance of `results/openforensics/` and `results/dfwildcup/`.** The audit found no notebook that produced the OpenForensics val/test confusion matrices, the qualitative panels, or the t-SNE analyses. The paper's selection argument (Sec. IV-H) depends on the OpenForensics results. Locate and archive the producing notebooks.
2. **OpenForensics run recipe.** The paper states it uses "the same backbone, schedule and averaging procedure" as the DF-Wild-Cup run. Confirm, including whether Stage-1 subsampling was applied and the SWA start epoch.
3. **Stage-1 double rescaling.** Confirm by searching the `nakibmunabbi/feature-extraction` log for: `It looks like you are trying to rescale already rescaled images`.
4. **DBSCAN features.** Confirm whether `dbscan-on-openclip` used pre-trained or fine-tuned OpenCLIP features.
5. **Face-crop counterpart.** The "about 34 points below full frame" figure comes from the audit; confirm which full-frame run it was compared against.
6. **The 95.51% entry** (`clip-vit-l-inference`): confirm whether it is the checkpoint's own accuracy or the XGBoost result. The paper currently calls it an earlier checkpoint, re-evaluated.
7. **Early stopping.** Whether patience-8 early stopping triggered at epoch 19 is not stated in the paper.

## Recommended runs (in priority order)

1. Stage-1 control: re-run mining with PIL inputs (or `do_rescale=False`), then train a random 1:1 subsample under the identical schedule. Without this, pair mining cannot be claimed as a contribution.
2. Two more seeds of the main run.
3. Threshold sweep on OpenForensics val and test.
4. FGSM / PGD evaluation.
5. Class-symmetric corruption augmentation (tests the "degradation implies fake" shortcut from Sec. IV-H).
