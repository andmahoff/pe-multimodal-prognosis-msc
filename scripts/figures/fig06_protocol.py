"""Figure 6 - evaluation protocol (Chapter 3.8)."""
import matplotlib.pyplot as plt
import fig_style as fs

fs.init()
fig, ax = plt.subplots(figsize=(7.6, 4.2))
fs.blank(ax)

fs.box(ax, 0.185, 0.880, 0.630, 0.095,
       "StratifiedGroupKFold (k = 5, shuffle, seed 42)\n"
       "grouped on subject_id $-$ no patient appears "
       "in two folds", fc="#E7EEF4", ec="#3E6E8E",
       fs=7.6, weight="bold")

for i in range(5):
    x = 0.050 + i * 0.187
    held = (i == 2)
    fs.box(ax, x, 0.735, 0.160, 0.072,
           "Fold %d" % (i + 1),
           fc=fs.C["fused"] if held else "#F2F2F2",
           ec=fs.C["fused"] if held else "#9A9A9A",
           fs=7.4,
           weight="bold" if held else "normal")
    fs.arrow(ax, 0.50, 0.880, x + 0.080, 0.810,
             color="#A8A8A8")

fs.box(ax, 0.040, 0.395, 0.435, 0.250,
       "TRAINING FOLDS (4 of 5)\n\n"
       "1.  rank-normalise each modality\n"
       "2.  simplex weight grid, 0.05 steps\n"
       "3.  select weights maximising AUC\n"
       "     within these folds only",
       fc="#D6E8F5", ec=fs.C["ehr"], fs=7.2)
fs.box(ax, 0.525, 0.395, 0.435, 0.250,
       "HELD-OUT FOLD\n\n"
       "4.  apply the selected weights\n"
       "5.  record out-of-fold predictions\n"
       "     (weights are not refitted)",
       fc="#FBE6C9", ec=fs.C["fused"], fs=7.2)
fs.arrow(ax, 0.475, 0.520, 0.525, 0.520, lw=1.2)
fs.arrow(ax, 0.170, 0.735, 0.170, 0.645, lw=1.2,
         color=fs.C["ehr"])
fs.arrow(ax, 0.610, 0.735, 0.745, 0.645, lw=1.2,
         color=fs.C["fused"])

fs.box(ax, 0.130, 0.240, 0.740, 0.092,
       "Pooled out-of-fold predictions for every "
       "admission\nPlatt calibration fitted "
       "out-of-fold in the same scheme",
       fc="#FFF3E2", ec=fs.C["fused"], fs=7.6,
       weight="bold")
fs.arrow(ax, 0.50, 0.395, 0.50, 0.332, lw=1.2)

fs.box(ax, 0.130, 0.025, 0.740, 0.172,
       "INFERENCE\n"
       "paired subject-level bootstrap, 2,000 "
       "resamples (patients, not rows)\n"
       "DeLong test for correlated ROC curves | "
       "categorical NRI | decision curves\n"
       "comparators pre-specified before results "
       "were seen", fc="#E3F2EA", ec=fs.C["ctpa"],
       fs=7.2)
fs.arrow(ax, 0.50, 0.240, 0.50, 0.197, lw=1.2)

fs.save(fig, "fig06_protocol", "01_Methodology")

