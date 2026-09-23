"""Figure 10 - ROC curves by outcome on the primary
cohort (Chapter 4.3). sPESI-6 is drawn as the step
function it actually is: seven attainable points.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import fig_style as fs
import fig_data as fd

fs.init()
fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
axes = axes.ravel()
tags = ["A", "B", "C", "D"]

SERIES = [("p_ehr", "EHR (zero-shot)", fs.C["ehr"],
           "-"),
          ("p_ecg", "ECG", fs.C["ecg"], "--"),
          ("p_wmean2", "WMEAN2", fs.C["fused"], "-")]

for i, o in enumerate(fd.OUTCOMES):
    ax = axes[i]
    d = fd.load(o)
    y = d["y"].values
    for col, lab, c, ls in SERIES:
        if col not in d.columns:
            continue
        p = d[col].values
        if np.isnan(p).any():
            continue
        fpr, tpr, _ = roc_curve(y, p)
        a = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=c, lw=1.6, ls=ls,
                label="%s  %.4f" % (lab, a), zorder=3)
    sp = d["spesi"].values
    fpr, tpr, _ = roc_curve(y, sp)
    a = roc_auc_score(y, sp)
    ax.plot(fpr, tpr, color=fs.C["spesi"], lw=1.2,
            ls=(0, (1, 1)), drawstyle="steps-post",
            label="sPESI-6  %.4f" % a, zorder=3)
    ax.plot(fpr, tpr, "o", color=fs.C["spesi"],
            markersize=3.2, zorder=4)

    ax.plot([0, 1], [0, 1], ls="--", lw=0.8,
            color=fs.C["ref"], zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    fs.grid(ax)
    ax.set_title("%s\nn = %s, %d events"
                 % (fd.NICE[o], format(len(d), ","),
                    int(y.sum())), fontsize=8.6, pad=8)
    ax.legend(loc="lower right", fontsize=6.8)
    fs.panel_tag(ax, tags[i], x=-0.19, y=1.20)
    if i in (2, 3):
        ax.set_xlabel("1 $-$ Specificity")
    if i in (0, 2):
        ax.set_ylabel("Sensitivity")

fs.suptitle(fig, "Receiver Operating Characteristic "
                 "Curves by Outcome", y=1.005)
fig.tight_layout()
fs.save(fig, "fig10_roc_panel", "02_Results")

