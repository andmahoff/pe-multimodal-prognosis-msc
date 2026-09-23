"""Figure 8 - fold-level AUC for the CXR modality, at
image and admission level (Chapter 4.2).

Each row is one outcome at one aggregation level. Open
circles are the five individual cross-validation folds;
the filled diamond is their mean. Wide scatter means an
unstable estimate.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score
import fig_style as fs
import fig_data as fd

fs.init()
LABS = ["composite_30d", "death_30d", "cv_first"]
rows = []

for lab in LABS:
    f = fd.DATA + "/cxr_harm_%s_preds.csv" % lab
    d = pd.read_csv(f)
    ycol = "_y" if "_y" in d.columns else d.columns[3]
    for fold in sorted(d["fold"].unique()):
        s = d[d["fold"] == fold]
        if s[ycol].nunique() < 2:
            continue
        img = roc_auc_score(s[ycol], s["p_cxr"])
        g = s.groupby("hadm_id").agg(
            y=(ycol, "max"), p=("p_cxr", "mean"))
        adm = roc_auc_score(g["y"], g["p"])
        rows.append((lab, int(fold), img, adm))

res = pd.DataFrame(rows, columns=["outcome", "fold",
                                  "image", "admission"])
print(res.to_string())

LEVEL = [("image", "one radiograph at a time",
          fs.C["spesi"]),
         ("admission", "averaged per admission",
          fs.C["cxr"])]

fig, ax = plt.subplots(figsize=(7.6, 3.8))
ticks = []
for i, lab in enumerate(LABS):
    s = res[res["outcome"] == lab]
    for k, (level, nice, col) in enumerate(LEVEL):
        y = i * 2.4 + (1.0 - k)
        ax.scatter(s[level], [y] * len(s), s=30,
                   facecolors="none", edgecolors=col,
                   linewidths=1.3, zorder=4)
        ax.scatter([s[level].mean()], [y], s=80,
                   color=col, marker="D", zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.plot([s[level].min(), s[level].max()],
                [y, y], color=col, lw=1.0, alpha=0.35,
                zorder=3)
        ticks.append((y, "%s\n%s"
                      % (fd.NICE.get(lab, lab), nice)))

ax.axvline(0.5, ls="--", lw=1.0, color=fs.C["ref"],
           zorder=2)
ax.text(0.5, max(t[0] for t in ticks) + 0.75, "chance",
        fontsize=6.8, color="#888888", ha="center")
ax.set_yticks([t[0] for t in ticks])
ax.set_yticklabels([t[1] for t in ticks], fontsize=6.8)
ax.set_xlabel("Area Under the ROC Curve, One Point "
              "Per Cross-Validation Fold")
ax.set_xlim(0.42, 0.82)
fs.grid(ax, axis="x")
ax.set_ylim(-0.8, max(t[0] for t in ticks) + 1.1)

leg = [Line2D([], [], marker="o", linestyle="none",
              markerfacecolor="none",
              markeredgecolor="#555555",
              markersize=6.5,
              label="one cross-validation fold"),
       Line2D([], [], marker="D", linestyle="none",
              color="#555555", markersize=7.5,
              label="mean of the five folds")]
ax.legend(handles=leg, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), fontsize=7.4,
          handlelength=1.2, columnspacing=2.0)
ax.set_title("Fold-Level Stability of the CXR "
             "Modality", fontsize=10.5, pad=30)

fs.save(fig, "fig08_fold_stability", "02_Results")

