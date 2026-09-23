"""Earlier two-panel version of Figure 9: discrimination
and average precision side by side (Chapter 4.2).
fig09_average_precision.py is the single-panel version,
with discrimination shown in Figure 7."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
df = pd.read_csv(fd.DATA + "/ro4_harmonised_results.csv")

PREF = ["ehr", "ecg", "spesi", "mean2", "wmean2"]
COL = {"ehr": fs.C["ehr"], "ecg": fs.C["ecg"],
       "spesi": fs.C["spesi"], "mean2": "#9ECAE1",
       "wmean2": fs.C["fused"]}
LAB = {"ehr": "EHR", "ecg": "ECG", "spesi": "sPESI-6",
       "mean2": "MEAN2", "wmean2": "WMEAN2"}

models = list(pd.unique(df["model"]))
order = [m for k in PREF for m in models
         if str(m).strip().lower() == k]
order += [m for m in models if m not in order]
outs = [o for o in fd.OUTCOMES if o in set(df["outcome"])]

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8))
xs = np.arange(len(outs))
w = 0.80 / len(order)

for metric, ax, tag, ttl in [
        ("auc", axes[0], "A", "Discrimination"),
        ("ap", axes[1], "B", "Precision-Recall")]:
    for j, m in enumerate(order):
        vals = []
        for o in outs:
            r = df[(df["outcome"] == o)
                   & (df["model"] == m)]
            vals.append(float(r[metric].iloc[0])
                        if len(r) else np.nan)
        key = str(m).strip().lower()
        ax.bar(xs - 0.40 + w * (j + 0.5), vals,
               width=w * 0.90,
               color=COL.get(key, "#BBBBBB"),
               edgecolor="white", linewidth=0.6,
               label=LAB.get(key, str(m)))
    ax.set_xticks(xs)
    ax.set_xticklabels([fd.NICE.get(o, o) for o in outs],
                       fontsize=7, rotation=18,
                       ha="right")
    ax.set_title(ttl, fontsize=9.5, pad=8)
    fs.panel_tag(ax, tag)
    if metric == "auc":
        fs.chance(ax, 0.5)
        ax.set_ylim(0.5, 0.90)
        ax.set_ylabel("Area Under the ROC Curve")
    else:
        ax.set_ylim(0, 0.55)
        ax.set_ylabel("Average Precision")
        for o, x in zip(outs, xs):
            r = df[df["outcome"] == o].iloc[0]
            prev = float(r["events"]) / float(r["n"])
            ax.plot([x - 0.44, x + 0.44], [prev, prev],
                    color="#333333", lw=1.2, ls=":")

axes[0].legend(ncol=5, loc="lower center",
               bbox_to_anchor=(1.08, 1.16),
               fontsize=7.4, columnspacing=1.4,
               handlelength=1.3)
fig.suptitle("Discrimination and Average Precision",
             fontsize=10.5, y=1.10)
fig.tight_layout()
fs.save(fig, "fig09_auc_ap", "02_Results")

