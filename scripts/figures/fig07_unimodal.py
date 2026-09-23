"""Figure 7 - discrimination by model and outcome on
the primary cohort (Chapter 4.2)."""
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
LAB = {"ehr": "EHR (zero-shot)", "ecg": "ECG",
       "spesi": "sPESI-6", "mean2": "MEAN2",
       "wmean2": "WMEAN2"}

models = list(pd.unique(df["model"]))
order = [m for k in PREF for m in models
         if str(m).strip().lower() == k]
order += [m for m in models if m not in order]
outs = [o for o in fd.OUTCOMES if o in set(df["outcome"])]

fig, ax = plt.subplots(figsize=(7.6, 4.2))
xs = np.arange(len(outs))
w = 0.84 / len(order)
for j, m in enumerate(order):
    vals = []
    for o in outs:
        r = df[(df["outcome"] == o) & (df["model"] == m)]
        vals.append(float(r["auc"].iloc[0])
                    if len(r) else np.nan)
    key = str(m).strip().lower()
    pos = xs - 0.42 + w * (j + 0.5)
    ax.bar(pos, vals, width=w * 0.92,
           color=COL.get(key, "#BBBBBB"),
           edgecolor="white", linewidth=0.7,
           hatch=fs.HATCH[j % len(fs.HATCH)],
           label=LAB.get(key, str(m)), zorder=3)
    for x, v in zip(pos, vals):
        if not np.isnan(v):
            ax.text(x, v + 0.004, "%.3f" % v,
                    ha="center", va="bottom",
                    fontsize=5.6, rotation=90)

fs.chance(ax, 0.5)
fs.grid(ax, axis="y")
ax.set_ylim(0.50, 0.90)
ax.set_xticks(xs)
ax.set_xticklabels(["%s\nn = %s, %s events"
                    % (fd.NICE.get(o, o),
                       format(int(df[df["outcome"] == o]
                                  ["n"].iloc[0]), ","),
                       format(int(df[df["outcome"] == o]
                                  ["events"].iloc[0]), ","))
                    for o in outs], fontsize=7.4)
ax.set_ylabel("Area Under the ROC Curve")
ax.set_xlim(-0.55, len(outs) - 0.45)
ax.legend(ncol=5, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), fontsize=7.4,
          columnspacing=1.3, handlelength=1.3)
ax.set_title("Discrimination by Model and Outcome",
             fontsize=10.5, pad=30)

fs.save(fig, "fig07_unimodal", "02_Results")

