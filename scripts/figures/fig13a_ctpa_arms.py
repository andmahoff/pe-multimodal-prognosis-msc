"""Figure 13a - performance of each modality on the
three-modality presentation cohort (Chapter 4.4)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
df = pd.read_csv(fd.DATA + "/ctpa_fusion_pres.csv")
VARS = list(pd.unique(df["var"]))
main = "base_cm_dv" if "base_cm_dv" in VARS else VARS[0]
outs = [o for o in fd.OUTCOMES if o in set(df["outcome"])]

ARMS = [("ehr", "EHR", fs.C["ehr"]),
        ("ecg", "ECG", fs.C["ecg"]),
        ("ctpa", "CTPA report", fs.C["ctpa"]),
        ("wmean2", "WMEAN2", "#9ECAE1"),
        ("wmean3", "WMEAN3-CTPA", fs.C["fused"])]

fig, ax = plt.subplots(figsize=(8.0, 4.2))
xs = np.arange(len(outs))
w = 0.88 / len(ARMS)
for j, (c, lab, col) in enumerate(ARMS):
    vals = []
    for o in outs:
        r = df[(df["outcome"] == o) & (df["var"] == main)]
        vals.append(float(r[c].iloc[0]) if len(r)
                    else np.nan)
    pos = xs - 0.44 + w * (j + 0.5)
    ax.bar(pos, vals, width=w * 0.94, color=col,
           edgecolor="white", linewidth=0.7,
           hatch=fs.HATCH[j % len(fs.HATCH)],
           label=lab, zorder=3)
    for x, v in zip(pos, vals):
        if not np.isnan(v):
            ax.text(x, v + 0.004, "%.3f" % v,
                    ha="center", va="bottom",
                    fontsize=6.0, rotation=90)

fs.chance(ax, 0.5)
fs.grid(ax, axis="y")
ax.set_ylim(0.5, 0.93)
ax.set_xticks(xs)
ax.set_xticklabels([fd.NICE.get(o, o) for o in outs],
                   fontsize=7.8)
ax.set_ylabel("Area Under the ROC Curve")
ax.set_xlim(-0.52, len(outs) - 0.48)
ax.legend(ncol=5, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), fontsize=7.4,
          columnspacing=1.2, handlelength=1.3)
ax.set_title("Performance of Each Modality on the "
             "Three-Modality Cohort (n = 1,703)", fontsize=10.5,
             pad=28)

fs.save(fig, "fig13a_ctpa_arms", "02_Results")

