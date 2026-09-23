"""Figure 13b - incremental discrimination from adding
the CTPA modality to the two-modality ensemble
(Chapter 4.4).

Each row is one outcome under one report feature set.
A point right of the dashed zero line means the CTPA
modality improved the ensemble on that outcome.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fig_style as fs
import fig_data as fd

fs.init()
df = pd.read_csv(fd.DATA + "/ctpa_fusion_pres.csv")
NICEV = {"base": "findings only",
         "base_cm": "findings + comorbidity",
         "base_cm_dv": "findings + comorbidity "
                       "+ devices"}
VARS = list(pd.unique(df["var"]))
outs = [o for o in fd.OUTCOMES if o in set(df["outcome"])]

rows = []
for o in outs:
    for v in VARS:
        r = df[(df["outcome"] == o) & (df["var"] == v)]
        if len(r):
            rows.append((o, v, float(r["inc"].iloc[0]),
                         float(r["lo"].iloc[0]),
                         float(r["hi"].iloc[0]),
                         int(r["ev"].iloc[0])))

fig, ax = plt.subplots(figsize=(7.8,
                                0.36 * len(rows) + 1.6))
ypos = np.arange(len(rows))[::-1]
for y0, r in zip(ypos, rows):
    sig = r[3] > 0
    col = fs.C["fused"] if sig else "#8A8A8A"
    mk = "s" if sig else "o"
    ax.plot([r[3], r[4]], [y0, y0], color=col, lw=1.6,
            zorder=3)
    ax.scatter([r[2]], [y0], s=46, marker=mk,
               color=col, zorder=4, edgecolors="white",
               linewidths=0.7)
    ax.text(0.038, y0, "%+.4f  [%+.4f, %+.4f]"
            % (r[2], r[3], r[4]), fontsize=6.4,
            va="center", family="monospace", color=col)

ax.axvline(0, color="#999999", lw=1.0, ls="--",
           zorder=2)
ax.set_yticks(ypos)
ax.set_yticklabels(["%s   (%d events)\n%s"
                    % (fd.NICE.get(r[0], r[0]), r[5],
                       NICEV.get(r[1], r[1]))
                    for r in rows], fontsize=6.6)
fs.grid(ax, axis="x")
ax.set_xlim(-0.012, 0.072)
ax.set_xticks([-0.01, 0.0, 0.01, 0.02, 0.03])
ax.set_xlabel("WMEAN3-CTPA Minus WMEAN2 "
              "(AUC, 95% Bootstrap CI)")
ax.set_ylim(-0.7, len(rows) - 0.3)

leg = [Line2D([], [], marker="s", linestyle="none",
              color=fs.C["fused"], markersize=7,
              label="interval excludes zero"),
       Line2D([], [], marker="o", linestyle="none",
              color="#8A8A8A", markersize=7,
              label="interval spans zero")]
ax.legend(handles=leg, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), fontsize=7.2,
          handlelength=1.2, columnspacing=1.8)
ax.set_title("Incremental Discrimination From Adding "
             "the CTPA Modality", fontsize=10.5,
             pad=28)

fs.save(fig, "fig13b_ctpa_increment", "02_Results")

