"""Earlier two-panel version of Figure 13: performance
of each modality and the CTPA increment on the
three-modality presentation cohort (Ch 4.4).
fig13a_ctpa_arms.py and fig13b_ctpa_increment.py draw
the two panels as separate figures."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
df = pd.read_csv(fd.DATA + "/ctpa_fusion_pres.csv")
print(df.to_string())

NICEV = {"base_cm": "findings + comorbidity",
         "base_cm_dv": "+ support devices",
         "base": "findings only"}
VARS = list(pd.unique(df["var"]))
outs = [o for o in fd.OUTCOMES if o in set(df["outcome"])]
main = "base_cm_dv" if "base_cm_dv" in VARS else VARS[0]

ARMS = [("ehr", "EHR", fs.C["ehr"]),
        ("ecg", "ECG", fs.C["ecg"]),
        ("ctpa", "CTPA report", fs.C["ctpa"]),
        ("wmean2", "WMEAN2", "#9ECAE1"),
        ("wmean3", "WMEAN3-CTPA", fs.C["fused"])]

fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2),
                         gridspec_kw={"width_ratios":
                                      [1.25, 1.0]})

ax = axes[0]
fs.panel_tag(ax, "A")
xs = np.arange(len(outs))
w = 0.80 / len(ARMS)
for j, (c, lab, col) in enumerate(ARMS):
    vals = []
    for o in outs:
        r = df[(df["outcome"] == o) & (df["var"] == main)]
        vals.append(float(r[c].iloc[0]) if len(r)
                    else np.nan)
    ax.bar(xs - 0.40 + w * (j + 0.5), vals,
           width=w * 0.90, color=col, edgecolor="white",
           linewidth=0.6, label=lab)
fs.chance(ax, 0.5)
ax.set_ylim(0.5, 0.92)
ax.set_xticks(xs)
ax.set_xticklabels([fd.NICE.get(o, o) for o in outs],
                   fontsize=7, rotation=18, ha="right")
ax.set_ylabel("Area Under the ROC Curve")
ax.set_title("Performance on the Three-Modality Cohort",
             fontsize=9.5, pad=8)

ax = axes[1]
fs.panel_tag(ax, "B")
rows = []
for o in outs:
    for v in VARS:
        r = df[(df["outcome"] == o) & (df["var"] == v)]
        if len(r):
            rows.append((o, v, float(r["inc"].iloc[0]),
                         float(r["lo"].iloc[0]),
                         float(r["hi"].iloc[0])))
ypos = np.arange(len(rows))[::-1]
for y0, r in zip(ypos, rows):
    sig = r[3] > 0
    col = fs.C["fused"] if sig else "#888888"
    mk = "s" if r[1] == main else "o"
    ax.plot([r[3], r[4]], [y0, y0], color=col, lw=1.4)
    ax.scatter([r[2]], [y0], s=34, marker=mk,
               color=col, zorder=3)
ax.axvline(0, color="#333333", lw=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels(["%s\n%s" % (fd.NICE.get(r[0], r[0]),
                                NICEV.get(r[1], r[1]))
                    for r in rows], fontsize=6.4)
ax.set_xlabel("WMEAN3-CTPA Minus WMEAN2 (AUC)")
ax.grid(axis="x")
ax.set_title("Increment From Adding the CTPA Modality",
             fontsize=9.5, pad=8)

axes[0].legend(ncol=5, loc="lower center",
               bbox_to_anchor=(0.95, 1.16), fontsize=7.2,
               columnspacing=1.3, handlelength=1.2)
fig.tight_layout()
fs.save(fig, "fig13_ctpa_cohort", "02_Results")

