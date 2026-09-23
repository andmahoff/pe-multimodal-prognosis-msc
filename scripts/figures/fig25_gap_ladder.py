"""Figure 25 - fusion effect against the gap between the
EHR and ECG modalities.

Panel A (observational): fusion effect against the
inter-modality gap, across both pipeline versions.
Panel B (near-controlled): the supervision ladder, in
which the ECG and CTPA predictions are identical across
all three steps and only the EHR modality changes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
res = pd.read_csv(fd.DATA + "/ro4_harmonised_results.csv")
inc = pd.read_csv("fig11_increments.csv")
lad = pd.read_csv(fd.DATA + "/supervised_fusion.csv")

# EHR AUC, ECG AUC and equal-weight fusion effect for
# the earlier 15-feature EHR modality, from the
# two-way cohort results of script 84.
PRE = [("Composite (30-day)", 0.7714, 0.7207, 0.0148),
       ("30-day death", 0.7890, 0.7229, 0.0205),
       ("In-hospital death", 0.8149, 0.7063, -0.0026),
       ("CV readmission", 0.7965, 0.6849, -0.0199)]

pts = []
for lab, e, c, eff in PRE:
    pts.append((e - c, eff, lab, "pre"))
for o in fd.OUTCOMES:
    r = res[res["outcome"] == o]
    e = float(r[r["model"] == "EHR"]["auc"].iloc[0])
    c = float(r[r["model"] == "ECG"]["auc"].iloc[0])
    row = inc[(inc["outcome"] == fd.NICE[o])
              & (inc["comp"] == "MEAN2 vs EHR alone")]
    if len(row):
        pts.append((e - c, float(row["diff"].iloc[0]),
                    fd.NICE[o], "post"))
for p in pts:
    print("%-20s %-5s gap %.4f  effect %+.4f"
          % (p[2], p[3], p[0], p[1]))

LADX = {"zero-shot": "Zero-Shot",
        "MIMIC-28": "MIMIC-28",
        "MIMIC-all": "MIMIC-All"}

fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2))

ax = axes[0]
fs.panel_tag(ax, "A")
for gap, eff, lab, run in pts:
    if run == "pre":
        ax.scatter([gap], [eff], s=52, facecolors="none",
                   edgecolors=fs.C["ehr"], linewidths=1.5,
                   zorder=3)
    else:
        ax.scatter([gap], [eff], s=52,
                   color=fs.C["fused"], zorder=3)
xs = np.array([p[0] for p in pts])
ys = np.array([p[1] for p in pts])
b = np.polyfit(xs, ys, 1)
gx = np.linspace(xs.min() - 0.005, xs.max() + 0.005, 20)
ax.plot(gx, np.polyval(b, gx), color="#999999", lw=1.0,
        ls="--", zorder=0)
r = float(np.corrcoef(xs, ys)[0, 1])
ax.axhline(0, color="#333333", lw=0.9)
ax.set_xlabel("Inter-Modality Gap (EHR AUC $-$ ECG AUC)")
ax.set_ylabel("Equal-Weight Fusion Effect")
ax.set_title("Observational (r = %.2f, 8 Points)" % r,
             fontsize=9.5, pad=8)
ax.grid(True, color="#EEEEEE")
ax.scatter([], [], s=52, facecolors="none",
           edgecolors=fs.C["ehr"], linewidths=1.5,
           label="15-feature EHR modality")
ax.scatter([], [], s=52, color=fs.C["fused"],
           label="28-feature EHR modality")
ax.legend(fontsize=7.0, loc="lower left")

ax = axes[1]
fs.panel_tag(ax, "B")
order = list(pd.unique(lad["ehr"]))
xpos = {k: i for i, k in enumerate(order)}
cols = {"composite_30d": fs.C["ctpa"],
        "death_30d": fs.C["ehr"]}
for o in pd.unique(lad["outcome"]):
    s = lad[lad["outcome"] == o]
    x = [xpos[k] for k in s["ehr"]]
    ax.errorbar(x, s["gain"],
                yerr=[s["gain"] - s["lo"],
                      s["hi"] - s["gain"]],
                fmt="o-", capsize=3.0, lw=1.5,
                markersize=5,
                color=cols.get(o, "#777777"),
                label=fd.NICE.get(o, o))
ax.axhline(0, color="#333333", lw=0.9)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([LADX.get(str(k), str(k))
                    for k in order], fontsize=7.6)
ax.set_xlim(-0.35, len(order) - 0.45)
ax.set_xlabel("EHR Modality, Increasingly Unconstrained")
ax.set_ylabel("WMEAN3-CTPA Gain Over the EHR Modality")
ax.set_title("Controlled: Only the EHR Modality Changes",
             fontsize=9.5, pad=8)
ax.legend(fontsize=7.0, loc="upper right")
ax.grid(axis="y")

fig.suptitle("The Value of Fusion Falls as the Primary "
             "Modality Strengthens", fontsize=10.5, y=1.03)
fig.tight_layout()
fs.save(fig, "fig25_gap_ladder", "02_Results")

