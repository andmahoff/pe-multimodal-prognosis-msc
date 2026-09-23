"""Figure 33 - converting the gap mechanism into a
usable rule (Chapter 5.3, cross-referenced from 6.3).

A decision aid rather than a data plot. Each measured
configuration gets its own row, so the observed points
can be read against the bands without collision.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
res = pd.read_csv(fd.DATA + "/ro4_harmonised_results.csv")
inc = pd.read_csv("fig11_increments.csv")

# EHR AUC, ECG AUC and equal-weight fusion effect for
# the earlier 15-feature EHR modality, from the
# two-way cohort results of script 84.
PRE = [("Composite (30-day)", 0.7714, 0.7207, 0.0148),
       ("30-day death", 0.7890, 0.7229, 0.0205),
       ("In-hospital death", 0.8149, 0.7063, -0.0026),
       ("CV readmission", 0.7965, 0.6849, -0.0199)]

pts = []
for lab, e, c, eff in PRE:
    pts.append((e - c, eff, lab, "15-feature"))
for o in fd.OUTCOMES:
    r = res[res["outcome"] == o]
    e = float(r[r["model"] == "EHR"]["auc"].iloc[0])
    c = float(r[r["model"] == "ECG"]["auc"].iloc[0])
    row = inc[(inc["outcome"] == fd.NICE[o])
              & (inc["comp"] == "MEAN2 vs EHR alone")]
    if len(row):
        pts.append((e - c, float(row["diff"].iloc[0]),
                    fd.NICE[o], "28-feature"))

pts = sorted(pts, key=lambda p: p[0])
for p in pts:
    print("%-20s %-11s gap %.4f  effect %+.4f"
          % (p[2], p[3], p[0], p[1]))
gmin = min(p[0] for p in pts)

fig, ax = plt.subplots(figsize=(7.8, 4.4))

BANDS = [(0.00, 0.05, "#EDEDED",
          "Extrapolated", "no observations here"),
         (0.05, 0.09, "#CDE7D4",
          "Fusion Pays", "equal weighting is enough"),
         (0.09, 0.16, "#F3CFC6",
          "Weight, or Do Not Fuse",
          "equal weighting hurts")]
for lo, hi, col, title, sub in BANDS:
    ax.axvspan(lo, hi, color=col, lw=0, zorder=0)
    mid = (lo + min(hi, 0.152)) / 2.0
    ax.text(mid, len(pts) - 0.20, title, ha="center",
            fontsize=8.4, fontweight="bold")
    ax.text(mid, len(pts) - 0.72, sub, ha="center",
            fontsize=6.8, color="#333333")
ax.axvspan(0.0, gmin, facecolor="none", hatch="////",
           edgecolor="#B8B8B8", lw=0.0, zorder=1)

ypos = np.arange(len(pts))[::-1]
for y0, (gap, eff, lab, run) in zip(ypos, pts):
    col = fs.C["fused"] if eff > 0 else "#4C72A8"
    mk = "o" if run == "15-feature" else "s"
    ax.scatter([gap], [y0], s=58, color=col, marker=mk,
               zorder=4, edgecolors="white",
               linewidths=0.8)
    ax.text(gap + 0.004, y0, "%+.4f" % eff,
            fontsize=6.8, va="center")

ax.set_yticks(ypos)
ax.set_yticklabels(["%s\n%s EHR modality" % (p[2], p[3])
                    for p in pts], fontsize=6.6)
ax.set_xlim(0, 0.16)
ax.set_ylim(-0.7, len(pts) + 0.4)
ax.set_xlabel("Performance Gap Between the Strongest "
              "and Weakest Modality (AUC)")
ax.grid(axis="x", color="#FFFFFF", lw=0.8)

ax.scatter([], [], s=58, color=fs.C["fused"],
           marker="o", label="equal-weight fusion helped")
ax.scatter([], [], s=58, color="#4C72A8", marker="o",
           label="equal-weight fusion hurt")
ax.scatter([], [], s=58, facecolors="#888888",
           marker="s", label="28-feature run (squares)")
ax.legend(ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), fontsize=7.0,
          handlelength=1.2, columnspacing=1.4)
ax.set_title("Fusion Effect by Performance Gap Between Modalities",
             fontsize=10.5, pad=30)

fs.save(fig, "fig33_when_to_fuse", "03_Discussion")

