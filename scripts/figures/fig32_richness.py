"""Figure 32 - in-distribution AUC of count-based features
against the earlier 15-feature EHR modality in INSPECT,
and the depth of prior diagnosis history in MIMIC-IV
(Chapter 4.9)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
cf = pd.read_csv(fd.DATA + "/count_feature_results.csv")
hd = pd.read_csv(fd.DATA + "/mimic_history_depth.csv")

# INSPECT in-distribution AUC of the earlier
# 15-feature EHR modality: the reference value
# printed by 116b_count_models.py.
BASE = 0.6976
INSPECT_MEDIAN = 377.0
best = cf.sort_values("auc", ascending=False).iloc[0]
bars = [("15-feature EHR\nmodality", BASE,
         fs.C["ehr"]),
        ("%d count features"
         % int(best["nfeat"]), float(best["auc"]),
         fs.C["fused"])]

fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))

ax = axes[0]
fs.panel_tag(ax, "A")
xs = np.arange(len(bars))
ax.bar(xs, [b[1] for b in bars], width=0.52,
       color=[b[2] for b in bars], edgecolor="white")
for x, b in zip(xs, bars):
    ax.text(x, b[1] + 0.006, "%.4f" % b[1],
            ha="center", fontsize=7.4)
ax.annotate("", xy=(1, bars[1][1] + 0.030),
            xytext=(0, bars[0][1] + 0.030),
            arrowprops=dict(arrowstyle="-|>", lw=1.1,
                            color="#333333"))
ax.text(0.5, bars[1][1] + 0.048,
        "$+$%.4f" % (bars[1][1] - bars[0][1]),
        ha="center", fontsize=8.0, fontweight="bold")
fs.chance(ax, 0.5)
ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in bars], fontsize=7.4)
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("INSPECT In-Distribution AUC")
ax.set_title("In-Distribution AUC by Feature Set", fontsize=9.5,
             pad=10)
ax.grid(axis="y")

ax = axes[1]
fs.panel_tag(ax, "B")
n = hd["n_codes"].values
ax.hist(np.clip(n, 0, 400), bins=40, color=fs.C["ehr"],
        edgecolor="white", linewidth=0.5)
med = float(np.median(n))
zero = float(np.mean(n == 0))
ax.axvline(med, color="#333333", ls=":", lw=1.2)
ax.axvline(INSPECT_MEDIAN, color=fs.C["fused"],
           ls="--", lw=1.4)
ax.annotate("MIMIC-IV median\n%d codes" % int(med),
            xy=(med, ax.get_ylim()[1] * 0.72),
            xytext=(70, ax.get_ylim()[1] * 0.72),
            fontsize=7.0, va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.9,
                            color="#333333"))
ax.text(INSPECT_MEDIAN - 12, ax.get_ylim()[1] * 0.42,
        "INSPECT median\n%d codes" % int(INSPECT_MEDIAN),
        fontsize=7.0, ha="right", color=fs.C["fused"])
ax.set_xlabel("Prior Diagnosis Codes Before Admission")
ax.set_ylabel("Admissions")
ax.set_title("Prior Diagnosis History in MIMIC-IV", fontsize=9.5,
             pad=10)
ax.text(0.97, 0.96, "%.0f%% have no prior history"
        % (100 * zero), transform=ax.transAxes,
        ha="right", va="top", fontsize=7.4,
        fontweight="bold")
print("MIMIC median %.0f | zero-history %.3f"
      % (med, zero))

fig.suptitle("Longitudinal Depth Is Available at One "
             "Site Only", fontsize=10.5, y=1.03)
fig.tight_layout()
fs.save(fig, "fig32_richness", "02_Results")

