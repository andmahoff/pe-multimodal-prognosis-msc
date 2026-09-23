"""Figure 27 - late fusion against feature-level
concatenation, training data held constant (Ch 4.8)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
d = pd.read_csv(fd.DATA + "/early3way.csv")
d.to_csv("table14_early_late.csv", index=False)

d["cell"] = (d["outcome"].map(lambda o: fd.NICE.get(o, o))
             + "\n" + d["ehr"].astype(str) + " / "
             + d["learner"].astype(str))

fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8),
                         gridspec_kw={"width_ratios":
                                      [1.25, 1.0]})

ax = axes[0]
fs.panel_tag(ax, "A")
ypos = np.arange(len(d))[::-1]
SER = [("ehr_auc", "EHR alone", "#BBBBBB", "o"),
       ("early_ehr_ecg", "early: EHR + ECG",
        fs.C["ecg"], "s"),
       ("early3", "early: all three", "#9ECAE1", "^"),
       ("late", "late (WMEAN3-CTPA)",
        fs.C["fused"], "D")]
for y0, r in zip(ypos, d.itertuples()):
    ax.plot([r.ehr_auc, r.late], [y0, y0],
            color="#E4E4E4", lw=1.4, zorder=1)
for col, lab, c, mk in SER:
    if col in d.columns:
        ax.scatter(d[col], ypos, s=36, color=c,
                   marker=mk, label=lab, zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels(d["cell"], fontsize=6.2)
ax.set_xlabel("Area Under the ROC Curve")
ax.grid(axis="x")
ax.set_title("All Eight Cells", fontsize=9.5, pad=10)

ax = axes[1]
fs.panel_tag(ax, "B")
for y0, r in zip(ypos, d.itertuples()):
    sig = (r.lo > 0) or (r.hi < 0)
    c = fs.C["fused"] if sig else "#888888"
    ax.plot([r.lo, r.hi], [y0, y0], color=c, lw=1.5)
    ax.scatter([r.diff], [y0], s=32, marker="s",
               color=c, zorder=3)
ax.axvline(0, color="#333333", lw=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels([])
ax.set_xlabel("Late Minus Early (AUC)")
ax.grid(axis="x")
ax.set_title("Late Minus Early, by Cell",
             fontsize=9.5, pad=10)

hand, labs = axes[0].get_legend_handles_labels()
fig.legend(hand, labs, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, 1.00), fontsize=7.2,
           handlelength=1.2, columnspacing=1.6)
fig.suptitle("Late Fusion Against Feature "
             "Concatenation", fontsize=10.5, y=1.10)
fig.tight_layout()
fs.save(fig, "fig27_early_late", "02_Results")

