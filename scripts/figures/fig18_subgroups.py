"""Figure 18 - performance by age band and sex, model
against sPESI-6 (Chapters 4.5 and 5.6)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
sg = pd.read_csv(fd.DATA + "/subgroup_spesi.csv")
AGEVAR = [v for v in pd.unique(sg["var"])
          if "age" in str(v).lower() or v == "band"]
AGEVAR = AGEVAR[0] if AGEVAR else pd.unique(sg["var"])[0]
apcol = "spesi_ap" if "spesi_ap" in sg.columns else None

outs = [o for o in fd.OUTCOMES if o in set(sg["outcome"])]
sub = sg[sg["var"] == AGEVAR]
levels = list(pd.unique(sub["level"]))
xs = np.arange(len(levels))
COLS = [fs.C["fused"], fs.C["ehr"], fs.C["ctpa"],
        fs.C["cxr"]]

fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.2))

ax = axes[0]
fs.panel_tag(ax, "A")
for i, o in enumerate(outs):
    s = sub[sub["outcome"] == o].set_index("level")
    s = s.reindex(levels)
    ax.errorbar(xs - 0.08, s["model_auc"],
                yerr=[s["model_auc"] - s["m_lo"],
                      s["m_hi"] - s["model_auc"]],
                fmt="o-", color=COLS[i % 4], lw=1.3,
                markersize=4, capsize=2,
                label=fd.NICE.get(o, o))
    ax.errorbar(xs + 0.08, s["spesi_auc"],
                yerr=[s["spesi_auc"] - s["s_lo"],
                      s["s_hi"] - s["spesi_auc"]],
                fmt="s--", color=COLS[i % 4], lw=0.9,
                markersize=3.4, alpha=0.50, capsize=2)
fs.chance(ax, 0.5)
ax.set_xticks(xs)
ax.set_xticklabels(levels, fontsize=7.6)
ax.set_xlabel("Age Band")
ax.set_ylabel("Area Under the ROC Curve")
ax.set_ylim(0.45, 1.0)
ax.set_title("Discrimination by Age Band",
             fontsize=9.5, pad=8)

ax = axes[1]
fs.panel_tag(ax, "B")
for i, o in enumerate(outs):
    s = sub[sub["outcome"] == o].set_index("level")
    s = s.reindex(levels)
    ax.plot(xs, s["model_ap"], "o-", color=COLS[i % 4],
            lw=1.3, markersize=4)
    if apcol is not None:
        ax.plot(xs, s[apcol], "s--", color=COLS[i % 4],
                lw=0.9, markersize=3.4, alpha=0.50)
    ax.plot(xs, s["prev"], ":", color="#999999", lw=0.9)
ax.set_xticks(xs)
ax.set_xticklabels(levels, fontsize=7.6)
ax.set_xlabel("Age Band")
ax.set_ylabel("Average Precision")
ax.set_title("Average Precision by Age Band", fontsize=9.5,
             pad=8)

hand, labs = axes[0].get_legend_handles_labels()
fig.legend(hand, labs, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, 1.00), fontsize=7.2,
           handlelength=1.6, columnspacing=1.6)
fig.suptitle("Subgroup Performance by Age Band",
             fontsize=10.5, y=1.12)
fig.tight_layout()
fs.save(fig, "fig18_subgroups", "02_Results")

