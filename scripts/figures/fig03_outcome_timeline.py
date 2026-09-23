"""Figure 3 - outcome definition and competing-risk
partition (Chapter 3.5)."""
import matplotlib.pyplot as plt
import fig_style as fs

fs.init()
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7),
                         gridspec_kw={"height_ratios":
                                      [1.05, 1.0]})

ax = axes[0]
fs.panel_tag(ax, "A")
ax.set_xlim(-2, 33)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xticks([0, 7, 14, 21, 30])
ax.set_xlabel("Days From Index Admission")
ax.grid(False)
ax.spines["left"].set_visible(False)

ax.axvspan(0, 8, ymin=0.62, ymax=0.90,
           color=fs.C["tgt"], lw=0)
ax.text(4, 0.76, "index admission\n(median LOS 8 d)",
        ha="center", va="center", fontsize=7.2)
ax.axvline(0, color="#444444", lw=1.0)
ax.axvline(30, color=fs.C["neg"], lw=1.2, ls="--")
ax.text(29.4, 0.94, "30-day boundary", ha="right",
        va="top", fontsize=7.2, color=fs.C["neg"])

ax.annotate("all-cause death (dod)\n"
            "n = 398, of which 242 in hospital",
            xy=(5.5, 0.44), xytext=(9.0, 0.40),
            fontsize=7.2, va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.9,
                            color=fs.C["ehr"]))
ax.annotate("cardiovascular readmission\n"
            "n = 171 first events",
            xy=(19.5, 0.14), xytext=(22.5, 0.14),
            fontsize=7.2, va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.9,
                            color=fs.C["ctpa"]))
ax.set_title("Observation Window and Event Routes",
             fontsize=9.5)

ax = axes[1]
fs.panel_tag(ax, "B")
ax.set_xlim(0, 3507)
ax.set_ylim(-0.15, 2.30)
ax.set_yticks([])
ax.grid(False)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Admissions in the Primary Cohort "
              "(n = 3,507)")

segs = [(0, 376, fs.C["ehr"]),
        (376, 171, fs.C["ctpa"]),
        (547, 2960, "#DDDDDD")]
for x0, w, col in segs:
    ax.barh(1.32, w, left=x0, height=0.36, color=col,
            edgecolor="white", linewidth=0.8)
ax.text(547 + 2960 / 2.0, 1.32,
        "No event within 30 days   2,960", ha="center",
        va="center", fontsize=7.2, color="#333333")
ax.annotate("Death first\n376", xy=(188, 1.50),
            xytext=(120, 1.80), fontsize=7.0,
            ha="center", color=fs.C["ehr"],
            arrowprops=dict(arrowstyle="-", lw=0.7,
                            color=fs.C["ehr"]))
ax.annotate("CV readmission first\n171", xy=(462, 1.50),
            xytext=(760, 1.80), fontsize=7.0,
            ha="center", color=fs.C["ctpa"],
            arrowprops=dict(arrowstyle="-", lw=0.7,
                            color=fs.C["ctpa"]))
ax.set_title("Competing-Risk Partition of the "
             "Composite Endpoint", fontsize=9.5, pad=14)

ax.barh(0.42, 3131, left=376, height=0.30,
        color=fs.C["ctpa"], alpha=0.30,
        edgecolor=fs.C["ctpa"], linewidth=0.9)
ax.text(376 + 3131 / 2.0, 0.42,
        "Cardiovascular-readmission analysis set   "
        "n = 3,131   (171 events, 5.5%)",
        ha="center", va="center", fontsize=7.2)
ax.annotate("", xy=(376, 0.60), xytext=(376, 1.12),
            arrowprops=dict(arrowstyle="-|>", lw=0.9,
                            color="#666666"))
ax.text(400, 0.86, "death-first admissions removed: "
        "they cannot be readmitted", fontsize=6.8,
        va="center", color="#444444")

fig.tight_layout()
fs.save(fig, "fig03_outcome_timeline", "01_Methodology")

