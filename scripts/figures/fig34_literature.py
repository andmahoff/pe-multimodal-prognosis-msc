"""Figure 34 - this study against published
comparators, with marker area proportional to the
number of positive test events (Chapter 5.5).

Published values are as reported in each source
paper; where a study does not state an event count
it is marked n/a.
"""
import numpy as np
import matplotlib.pyplot as plt
import fig_style as fs

fs.init()
# label, AUC, positive test events, internal?, ours?
ROWS = [
    ("Cahan 2023, intermediate fusion", 0.960, 10,
     True, False),
    ("Cahan 2023, late average", 0.850, 10, True, False),
    ("Cahan 2023, EHR alone", 0.870, 10, True, False),
    ("Cahan 2023, sPESI", 0.770, 10, True, False),
    ("Eini 2025, pooled (17 studies)", 0.910, 0,
     True, False),
    ("INSPECT benchmark, EHR (MOTOR)", 0.923, 180,
     True, False),
    ("INSPECT benchmark, CT only", 0.794, 180,
     True, False),
    ("This study, MIMIC-trained ceiling", 0.9191, 398,
     True, True),
    ("This study, WMEAN3-CTPA", 0.8715, 157,
     False, True),
    ("This study, zero-shot EHR modality", 0.8234, 398,
     False, True),
    ("This study, sPESI-6", 0.7394, 398, False, True),
]

fig, ax = plt.subplots(figsize=(7.8, 4.8))
ypos = np.arange(len(ROWS))[::-1]
for y0, (lab, auc, ev, internal, ours) in zip(ypos,
                                              ROWS):
    s = 26 + 150 * (ev / 400.0) if ev else 26
    col = fs.C["fused"] if ours else fs.C["spesi"]
    if internal:
        ax.scatter([auc], [y0], s=s, color=col,
                   zorder=3)
    else:
        ax.scatter([auc], [y0], s=s, facecolors="none",
                   edgecolors=col, linewidths=1.6,
                   zorder=3)
    note = "n/a" if ev == 0 else str(ev)
    ax.text(auc + 0.010, y0, "%.3f   %s events"
            % (auc, note), fontsize=6.8, va="center")

ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=7.2)
ax.set_xlim(0.70, 1.02)
ax.set_ylim(-0.7, len(ROWS) - 0.3)
ax.set_xlabel("Reported AUC for 30-Day Mortality or "
              "the Nearest Equivalent Endpoint")
ax.grid(axis="x")

ax.scatter([], [], s=60, color=fs.C["spesi"],
           label="published, internally validated")
ax.scatter([], [], s=60, facecolors="none",
           edgecolors=fs.C["spesi"], linewidths=1.6,
           label="externally validated")
ax.scatter([], [], s=60, color=fs.C["fused"],
           label="this study")
ax.legend(ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), fontsize=7.0,
          handlelength=1.2, columnspacing=1.4)
ax.set_title("This Study in the Context of Published "
             "Comparators", fontsize=10.5, pad=30)

fs.save(fig, "fig34_literature", "03_Discussion")

