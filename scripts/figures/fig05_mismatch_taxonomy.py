"""Figure 5 - taxonomy of cross-institutional
measurement mismatch (Chapter 3.6).

Drawn as a grid: dark header band, white cell
dividers, one colour per severity row with a matching
accent on the effect column.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import fig_style as fs

fs.init()
fig, ax = plt.subplots(figsize=(8.4, 4.0))
fs.blank(ax)
ax.set_xlim(0, 1)

XB = [0.055, 0.245, 0.500, 0.755, 0.985]   # 4 columns
PAD = 0.012
HDR_Y, HDR_H = 0.905, 0.075
ROW_H, GAP = 0.185, 0.016

HEADERS = ["Mismatch", "How the Sites Differ",
           "Structural Consequence", "Measured Effect"]

ROWS = [
    ("Anion gap", "Additive constant",
     "Na + K $-$ Cl $-$ HCO$_3$ at the\n"
     "target vs Na $-$ Cl $-$ HCO$_3$\n"
     "at the source; medians\n13.78 vs 9.50",
     "Absorbed entirely by the\nintercept of a "
     "standardised\nlinear model",
     "$\\Delta$AUC 0.0001 to 0.0008\nharmless",
     "#C7E5D3", "#3F8F63"),
    ("Troponin", "Different assay",
     "Troponin I at the source\nvs troponin T at the "
     "target;\ndifferent reference ranges",
     "Feature degraded to noise;\ncarried no "
     "information\neither way",
     "$\\Delta$AUC $+$0.0006\nfeature dropped",
     "#FBE4B8", "#C08A20"),
    ("Oxygen saturation", "Different method",
     "Calculated from pO$_2$ at\nthe source (12.8% "
     "coverage)\nvs pulse oximetry",
     "Feature excluded although\ntarget coverage was "
     "81%\nand it ranks 5th of 84",
     "genuine\ninformation loss",
     "#F7CDA6", "#C87A2E"),
    ("Concept identifier", "Different quantity",
     "Smoking pack-per-day read\nas systolic pressure; "
     "an\ninvalid concept as heart rate",
     "The feature measures\nsomething unrelated to\n"
     "its name",
     "first pipeline\ninvalidated",
     "#F0AFA6", "#B4483C"),
]

ax.add_patch(Rectangle((XB[0], HDR_Y), XB[4] - XB[0],
                       HDR_H, facecolor="#3C3C3C",
                       edgecolor="none"))
for i, h in enumerate(HEADERS):
    ax.text(XB[i] + PAD, HDR_Y + HDR_H / 2.0, h,
            fontsize=7.4, fontweight="bold",
            color="white", va="center", ha="left")

for r, row in enumerate(ROWS):
    y = HDR_Y - (r + 1) * (ROW_H + GAP)
    ax.add_patch(Rectangle((XB[0], y), XB[4] - XB[0],
                           ROW_H, facecolor=row[5],
                           edgecolor=row[6], lw=1.0))
    for k in range(1, 4):
        ax.plot([XB[k], XB[k]], [y + 0.012,
                                 y + ROW_H - 0.012],
                color="white", lw=1.0, zorder=3)
    yc = y + ROW_H / 2.0
    ax.text(XB[0] + PAD, yc + 0.030, row[0],
            fontsize=8.0, fontweight="bold",
            va="center", ha="left")
    ax.text(XB[0] + PAD, yc - 0.038, row[1],
            fontsize=6.8, style="italic", va="center",
            ha="left", color="#3A3A3A")
    ax.text(XB[1] + PAD, yc, row[2], fontsize=6.4,
            va="center", ha="left", linespacing=1.5)
    ax.text(XB[2] + PAD, yc, row[3], fontsize=6.4,
            va="center", ha="left", linespacing=1.5)
    ax.text(XB[3] + PAD, yc, row[4], fontsize=7.0,
            va="center", ha="left", fontweight="bold",
            linespacing=1.5, color=row[6])

fs.arrow(ax, 0.022, HDR_Y - 0.02, 0.022, 0.035,
         color=fs.C["neg"], lw=1.6)
ax.text(0.006, 0.46, "increasing severity", rotation=90,
        fontsize=7.4, ha="center", va="center",
        color=fs.C["neg"])
fs.save(fig, "fig05_mismatch_taxonomy", "01_Methodology")

