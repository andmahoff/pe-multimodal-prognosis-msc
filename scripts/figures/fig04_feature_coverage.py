"""Figure 4 - dual-dataset coverage of the 28
transferable features (Chapter 3.6)."""
import numpy as np
import matplotlib.pyplot as plt
import fig_style as fs

fs.init()

ROWS = [
    ("Temperature",        0.948, 0.811, "Vitals"),
    ("Heart rate",         0.927, 0.817, "Vitals"),
    ("Systolic BP",        0.926, 0.893, "Vitals"),
    ("Diastolic BP",       0.925, 0.893, "Vitals"),
    ("Respiratory rate",   0.914, 0.815, "Vitals"),
    ("Creatinine",         0.851, 0.965, "Chemistry"),
    ("Urea nitrogen",      0.850, 0.964, "Chemistry"),
    ("Sodium",             0.852, 0.963, "Chemistry"),
    ("Potassium",          0.851, 0.963, "Chemistry"),
    ("Chloride",           0.849, 0.963, "Chemistry"),
    ("Bicarbonate",        0.848, 0.962, "Chemistry"),
    ("Anion gap",          0.846, 0.962, "Chemistry"),
    ("Glucose",            0.850, 0.961, "Chemistry"),
    ("Calcium",            0.847, 0.910, "Chemistry"),
    ("Haematocrit",        0.850, 0.970, "Haematology"),
    ("Haemoglobin",        0.849, 0.970, "Haematology"),
    ("Erythrocytes",       0.847, 0.970, "Haematology"),
    ("Leukocytes",         0.849, 0.970, "Haematology"),
    ("Platelets",          0.848, 0.970, "Haematology"),
    ("MCV",                0.845, 0.970, "Haematology"),
    ("MCH",                0.845, 0.970, "Haematology"),
    ("MCHC",               0.845, 0.970, "Haematology"),
    ("RDW",                0.844, 0.970, "Haematology"),
    ("Age",                1.000, 1.000, "Demographic"),
    ("Cancer",             1.000, 1.000, "Demographic"),
    ("Heart failure",      1.000, 1.000, "Demographic"),
    ("COPD",               1.000, 1.000, "Demographic"),
    ("Atrial fibrillation", 1.000, 1.000, "Demographic"),
]
EXCL = [("Troponin", 0.501, 0.254),
        ("Oxygen saturation", 0.128, 0.810),
        ("Mean arterial pressure", 0.873, 0.380),
        ("Body weight", 0.904, 0.380)]

names = [r[0] for r in ROWS] + [e[0] for e in EXCL]
mat = np.array([[r[1], r[2]] for r in ROWS]
               + [[e[1], e[2]] for e in EXCL])
n = len(names)

fig, ax = plt.subplots(figsize=(5.8, 6.9))
ax.grid(False)
im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0,
               aspect="auto")
ax.set_xticks([0, 1])
ax.set_xticklabels(["INSPECT\n(Stanford)",
                    "MIMIC-IV\n(BIDMC)"], fontsize=8)
ax.set_yticks(np.arange(n))
ax.set_yticklabels(names, fontsize=6.8)
ax.set_ylim(n - 0.5, -0.5)
for i in range(n):
    for j in range(2):
        v = mat[i, j]
        ax.text(j, i, "%.2f" % v, ha="center",
                va="center", fontsize=6.2,
                color="white" if v > 0.62 else "#333333")
ax.axhline(len(ROWS) - 0.5, color=fs.C["neg"], lw=1.4)
for i in range(len(ROWS), n):
    ax.axhspan(i - 0.5, i + 0.5, color=fs.C["neg"],
               alpha=0.10, zorder=3)

blocks, start = [], 0
for i in range(1, len(ROWS) + 1):
    if i == len(ROWS) or ROWS[i][3] != ROWS[start][3]:
        blocks.append((ROWS[start][3], start, i - 1))
        start = i
for lab, a, b in blocks:
    ax.text(1.60, (a + b) / 2.0, lab, rotation=90,
            ha="center", va="center", fontsize=6.8,
            color="#444444")
ax.text(1.60, (len(ROWS) + n - 1) / 2.0, "Excluded",
        rotation=90, ha="center", va="center",
        fontsize=6.8, color=fs.C["neg"])
ax.set_xlim(-0.5, 1.85)

cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.16)
cb.set_label("Proportion of Cohort With the "
             "Measurement", fontsize=7.6, labelpad=6)
cb.ax.tick_params(labelsize=7)
ax.set_title("Feature Availability in Both Datasets",
             fontsize=10.5, pad=16)
fig.tight_layout()
fs.save(fig, "fig04_feature_coverage", "01_Methodology")

