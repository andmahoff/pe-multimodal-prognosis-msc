"""Earlier two-panel version of Figure 19: how composite
events arrive, and how well each route is predicted
(Chapter 4.6), using the harmonised labels.
fig19a_route_timing.py and fig19b_route_auc.py draw the
two panels as separate figures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import fig_style as fs
import fig_data as fd

fs.init()
lab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")
w = pd.read_csv(fd.DATA + "/p_wmean2_composite_30d.csv")
d = lab.merge(w, on=fd.KEY)

death = (d["death_first"] == 1).values
cv = (d["cv_first"] == 1).values
none = ~(death | cv)
dd = pd.to_numeric(d["dod_days"], errors="coerce").values
cd = pd.to_numeric(d["cv_days"], errors="coerce").values
day = np.where(death, dd, np.where(cv, cd, np.nan))
print("death-first %d | cv-first %d | no event %d"
      % (death.sum(), cv.sum(), none.sum()))

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8),
                         gridspec_kw={"width_ratios":
                                      [1.35, 1.0]})

ax = axes[0]
fs.panel_tag(ax, "A")
bins = np.arange(0, 32, 2)
ax.hist([day[death][~np.isnan(day[death])],
         day[cv][~np.isnan(day[cv])]],
        bins=bins, stacked=True,
        color=[fs.C["ehr"], fs.C["ctpa"]],
        edgecolor="white", linewidth=0.6,
        label=["all-cause death first (%d)"
               % int(death.sum()),
               "CV readmission first (%d)"
               % int(cv.sum())])
ax.set_xlabel("Days From Admission to First Event")
ax.set_ylabel("Events")
ax.set_xlim(0, 30)
ax.legend(fontsize=6.8, loc="upper right")
ax.set_title("Timing by Route", fontsize=9.5, pad=10)

ax = axes[1]
fs.panel_tag(ax, "B")
p = d["p_wmean2"].values
rows = []
for nm, m, col in [("All composite events",
                    death | cv, "#777777"),
                   ("via all-cause death", death,
                    fs.C["ehr"]),
                   ("via CV readmission", cv,
                    fs.C["ctpa"])]:
    keep = m | none
    y = m[keep].astype(float)
    a = roc_auc_score(y, p[keep])
    boot = []
    rng = np.random.RandomState(42)
    ids = d["subject_id"].values[keep]
    uniq = np.unique(ids)
    idx = {g: np.where(ids == g)[0] for g in uniq}
    for _ in range(1000):
        pick = rng.choice(uniq, size=len(uniq),
                          replace=True)
        r = np.concatenate([idx[g] for g in pick])
        if y[r].min() == y[r].max():
            continue
        boot.append(roc_auc_score(y[r], p[keep][r]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append((nm, a, lo, hi, int(m.sum()), col))
    print("%-24s AUC %.4f [%.4f, %.4f]  n_ev=%d"
          % (nm, a, lo, hi, int(m.sum())))

ypos = np.arange(len(rows))[::-1]
for y0, r in zip(ypos, rows):
    ax.plot([r[2], r[3]], [y0, y0], color=r[5], lw=1.8)
    ax.scatter([r[1]], [y0], s=46, color=r[5], zorder=3)
    ax.text(r[3] + 0.006, y0, "%.3f" % r[1],
            fontsize=6.8, va="center")
ax.axvline(0.5, ls="--", lw=0.9, color=fs.C["ref"])
ax.set_yticks(ypos)
ax.set_yticklabels(["%s\n%d events" % (r[0], r[4])
                    for r in rows], fontsize=7)
ax.set_xlim(0.5, 0.95)
ax.set_xlabel("WMEAN2 AUC Against All Non-Events")
ax.grid(axis="x")
ax.set_title("Discrimination by Route", fontsize=9.5,
             pad=10)

fig.suptitle("Route and Timing of Composite Events",
             fontsize=10.5, y=1.03)
fig.tight_layout()
fs.save(fig, "fig19_route_timing", "02_Results")

