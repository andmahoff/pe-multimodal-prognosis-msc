"""Figure 19b - discrimination for each route of
ascertainment against the same non-event group
(Chapter 4.6)."""
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
p = d["p_wmean2"].values
ids = d["subject_id"].values

rows = []
for nm, m, col in [("All composite\nevents",
                    death | cv, "#7F7F7F"),
                   ("via all-cause\ndeath", death,
                    fs.C["ehr"]),
                   ("via CV\nreadmission", cv,
                    fs.C["ctpa"])]:
    keep = m | none
    y = m[keep].astype(float)
    a = roc_auc_score(y, p[keep])
    rng = np.random.RandomState(42)
    sub = ids[keep]
    uniq = np.unique(sub)
    idx = {g: np.where(sub == g)[0] for g in uniq}
    boot = []
    for _ in range(1000):
        pick = rng.choice(uniq, size=len(uniq),
                          replace=True)
        r = np.concatenate([idx[g] for g in pick])
        if y[r].min() == y[r].max():
            continue
        boot.append(roc_auc_score(y[r], p[keep][r]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append((nm, a, lo, hi, int(m.sum()), col))
    print("%-22s AUC %.4f [%.4f, %.4f] ev=%d"
          % (nm.replace("\n", " "), a, lo, hi,
             int(m.sum())))

fig, ax = plt.subplots(figsize=(7.2, 3.0))
ypos = np.arange(len(rows))[::-1]
for y0, r in zip(ypos, rows):
    ax.plot([r[2], r[3]], [y0, y0], color=r[5], lw=2.0,
            zorder=3)
    ax.scatter([r[1]], [y0], s=60, color=r[5], zorder=4,
               edgecolors="white", linewidths=0.8)
    ax.text(r[3] + 0.004, y0, "%.3f [%.3f, %.3f]"
            % (r[1], r[2], r[3]), fontsize=6.6,
            va="center", color=r[5])
ax.axvline(0.5, ls="--", lw=0.9, color=fs.C["ref"],
           zorder=1)
ax.set_yticks(ypos)
ax.set_yticklabels(["%s\n%d events" % (r[0], r[4])
                    for r in rows], fontsize=7.0)
ax.set_xlim(0.5, 0.98)
ax.set_ylim(-0.6, len(rows) - 0.4)
fs.grid(ax, axis="x")
ax.set_xlabel("WMEAN2 AUC Against All Non-Events "
              "(95% Bootstrap CI)")
ax.set_title("Discrimination by Route of "
             "Ascertainment", fontsize=10.5, pad=12)

fs.save(fig, "fig19b_route_auc", "02_Results")

