"""Figure 11 - fusion increments with paired
subject-level bootstrap CIs (Chapter 4.3).

For the three-modality cohort, both p_wmean2 and
p_wmean3 are read from p_wmean3_ctpa_<outcome>.csv.
Writes fig11_increments.csv to the folder it is run
from, where tables_export.py reads it.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score
import fig_style as fs
import fig_data as fd

fs.init()
NBOOT = 2000
rng = np.random.RandomState(42)


def paired_ci(y, pa, pb, groups):
    obs = roc_auc_score(y, pb) - roc_auc_score(y, pa)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(NBOOT):
        pick = rng.choice(uniq, size=len(uniq),
                          replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        diffs.append(roc_auc_score(yy, pb[rows])
                     - roc_auc_score(yy, pa[rows]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, lo, hi


rows = []
for o in fd.OUTCOMES:
    d = fd.load(o)
    y = d["y"].values
    g = d["subject_id"].values
    ehr = d["p_ehr"].values
    mean2 = 0.5 * (fd.rank01(ehr)
                   + fd.rank01(d["p_ecg"].values))
    obs, lo, hi = paired_ci(y, ehr, mean2, g)
    rows.append((fd.NICE[o], "MEAN2 vs EHR alone",
                 obs, lo, hi, int(y.sum())))
    if "p_wmean2" in d.columns \
            and not d["p_wmean2"].isna().any():
        obs, lo, hi = paired_ci(y, ehr,
                                d["p_wmean2"].values, g)
        rows.append((fd.NICE[o], "WMEAN2 vs EHR alone",
                     obs, lo, hi, int(y.sum())))

lab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")
for o in ["composite_30d", "death_30d"]:
    w = pd.read_csv(fd.DATA + "/p_wmean3_ctpa_%s.csv" % o)
    w = w.merge(lab[fd.KEY + [o]], on=fd.KEY)
    w = w.dropna(subset=["p_wmean2", "p_wmean3", o])
    y = w[o].values.astype(float)
    g = w["subject_id"].values
    obs, lo, hi = paired_ci(y, w["p_wmean2"].values,
                            w["p_wmean3"].values, g)
    rows.append((fd.NICE[o], "WMEAN3-CTPA vs WMEAN2",
                 obs, lo, hi, int(y.sum())))

res = pd.DataFrame(rows, columns=["outcome", "comp",
                                  "diff", "lo", "hi",
                                  "ev"])
res.to_csv("fig11_increments.csv", index=False)
print(res.to_string())

fig, ax = plt.subplots(figsize=(7.6,
                                0.34 * len(res) + 1.5))
ypos = np.arange(len(res))[::-1]
sizes = 22.0 + 120.0 * (res["ev"] / res["ev"].max())

for y0, r, s in zip(ypos, res.itertuples(), sizes):
    sig = (r.lo > 0) or (r.hi < 0)
    c = fs.C["fused"] if sig else "#8A8A8A"
    mk = "s" if sig else "o"
    ax.plot([r.lo, r.hi], [y0, y0], color=c, lw=1.5,
            zorder=3)
    ax.scatter([r.diff], [y0], s=s, marker=mk,
               color=c, zorder=4, edgecolors="white",
               linewidths=0.7)
    ax.text(0.060, y0, "%+.4f  [%+.4f, %+.4f]"
            % (r.diff, r.lo, r.hi), fontsize=6.6,
            va="center", family="monospace", color=c)

ax.axvline(0, color="#999999", lw=1.0, ls="--",
           zorder=2)
ax.set_yticks(ypos)
ax.set_yticklabels(["%s\n%s" % (r.outcome, r.comp)
                    for r in res.itertuples()],
                   fontsize=7)
ax.set_xlim(-0.050, 0.112)
ax.set_xticks([-0.04, -0.02, 0.0, 0.02, 0.04])
ax.set_xlabel("Change in AUC Against the Comparator "
              "(95% Paired Bootstrap CI)")
fs.grid(ax, axis="x")

leg = [Line2D([], [], marker="s", linestyle="none",
              color=fs.C["fused"], markersize=7,
              label="interval excludes zero"),
       Line2D([], [], marker="o", linestyle="none",
              color="#8A8A8A", markersize=7,
              label="interval spans zero")]
ax.legend(handles=leg, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), fontsize=7.2,
          handlelength=1.2, columnspacing=1.8)
ax.set_title("Change in Discrimination From Fusion",
             fontsize=10.5, pad=28)

fs.save(fig, "fig11_forest", "02_Results")

