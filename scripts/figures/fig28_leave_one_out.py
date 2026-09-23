"""Figure 28 - what each modality contributes, measured
by removing it from the three-modality ensemble (Ch 4.8).
Also produces Table 15.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import fig_style as fs
import fig_data as fd

fs.init()
STEP = 0.05
GRID3 = [(a, b, round(1 - a - b, 2))
         for a in np.round(np.arange(0, 1.001, STEP), 2)
         for b in np.round(np.arange(0, 1.001 - a,
                                     STEP), 2)]
GRID2 = [(w, round(1 - w, 2))
         for w in np.round(np.arange(0, 1.001, STEP), 2)]


def tune_apply(R, y, g, seed=42):
    k = len(R)
    grid = GRID3 if k == 3 else GRID2
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                              random_state=seed)
    M = np.vstack(R).T
    for tr, te in cv.split(M, y, g):
        best, bw = -1.0, grid[0]
        for w in grid:
            a = roc_auc_score(y[tr],
                              M[tr].dot(np.array(w)))
            if a > best:
                best, bw = a, w
        p[te] = M[te].dot(np.array(bw))
    return roc_auc_score(y, p)


lab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")
OUTS = ["death_30d", "composite_30d",
        "death_30d_inhosp"]
rows = []

for o in OUTS:
    ehr = pd.read_csv(fd.DATA + "/p_ehr_harm_%s.csv"
                      % fd.EHRMAP[o])
    ecg = pd.read_csv(fd.DATA + "/p_ecg_harm_%s.csv" % o)
    ct = pd.read_csv(fd.DATA
                     + "/p_ctpa_pres_base_cm_dv_%s.csv" % o)
    d = lab[fd.KEY + [o]].merge(ehr, on=fd.KEY)
    d = d.merge(ecg, on=fd.KEY).merge(ct, on=fd.KEY)
    d = d.dropna()
    y = d[o].values.astype(float)
    g = d["subject_id"].values
    R = {"EHR": fd.rank01(d["p_ehr"].values),
         "ECG": fd.rank01(d["p_ecg"].values),
         "CTPA": fd.rank01(d["p_ctpa"].values)}
    full = tune_apply([R["EHR"], R["ECG"], R["CTPA"]],
                      y, g)
    print("\n%s n=%d ev=%d  all three %.4f"
          % (o, len(d), int(y.sum()), full))
    for drop in ["EHR", "ECG", "CTPA"]:
        keep = [k for k in ["EHR", "ECG", "CTPA"]
                if k != drop]
        a = tune_apply([R[k] for k in keep], y, g)
        rows.append((o, drop, " + ".join(keep), a,
                     a - full, int(y.sum())))
        print("  drop %-5s -> %s  %.4f  (%+.4f)"
              % (drop, " + ".join(keep), a, a - full))
    rows.append((o, "none", "EHR + ECG + CTPA", full,
                 0.0, int(y.sum())))

res = pd.DataFrame(rows, columns=["outcome", "dropped",
                                  "kept", "auc",
                                  "delta", "ev"])
res.to_csv("table15_leave_one_out.csv", index=False)

fig, ax = plt.subplots(figsize=(7.2, 4.0))
drops = ["EHR", "ECG", "CTPA"]
COL = {"EHR": fs.C["ehr"], "ECG": fs.C["ecg"],
       "CTPA": fs.C["ctpa"]}
xs = np.arange(len(OUTS))
w = 0.78 / len(drops)
for j, dr in enumerate(drops):
    vals = [float(res[(res["outcome"] == o)
                      & (res["dropped"] == dr)]
                  ["delta"].iloc[0]) for o in OUTS]
    pos = xs - 0.39 + w * (j + 0.5)
    ax.bar(pos, vals, width=w * 0.90, color=COL[dr],
           edgecolor="white", linewidth=0.6,
           label="without " + dr)
    for x, v in zip(pos, vals):
        ax.text(x, v - 0.0022, "%+.4f" % v,
                ha="center", va="top", fontsize=6.0)

ax.axhline(0, color="#333333", lw=0.9)
ax.set_ylim(-0.075, 0.006)
ax.set_xticks(xs)
ax.set_xticklabels([fd.NICE.get(o, o) for o in OUTS],
                   fontsize=7.6)
ax.set_ylabel("AUC Change From Removing One Modality")
ax.grid(axis="y")
ax.legend(ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), fontsize=7.4,
          handlelength=1.3, columnspacing=1.8)
ax.set_title("Leave-One-Modality-Out on the "
             "Three-Modality Cohort (n = 1,703)",
             fontsize=10.5, pad=30)

fs.save(fig, "fig28_leave_one_out", "02_Results")

