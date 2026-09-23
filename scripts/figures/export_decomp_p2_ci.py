"""Subject-level bootstrap intervals for the Phase 2
route-stratified AUROCs, and for their difference.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

ROOT = "."
DATA = ROOT + "/fusion_workspace/data"
OUT = ROOT + "/fusion_workspace/tables"
KEY = ["subject_id", "hadm_id"]
O = "composite_30d"
NB = 2000
SEED = 42


def rank01(x):
    return rankdata(x) / (len(x) + 1.0)


lab = pd.read_csv(DATA + "/mimic_labels_harmonised.csv")
e = pd.read_csv(DATA + "/p_ehr_harm_%s.csv" % O)
c = pd.read_csv(DATA + "/p_ecg_harm_%s.csv" % O)

d = e[KEY + ["p_ehr"]].merge(
    c[KEY + ["p_ecg"]], on=KEY, how="inner")
d = d.merge(lab, on=KEY, how="inner")
d = d.dropna(subset=[O, "p_ehr", "p_ecg"])
d = d.reset_index(drop=True)

y = d[O].values.astype(float)
g = d["subject_id"].values
d["r_ehr"] = rank01(d["p_ehr"].values)
d["r_ecg"] = rank01(d["p_ecg"].values)
R = d[["r_ehr", "r_ecg"]].values
grid = [(w, 1.0 - w)
        for w in np.arange(0, 1.05, 0.05)]

cv = StratifiedGroupKFold(5, shuffle=True,
                          random_state=42)
sc = np.zeros(len(y))
for tr, te in cv.split(R, y, groups=g):
    best, ba = None, -1.0
    for wv in grid:
        a_ = roc_auc_score(y[tr], R[tr] @ np.array(wv))
        if a_ > ba:
            ba, best = a_, wv
    sc[te] = R[te] @ np.array(best)
d["score"] = sc

d["route"] = np.where(
    d["death_first"] == 1, "Death",
    np.where(d["cv_first"] == 1, "CV", "None"))

is_d = ((d[O] == 0) | (d["route"] == "Death")).values
is_c = ((d[O] == 0) | (d["route"] == "CV")).values

a_d = roc_auc_score(y[is_d], sc[is_d])
a_c = roc_auc_score(y[is_c], sc[is_c])
print("Death route AUROC  %.4f" % a_d)
print("CV route AUROC     %.4f" % a_c)
print("difference         %.4f" % (a_d - a_c))

rng = np.random.RandomState(SEED)
uq = np.unique(g)
idx = {u: np.where(g == u)[0] for u in uq}
bd, bc, bdiff = [], [], []
for _ in range(NB):
    pick = rng.choice(uq, len(uq), replace=True)
    ii = np.concatenate([idx[u] for u in pick])
    yy, ss = y[ii], sc[ii]
    rr = d["route"].values[ii]
    md = (yy == 0) | (rr == "Death")
    mc = (yy == 0) | (rr == "CV")
    try:
        if len(np.unique(yy[md])) < 2:
            continue
        if len(np.unique(yy[mc])) < 2:
            continue
        x1 = roc_auc_score(yy[md], ss[md])
        x2 = roc_auc_score(yy[mc], ss[mc])
    except ValueError:
        continue
    bd.append(x1)
    bc.append(x2)
    bdiff.append(x1 - x2)


def ci(v):
    return np.percentile(v, [2.5, 97.5])


print("\nresamples used:", len(bdiff))
lo, hi = ci(bd)
print("Death route  %.4f (%.4f, %.4f)" % (a_d, lo, hi))
lo, hi = ci(bc)
print("CV route     %.4f (%.4f, %.4f)" % (a_c, lo, hi))
lo, hi = ci(bdiff)
print("difference   %.4f (%.4f, %.4f)"
      % (a_d - a_c, lo, hi))
print("excludes zero:", bool(lo > 0 or hi < 0))

pd.DataFrame([
    {"stratum": "Death", "auc": a_d,
     "lo": ci(bd)[0], "hi": ci(bd)[1]},
    {"stratum": "CV readmission", "auc": a_c,
     "lo": ci(bc)[0], "hi": ci(bc)[1]},
    {"stratum": "Difference", "auc": a_d - a_c,
     "lo": ci(bdiff)[0], "hi": ci(bdiff)[1]},
]).to_csv(OUT + "/decomp_p2_ci.csv", index=False)
print("\nwrote decomp_p2_ci.csv")