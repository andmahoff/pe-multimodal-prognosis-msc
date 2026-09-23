"""Fold-selected weights for the three-modality model.

Recomputes WMEAN3 (EHR + ECG + CTPA) on the presentation
cohort with the simplex grid fitted inside each training
fold, and records the weight triple per fold.

Reproduces WMEAN3 AUROC as a check: if the AUROCs match
the reported 0.8389 composite and 0.8715 death, the
weights are the ones behind those figures.

cv_first excludes patients who died first.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = "."
DATA = ROOT + "/fusion_workspace/data"
OUT = ROOT + "/fusion_workspace/tables"
KEY = ["subject_id", "hadm_id"]

OUTCOMES = ["composite_30d", "death_30d",
            "death_30d_inhosp", "cv_first"]
AT_RISK = ["cv_first"]
STEP = 0.05


def rank01(x):
    return rankdata(x) / (len(x) + 1.0)


def simplex(step=STEP):
    """All (a, b, c) on the grid summing to 1."""
    n = int(round(1.0 / step))
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((i * step, j * step, k * step))
    return out


GRID = simplex()
print("grid points:", len(GRID))

lab = pd.read_csv(DATA + "/mimic_labels_harmonised.csv")
wrows, arows = [], []

for o in OUTCOMES:
    fe = DATA + "/p_ehr_harm_%s.csv" % o
    fc = DATA + "/p_ecg_harm_%s.csv" % o
    ft = DATA + "/p_ctpa_pres_base_cm_dv_%s.csv" % o
    miss = [os.path.basename(f) for f in (fe, fc, ft)
            if not os.path.exists(f)]
    if miss:
        print("SKIP %s - missing %s" % (o, miss))
        continue

    print("=" * 50)
    print("outcome:", o)

    e = pd.read_csv(fe)
    c = pd.read_csv(fc)
    t = pd.read_csv(ft)
    tc = [x for x in t.columns if x.startswith("p_")][0]

    d = e[KEY + ["p_ehr"]].merge(
        c[KEY + ["p_ecg"]], on=KEY, how="inner")
    d = d.merge(t[KEY + [tc]].rename(
        columns={tc: "p_ctpa"}), on=KEY, how="inner")
    d = d.merge(lab[KEY + [o, "death_first"]], on=KEY)
    if o in AT_RISK:
        d = d[d["death_first"] == 0]
    d = d.dropna(subset=[o, "p_ehr", "p_ecg", "p_ctpa"])
    d = d.reset_index(drop=True)

    y = d[o].values.astype(float)
    g = d["subject_id"].values
    print("  n =", len(d), " events =", int(y.sum()))

    R = np.column_stack([
        rank01(d["p_ehr"].values),
        rank01(d["p_ecg"].values),
        rank01(d["p_ctpa"].values)])

    cv = StratifiedGroupKFold(5, shuffle=True,
                              random_state=42)
    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(
            cv.split(R, y, groups=g), 1):
        best, ba = None, -1.0
        for wv in GRID:
            s = R[tr] @ np.array(wv)
            a = roc_auc_score(y[tr], s)
            if a > ba:
                ba, best = a, wv
        oof[te] = R[te] @ np.array(best)
        wrows.append({
            "outcome": o, "fold": k,
            "w_ehr": round(best[0], 2),
            "w_ecg": round(best[1], 2),
            "w_ctpa": round(best[2], 2)})
        print("    fold %d: EHR %.2f  ECG %.2f  CTPA %.2f"
              % (k, best[0], best[1], best[2]))

    a3 = roc_auc_score(y, oof)
    print("  WMEAN3 AUROC %.4f  AP %.4f"
          % (a3, average_precision_score(y, oof)))

    mn = R.mean(axis=1)
    print("  MEAN3  AUROC %.4f" % roc_auc_score(y, mn))

    arows.append({
        "outcome": o, "n": len(y),
        "events": int(y.sum()),
        "wmean3": a3,
        "wmean3_ap": average_precision_score(y, oof),
        "mean3": roc_auc_score(y, mn),
        "ehr": roc_auc_score(y, R[:, 0]),
        "ecg": roc_auc_score(y, R[:, 1]),
        "ctpa": roc_auc_score(y, R[:, 2])})

w = pd.DataFrame(wrows)
w.to_csv(OUT + "/wmean3_weights.csv", index=False)
print("\n===== FOLD WEIGHTS =====")
print(w.to_string(index=False))

print("\n===== WEIGHT RANGES =====")
print(w.groupby("outcome").agg(
    ehr_lo=("w_ehr", "min"), ehr_hi=("w_ehr", "max"),
    ecg_lo=("w_ecg", "min"), ecg_hi=("w_ecg", "max"),
    ctpa_lo=("w_ctpa", "min"), ctpa_hi=("w_ctpa", "max"))
    .to_string(float_format="%.2f"))

a = pd.DataFrame(arows)
a.to_csv(OUT + "/wmean3_check.csv", index=False)
print("\n===== REPRODUCTION CHECK =====")
print(a.to_string(index=False, float_format="%.4f"))