"""Three-modality weighted fusion (EHR, ECG, CXR) using
the CXR predictions aggregated on dicom_id and hadm_id
(p_cxr_harm_*_fixed.csv, written by fix_cxr_agg.py).

Writes table25_cxr_fixed.csv and table26_cxr_fixed.csv,
including bootstrap intervals for the CXR modality alone.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

ROOT = "."
DATA = ROOT + "/fusion_workspace/data"
OUT = ROOT + "/fusion_workspace/tables"
KEY = ["subject_id", "hadm_id"]
NB = 2000
STEP = 0.05

OUTS = [("composite_30d", "Composite (30-day)"),
        ("death_30d", "30-day death"),
        ("cv_first", "CV readmission")]
AT_RISK = ["cv_first"]


def rank01(x):
    return rankdata(x) / (len(x) + 1.0)


def simplex(step=STEP):
    n = int(round(1.0 / step))
    return [(i * step, j * step, (n - i - j) * step)
            for i in range(n + 1)
            for j in range(n + 1 - i)]


GRID3 = simplex()
GRID2 = [(w, 1.0 - w)
         for w in np.arange(0, 1.0 + STEP, STEP)]
lab = pd.read_csv(DATA + "/mimic_labels_harmonised.csv")
rows, cmp_rows = [], []

for o, nm in OUTS:
    e = pd.read_csv(DATA + "/p_ehr_harm_%s.csv" % o)
    c = pd.read_csv(DATA + "/p_ecg_harm_%s.csv" % o)
    x = pd.read_csv(
        DATA + "/p_cxr_harm_%s_fixed.csv" % o)

    d = e[KEY + ["p_ehr"]].merge(
        c[KEY + ["p_ecg"]], on=KEY, how="inner")
    d = d.merge(x[KEY + ["p_cxr"]], on=KEY,
                how="inner")
    d = d.merge(lab[KEY + [o, "death_first"]], on=KEY)
    if o in AT_RISK:
        d = d[d["death_first"] == 0]
    d = d.dropna(subset=[o, "p_ehr", "p_ecg", "p_cxr"])
    d = d.reset_index(drop=True)

    y = d[o].values.astype(float)
    g = d["subject_id"].values
    for a in ["ehr", "ecg", "cxr"]:
        d["r_" + a] = rank01(d["p_" + a].values)
    R3 = d[["r_ehr", "r_ecg", "r_cxr"]].values
    R2 = d[["r_ehr", "r_ecg"]].values

    print("=" * 56)
    print(nm, " n =", len(d),
          " events =", int(y.sum()))

    cv = StratifiedGroupKFold(5, shuffle=True,
                              random_state=42)
    s3 = np.zeros(len(y))
    s2 = np.zeros(len(y))
    w3 = []
    for tr, te in cv.split(R3, y, groups=g):
        b, ba = None, -1.0
        for wv in GRID3:
            a_ = roc_auc_score(y[tr],
                               R3[tr] @ np.array(wv))
            if a_ > ba:
                ba, b = a_, wv
        s3[te] = R3[te] @ np.array(b)
        w3.append(b)
        b2, ba2 = None, -1.0
        for wv in GRID2:
            a_ = roc_auc_score(y[tr],
                               R2[tr] @ np.array(wv))
            if a_ > ba2:
                ba2, b2 = a_, wv
        s2[te] = R2[te] @ np.array(b2)

    mean3 = R3.mean(axis=1)
    res = {
        "EHR only": d["r_ehr"].values,
        "ECG only": d["r_ecg"].values,
        "CXR only": d["r_cxr"].values,
        "EHR + ECG (WMEAN2)": s2,
        "Equal-weight average (MEAN3)": mean3,
        "Weighted average (WMEAN3)": s3,
    }
    for k, v in res.items():
        a_ = roc_auc_score(y, v)
        ap = average_precision_score(y, v)
        inc = (a_ - roc_auc_score(y, s2)
               if "MEAN3" in k or "WMEAN3" in k
               else np.nan)
        rows.append({"outcome": nm, "model": k,
                     "n": len(d),
                     "events": int(y.sum()),
                     "auc": a_, "ap": ap,
                     "inc_vs_wmean2": inc})
        print("  %-30s %.4f  AP %.4f" % (k, a_, ap))

    wx = np.array([w[2] for w in w3])
    print("  CXR fold weights:",
          ", ".join("%.2f" % v for v in wx))

    rng = np.random.RandomState(42)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    dif = []
    cxr_a = []
    for _ in range(NB):
        pk = rng.choice(uq, len(uq), replace=True)
        ii = np.concatenate([idx[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        dif.append(roc_auc_score(y[ii], s3[ii])
                   - roc_auc_score(y[ii], s2[ii]))
        cxr_a.append(roc_auc_score(
            y[ii], d["r_cxr"].values[ii]))
    lo, hi = np.percentile(dif, [2.5, 97.5])
    xlo, xhi = np.percentile(cxr_a, [2.5, 97.5])
    print("  WMEAN3 vs WMEAN2 %+.4f (%.4f, %.4f)"
          % (np.mean(dif), lo, hi))
    print("  CXR alone 95%% CI (%.4f, %.4f)"
          % (xlo, xhi))
    cmp_rows.append({
        "outcome": nm, "n": len(d),
        "events": int(y.sum()),
        "wmean2": roc_auc_score(y, s2),
        "wmean3": roc_auc_score(y, s3),
        "inc": roc_auc_score(y, s3)
               - roc_auc_score(y, s2),
        "lo": lo, "hi": hi,
        "cxr_w_min": wx.min(), "cxr_w_max": wx.max(),
        "cxr_lo": xlo, "cxr_hi": xhi})

t1 = pd.DataFrame(rows)
t1.to_csv(OUT + "/table25_cxr_fixed.csv", index=False)
t2 = pd.DataFrame(cmp_rows)
t2.to_csv(OUT + "/table26_cxr_fixed.csv", index=False)
print("\n===== TABLE 25 =====")
print(t1.to_string(index=False, float_format="%.4f"))
print("\n===== TABLE 26 =====")
print(t2.to_string(index=False, float_format="%.4f"))
