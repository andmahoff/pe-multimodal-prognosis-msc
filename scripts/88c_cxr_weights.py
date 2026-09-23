"""Save the per-fold EHR, ECG and CXR weights of the
three-modality weighted average, using the CXR
predictions in p_cxr_harm_*_fixed.csv.
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
STEP = 0.05

OUTS = [("composite_30d", "Composite (30-day)"),
        ("death_30d", "30-day death"),
        ("cv_first", "CV readmission")]
AT_RISK = ["cv_first"]


def rank01(x):
    return rankdata(x) / (len(x) + 1.0)


n = int(round(1.0 / STEP))
GRID = [(i * STEP, j * STEP, (n - i - j) * STEP)
        for i in range(n + 1)
        for j in range(n + 1 - i)]

lab = pd.read_csv(DATA + "/mimic_labels_harmonised.csv")
rows = []

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
    R = d[["r_ehr", "r_ecg", "r_cxr"]].values

    cv = StratifiedGroupKFold(5, shuffle=True,
                              random_state=42)
    print("---", nm, " n =", len(d),
          " events =", int(y.sum()))
    for k, (tr, te) in enumerate(
            cv.split(R, y, groups=g), start=1):
        b, ba = None, -1.0
        for wv in GRID:
            a_ = roc_auc_score(y[tr],
                               R[tr] @ np.array(wv))
            if a_ > ba:
                ba, b = a_, wv
        rows.append({"outcome": nm, "fold": k,
                     "EHR": b[0], "ECG": b[1],
                     "CXR": b[2]})
        print("  fold %d  EHR %.2f  ECG %.2f  CXR %.2f"
              % (k, b[0], b[1], b[2]))

t = pd.DataFrame(rows)
t.to_csv(OUT + "/wmean3cxr_weights_fixed.csv",
         index=False)
print("\nwrote wmean3cxr_weights_fixed.csv")
print(t.groupby("outcome")[["EHR", "ECG", "CXR"]]
      .agg(["min", "max"]).to_string())

