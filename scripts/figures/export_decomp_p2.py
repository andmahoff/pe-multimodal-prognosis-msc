"""Phase 2 rebuild of the composite outcome
decomposition: route of ascertainment, timing, and
route-stratified discrimination.

Uses the harmonised labels and rebuilds WMEAN2 on the
primary EHR+ECG cohort.
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
STEP = 0.05


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
print("n =", len(d), " events =", int(y.sum()))

# ---- rebuild WMEAN2 ----
d["r_ehr"] = rank01(d["p_ehr"].values)
d["r_ecg"] = rank01(d["p_ecg"].values)
R = d[["r_ehr", "r_ecg"]].values
grid = [(w, 1.0 - w)
        for w in np.arange(0, 1.0 + STEP, STEP)]

cv = StratifiedGroupKFold(5, shuffle=True,
                          random_state=42)
score = np.zeros(len(y))
for tr, te in cv.split(R, y, groups=g):
    best, ba = None, -1.0
    for wv in grid:
        s = R[tr] @ np.array(wv)
        a_ = roc_auc_score(y[tr], s)
        if a_ > ba:
            ba, best = a_, wv
    score[te] = R[te] @ np.array(best)
d["score"] = score
print("WMEAN2 AUROC %.4f" % roc_auc_score(y, score))

# ---- route of ascertainment ----
d["route"] = np.where(
    d["death_first"] == 1, "Death",
    np.where(d["cv_first"] == 1, "CV readmission",
             "None"))
ev = d[d[O] == 1].copy()
print("\n--- route split ---")
print(ev["route"].value_counts().to_string())

# ---- days to first event ----
dd = pd.to_numeric(ev["death_days"], errors="coerce")
cc = pd.to_numeric(ev["cv_days"], errors="coerce")
ev["days"] = np.where(ev["route"] == "Death", dd, cc)
ev = ev[ev["days"].notna()]
print("\n--- days to first event ---")
print("mean %.2f  median %.1f  IQR %.0f-%.0f"
      % (ev["days"].mean(), ev["days"].median(),
         ev["days"].quantile(.25),
         ev["days"].quantile(.75)))
print(ev.groupby("route")["days"]
      .agg(["count", "mean", "median"])
      .to_string(float_format="%.2f"))

# ---- route-stratified discrimination ----
rows = []
print("\n--- route-stratified AUROC ---")
for r in ["Death", "CV readmission"]:
    sub = d[(d[O] == 0) | (d["route"] == r)]
    yy = sub[O].values.astype(float)
    if yy.min() == yy.max():
        continue
    a_ = roc_auc_score(yy, sub["score"].values)
    rows.append({"stratum": r, "n": len(sub),
                 "events": int(yy.sum()), "auc": a_})
    print("  %-15s n=%4d ev=%3d AUROC %.4f"
          % (r, len(sub), int(yy.sum()), a_))

# ---- mean predicted risk ----
print("\n--- mean predicted risk ---")
mr = d.groupby("route")["score"].agg(["count", "mean"])
print(mr.to_string(float_format="%.4f"))

# ---- timing windows ----
print("\n--- discrimination by window ---")
tw = [(0, 7), (8, 14), (15, 30)]
for lo, hi in tw:
    keep = (d[O] == 0)
    idx = ev[(ev["days"] >= lo) &
             (ev["days"] <= hi)].index
    sub = d.loc[keep | d.index.isin(idx)]
    yy = sub[O].values.astype(float)
    if yy.min() == yy.max():
        continue
    a_ = roc_auc_score(yy, sub["score"].values)
    print("  %2d-%2dd  ev=%3d  AUROC %.4f"
          % (lo, hi, int(yy.sum()), a_))

# ---- route x timing ----
print("\n--- route by window ---")
ct = pd.crosstab(
    ev["route"],
    pd.cut(ev["days"], [-1, 7, 14, 30],
           labels=["0-7d", "8-14d", "15-30d"]))
print(ct.to_string())

ev[["subject_id", "hadm_id", "route", "days",
    "score"]].to_csv(
    OUT + "/decomp_p2_events.csv", index=False)
pd.DataFrame(rows).to_csv(
    OUT + "/decomp_p2_routes.csv", index=False)
print("\nwrote decomp_p2_events.csv, "
      "decomp_p2_routes.csv")