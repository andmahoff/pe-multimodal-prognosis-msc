import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
IW = BASE + "/inspect_workspace/data/processed"

SEEDS = [42, 7, 13]
TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]
PAIRS = [("death_30d", "death_30d"),
         ("composite_30d", "composite_30d"),
         ("cv_first", "cv_first")]

ins = pd.read_csv(
    DATA + "/inspect_expanded_feats.csv")
ilab = pd.read_csv(
    DATA + "/inspect_labels_final.csv")
dem = pd.read_csv(
    IW + "/final_feature_matrix_v2_labeled.csv",
    usecols=["person_id", "age"] + FLAGS)
ins = ins.merge(dem.drop_duplicates("person_id"),
                on="person_id", how="left")
ins = ins.merge(
    ilab[["person_id", "death_30d",
          "composite_30d", "cv_first",
          "death_first"]],
    on="person_id", how="left")

mm = pd.read_csv(
    DATA + "/mimic_expanded_feats.csv")
v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv"
).drop_duplicates("hadm_id")
vit = {"mean_temp": "temp", "mean_hr": "hr",
       "mean_sbp": "sbp", "mean_dbp": "dbp",
       "mean_rr": "rr"}
keep = ["hadm_id"] + [c for c in vit
                      if c in v7.columns] \
    + [c for c in FLAGS if c in v7.columns]
mm = mm.merge(v7[keep].rename(columns=vit),
              on="hadm_id", how="left")
age = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v6.csv",
    usecols=["hadm_id", "age_at_admit"])
mm = mm.merge(
    age.dropna().drop_duplicates("hadm_id")
    .rename(columns={"age_at_admit": "age"}),
    on="hadm_id", how="left")
mlab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
mm = mm.merge(mlab, on=["subject_id",
                        "hadm_id"], how="inner")

# recomputed anion gap, same formula both sides
ins["ag_calc"] = (ins["m7_sodium"]
                  - ins["m7_chloride"]
                  - ins["m7_bicarb"])
mc_na = "mi_sodium" if "mi_sodium" in \
    mm.columns else "sodium"
mc_cl = "mi_chloride" if "mi_chloride" in \
    mm.columns else "chloride"
mc_hc = "mi_bicarb" if "mi_bicarb" in \
    mm.columns else "bicarb"
mm["ag_calc"] = (mm[mc_na] - mm[mc_cl]
                 - mm[mc_hc])

print("ANION GAP COMPARISON (median)")
print("  reported   INSPECT %.2f  MIMIC %.2f"
      % (float(ins["m7_aniongap"].median()),
         float(mm["mi_aniongap"].median())
         if "mi_aniongap" in mm.columns
         else float(mm["aniongap"].median())))
print("  recomputed INSPECT %.2f  MIMIC %.2f"
      % (float(ins["ag_calc"].median()),
         float(mm["ag_calc"].median())))
print("  potassium  INSPECT %.2f  MIMIC %.2f"
      % (float(ins["m7_potassium"].median()),
         float(mm[
             "mi_potassium"
             if "mi_potassium" in mm.columns
             else "potassium"].median())))

BASEI = ["m7_" + a for a in TA] + FLAGS \
    + ["age"]
BASEM = ["mi_" + a if "mi_" + a in mm.columns
         else a for a in TA] + FLAGS + ["age"]
NO_I = [c for c in BASEI
        if "aniongap" not in c]
NO_M = [c for c in BASEM
        if "aniongap" not in c]
CA_I = NO_I + ["ag_calc"]
CA_M = NO_M + ["ag_calc"]

SETS = {"reported (28)": (BASEI, BASEM),
        "drop AG (27)": (NO_I, NO_M),
        "recomputed AG (28)": (CA_I, CA_M)}


def mk():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1, max_iter=5000))])


def cvauc(X, y, grp):
    a = []
    for s in SEEDS:
        cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True,
            random_state=s)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, grp):
            m = mk()
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(
                X[te])[:, 1]
        a.append(roc_auc_score(y, oof))
    return float(np.mean(a))


rows = []
for icol, mcol in PAIRS:
    di, dm = ins, mm
    if icol == "cv_first":
        di = ins[ins["death_first"] == 0]
        dm = mm[mm["death_first"] == 0]
    yi = pd.to_numeric(
        di[icol], errors="coerce").fillna(
        0).astype(int).values
    ym = dm[mcol].values.astype(int)
    gi = di["person_id"].values

    print("")
    print("=" * 56)
    print("%s  (MIMIC ev=%d)"
          % (icol, int(ym.sum())))
    base = None
    for nm, (ic, mc) in SETS.items():
        Xi = di[ic].values.astype(float)
        Xm = dm[mc].values.astype(float)
        m = mk()
        m.fit(Xi, yi)
        p = m.predict_proba(Xm)[:, 1]
        zs = roc_auc_score(ym, p)
        ap = average_precision_score(ym, p)
        cv = cvauc(Xi, yi, gi)
        if base is None:
            base = zs
        print("  %-20s zero-shot %.4f"
              " (%+.4f)  AP %.4f | ins-CV"
              " %.4f" % (nm, zs, zs - base,
                         ap, cv))
        rows.append({"outcome": icol,
                     "set": nm, "zs": zs,
                     "delta": zs - base,
                     "ap": ap, "cv": cv})

r = pd.DataFrame(rows)
r.to_csv(DATA + "/aniongap_check.csv",
         index=False)
print("")
print(r.round(4).to_string())
