import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
IW = BASE + "/inspect_workspace/data/processed"

TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]
EXTRA = FLAGS + ["age"]

PAIRS = [("death_30d", "death_30d"),
         ("composite_30d", "composite_30d"),
         ("cv_first", "cv_first")]

# ---------- INSPECT ----------
ins = pd.read_csv(
    DATA + "/inspect_expanded_feats.csv")
lab = pd.read_csv(
    DATA + "/inspect_labels_final.csv")
dem = pd.read_csv(
    IW + "/final_feature_matrix_v2_labeled.csv",
    usecols=["person_id", "age"] + FLAGS)
dem = dem.drop_duplicates("person_id")
ins = ins.merge(dem, on="person_id", how="left")
ins = ins.merge(
    lab[["person_id", "death_30d",
         "composite_30d", "cv_first",
         "death_first"]],
    on="person_id", how="left")
print("INSPECT:", len(ins))

# ---------- MIMIC ----------
mm = pd.read_csv(
    DATA + "/mimic_expanded_feats.csv")
v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv")
v7 = v7.drop_duplicates("hadm_id")
vit = {"mean_temp": "temp", "mean_hr": "hr",
       "mean_sbp": "sbp", "mean_dbp": "dbp",
       "mean_rr": "rr"}
keep = ["hadm_id"] + list(vit) + FLAGS
keep = [c for c in keep if c in v7.columns]
mm = mm.merge(v7[keep].rename(columns=vit),
              on="hadm_id", how="left")

age = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v6.csv",
    usecols=["hadm_id", "age_at_admit"])
age = age.dropna().drop_duplicates("hadm_id")
age = age.rename(columns={"age_at_admit": "age"})
mm = mm.merge(age, on="hadm_id", how="left")

mlab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
mm = mm.merge(mlab, on=["subject_id",
                        "hadm_id"], how="inner")
print("MIMIC:", len(mm))

ICOLS = ["m7_" + a for a in TA] + EXTRA
MCOLS = ["mi_" + a if "mi_" + a in mm.columns
         else a for a in TA] + EXTRA
miss = [c for c in ICOLS
        if c not in ins.columns]
miss += [c for c in MCOLS
         if c not in mm.columns]
if miss:
    raise SystemExit("missing cols: %s" % miss)
print("features:", len(ICOLS))

BAN = ["30d", "first", "label", "composite",
       "death", "mace"]
for c in ICOLS + MCOLS:
    if any(b in c.lower() for b in BAN):
        raise SystemExit("LEAK: %s" % c)


def mk():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1, max_iter=5000))])


for icol, mcol in PAIRS:
    di = ins
    dm = mm
    if icol == "cv_first":
        di = ins[ins["death_first"] == 0]
        dm = mm[mm["death_first"] == 0]
    yi = pd.to_numeric(
        di[icol], errors="coerce").fillna(
        0).astype(int).values
    ym = dm[mcol].values.astype(int)

    m = mk()
    m.fit(di[ICOLS].values.astype(float), yi)
    p = m.predict_proba(
        dm[MCOLS].values.astype(float))[:, 1]

    a = roc_auc_score(ym, p)
    ap = average_precision_score(ym, p)
    print("")
    print("%-15s INSPECT n=%d ev=%d ->"
          " MIMIC n=%d ev=%d"
          % (icol, len(yi), int(yi.sum()),
             len(ym), int(ym.sum())))
    print("   zero-shot AUC %.4f  AP %.4f"
          % (a, ap))

    out = pd.DataFrame({
        "subject_id": dm["subject_id"].values,
        "hadm_id": dm["hadm_id"].values,
        "p_ehr": p})
    pth = DATA + "/p_ehr_harm_" + mcol + ".csv"
    out.to_csv(pth, index=False)
    print("   saved", os.path.basename(pth),
          out.shape)

    co = pd.Series(
        m.named_steps["lr"].coef_[0],
        index=ICOLS).sort_values(
        ascending=False)
    print("   top +:",
          co.head(5).round(3).to_dict())
    print("   top -:",
          co.tail(4).round(3).to_dict())

print("")
print("REF 15-feature EHR modality: death 0.7890"
      "  composite 0.7714  cv_first 0.7965")

