import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
IW = BASE + "/inspect_workspace/data/processed"
OUT = DATA + "/expanded_retrain_results.csv"

SEEDS = [42, 7, 13]
NBOOT = 2000

TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
TB = ["inr", "pt", "aptt", "magnesium",
      "albumin", "bili", "alp", "alt", "ast",
      "neut_pct", "lymph_pct", "mono_pct",
      "eos_pct", "baso_pct"]
TC = ["ntprobnp", "phosphate"]
TROP = ["troponin"]

# ---------- INSPECT ----------
ins = pd.read_csv(
    DATA + "/inspect_expanded_feats.csv")
lab = pd.read_csv(
    DATA + "/inspect_labels_final.csv")
dem = pd.read_csv(
    IW + "/final_feature_matrix_v2_labeled.csv",
    usecols=["person_id", "age", "afib",
             "cancer", "copd",
             "heart_failure"])
dem = dem.drop_duplicates("person_id")
ins = ins.merge(dem, on="person_id", how="left")
ins = ins.merge(
    lab[["person_id", "death_30d",
         "composite_30d", "cv_first",
         "death_first"]],
    on="person_id", how="left")
print("INSPECT rows:", len(ins))

# ---------- MIMIC ----------
mm = pd.read_csv(
    DATA + "/mimic_expanded_feats.csv")
v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv")
v7 = v7.drop_duplicates("hadm_id")
vit = {"mean_temp": "temp", "mean_hr": "hr",
       "mean_sbp": "sbp", "mean_dbp": "dbp",
       "mean_rr": "rr"}
keep = ["hadm_id"] + list(vit) + [
    "afib", "cancer", "copd",
    "heart_failure"]
keep = [c for c in keep if c in v7.columns]
mv = v7[keep].rename(columns=vit)
mm = mm.merge(mv, on="hadm_id", how="left")

age = None
for f in ["mimic_pe_ehr_baseline_v6.csv",
          "mimic_pe_ehr_baseline.csv"]:
    p = os.path.join(P2, f)
    if os.path.exists(p):
        a = pd.read_csv(p, usecols=["hadm_id",
                                    "age_at_admit"])
        age = a.dropna().drop_duplicates("hadm_id")
        age = age.rename(
            columns={"age_at_admit": "age"})
        print("age from", f)
        break
if age is not None:
    mm = mm.merge(age, on="hadm_id", how="left")

mlab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
mm = mm.merge(mlab, on=["subject_id",
                        "hadm_id"], how="inner")
print("MIMIC rows:", len(mm))

FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]
CELLS = {
    "current": (["temp", "hr", "sbp", "dbp",
                 "rr", "creatinine", "hgb",
                 "wbc", "neut_pct",
                 "lymph_pct", "albumin"],
                False),
    "tierA": (TA, True),
    "tierA_trop": (TA + TROP, True),
    "tierAB": (TA + TB, True),
    "tierABC": (TA + TB + TC, True),
    "tierABC_trop": (TA + TB + TC + TROP, True),
}

PAIRS = [("death_30d", "death_30d"),
         ("composite_30d", "composite_30d"),
         ("cv_first", "cv_first")]


def mk_lr():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1, max_iter=5000))])


def mk_gb():
    return HistGradientBoostingClassifier(
        random_state=42, max_depth=3,
        learning_rate=0.05, max_iter=300,
        l2_regularization=1.0)


def boot(y, p, grp):
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(42)
    b = []
    for _ in range(NBOOT):
        pk = rng.choice(us, len(us),
                        replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        b.append(roc_auc_score(y[ii], p[ii]))
    b = np.array(b)
    return (float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)))


res = []
for icol, mcol in PAIRS:
    di = ins
    dm = mm
    if icol == "cv_first":
        di = ins[ins["death_first"] == 0]
        dm = mm[mm["death_first"] == 0]
    yi = pd.to_numeric(di[icol],
                       errors="coerce").fillna(
        0).astype(int).values
    ym = dm[mcol].values.astype(int)
    if yi.sum() < 20 or ym.sum() < 20:
        continue
    print("")
    print("=" * 60)
    print("%s  INSPECT n=%d ev=%d |"
          " MIMIC n=%d ev=%d"
          % (icol, len(yi), int(yi.sum()),
             len(ym), int(ym.sum())))

    for cell, (an, use_age) in CELLS.items():
        ic = ["m7_" + a for a in an]
        mc = ["mi_" + a if "mi_" + a
              in dm.columns else a for a in an]
        ic = [c for c in ic if c in di.columns]
        mc = [c for c in mc if c in dm.columns]
        if len(ic) != len(mc):
            print("  %s: col mismatch %d/%d"
                  % (cell, len(ic), len(mc)))
            continue
        extra = FLAGS + (["age"] if use_age
                         else [])
        extra = [c for c in extra
                 if c in di.columns
                 and c in dm.columns]
        Xi = di[ic + extra].values.astype(float)
        Xm = dm[mc + extra].values.astype(float)

        for mn, mk in [("lr", mk_lr),
                       ("gb", mk_gb)]:
            m = mk()
            m.fit(Xi, yi)
            p = m.predict_proba(Xm)[:, 1]
            a = roc_auc_score(ym, p)
            ap = average_precision_score(ym, p)
            lo, hi = boot(
                ym, p, dm["subject_id"].values)

            cvs = []
            for s in SEEDS:
                cv = StratifiedGroupKFold(
                    n_splits=5, shuffle=True,
                    random_state=s)
                oof = np.zeros(len(yi))
                gi = di["person_id"].values
                for tr, te in cv.split(Xi, yi,
                                       gi):
                    mm2 = mk()
                    mm2.fit(Xi[tr], yi[tr])
                    oof[te] = mm2.predict_proba(
                        Xi[te])[:, 1]
                cvs.append(roc_auc_score(yi, oof))

            print("  %-13s %s  ZS %.4f"
                  " [%.4f,%.4f] AP %.4f |"
                  " ins-CV %.4f  (%d feats)"
                  % (cell, mn, a, lo, hi, ap,
                     float(np.mean(cvs)),
                     Xi.shape[1]))
            res.append({
                "outcome": icol, "cell": cell,
                "learner": mn,
                "nfeat": Xi.shape[1],
                "zs_auc": a, "lo": lo, "hi": hi,
                "ap": ap,
                "ins_cv": float(np.mean(cvs))})

r = pd.DataFrame(res)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string())
print("")
print("REF 15-feature zero-shot EHR modality:")
print("  death_30d 0.7890  composite 0.7714"
      "  cv_first 0.7965")
