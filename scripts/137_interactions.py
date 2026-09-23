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
OUT = DATA + "/interaction_results.csv"

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


def add_inter(d, pfx):
    def g(n):
        for c in [pfx + n, n]:
            if c in d.columns:
                return d[c].astype(float)
        return pd.Series(np.nan,
                         index=d.index)

    e = pd.DataFrame(index=d.index)
    hr, sbp, dbp = g("hr"), g("sbp"), g("dbp")
    e["ix_shock"] = hr / sbp.replace(0, np.nan)
    e["ix_pulsepr"] = sbp - dbp
    e["ix_map"] = (sbp + 2 * dbp) / 3.0
    e["ix_bun_cr"] = g("bun") / g(
        "creatinine").replace(0, np.nan)
    e["ix_ag_bicarb"] = g("aniongap") * g(
        "bicarb")
    e["ix_age_cancer"] = d["age"].astype(
        float) * d["cancer"].astype(float)
    e["ix_age_hf"] = d["age"].astype(
        float) * d["heart_failure"].astype(
        float)
    e["ix_rdw_hgb"] = g("rdw") / g(
        "hgb").replace(0, np.nan)
    e["ix_na_cl"] = g("sodium") - g("chloride")
    e["ix_ca_alb"] = g("calcium")
    e["ix_hrxrr"] = hr * g("rr")
    e = e.replace([np.inf, -np.inf], np.nan)
    return e


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
IE = add_inter(ins, "m7_")
for c in IE.columns:
    ins[c] = IE[c]

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
ME = add_inter(mm, "mi_")
for c in ME.columns:
    mm[c] = ME[c]

IX = list(IE.columns)
IC = ["m7_" + a for a in TA] + FLAGS + ["age"]
MC = ["mi_" + a if "mi_" + a in mm.columns
      else a for a in TA] + FLAGS + ["age"]
IC = [c for c in IC if c in ins.columns]
MC = [c for c in MC if c in mm.columns]
print("base:", len(IC), " interactions:",
      len(IX))

print("")
print("INTERACTION SANITY (median)")
print("  %-14s %10s %10s" % ("term",
                             "INSPECT",
                             "MIMIC"))
for c in IX:
    print("  %-14s %10.3f %10.3f"
          % (c, float(ins[c].median()),
             float(mm[c].median())))


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


def cvauc(X, y, grp, mk):
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
    print("=" * 60)
    print("%s  INSPECT ev=%d | MIMIC ev=%d"
          % (icol, int(yi.sum()),
             int(ym.sum())))

    for nm, ic, mc in [
            ("base", IC, MC),
            ("base+inter", IC + IX,
             MC + IX)]:
        Xi = di[ic].values.astype(float)
        Xm = dm[mc].values.astype(float)
        for ln, mk in [("lr", mk_lr),
                       ("gb", mk_gb)]:
            m = mk()
            m.fit(Xi, yi)
            p = m.predict_proba(Xm)[:, 1]
            zs = roc_auc_score(ym, p)
            ap = average_precision_score(ym, p)
            cv = cvauc(Xi, yi, gi, mk)
            print("  %-11s %s  zero-shot"
                  " %.4f  AP %.4f |"
                  " ins-CV %.4f  (%d feats)"
                  % (nm, ln, zs, ap, cv,
                     Xi.shape[1]))
            rows.append({
                "outcome": icol, "set": nm,
                "learner": ln,
                "nfeat": Xi.shape[1],
                "zs_auc": zs, "ap": ap,
                "ins_cv": cv})

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string())
print("")
print("INTERACTION EFFECT (base+inter - base)")
for o in r["outcome"].unique():
    for ln in ["lr", "gb"]:
        s = r[(r["outcome"] == o)
              & (r["learner"] == ln)]

        def g(k, col):
            v = s[s["set"] == k][col]
            return float(v.iloc[0]) if len(v) \
                else np.nan
        print("  %-14s %s  zero-shot %+.4f |"
              " in-dist %+.4f"
              % (o, ln,
                 g("base+inter", "zs_auc")
                 - g("base", "zs_auc"),
                 g("base+inter", "ins_cv")
                 - g("base", "ins_cv")))
