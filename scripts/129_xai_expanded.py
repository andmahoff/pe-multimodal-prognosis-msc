import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
IW = BASE + "/inspect_workspace/data/processed"
OUT = DATA + "/xai_expanded"
os.makedirs(OUT, exist_ok=True)

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
keep = ["hadm_id"] + list(vit) + FLAGS
mm = mm.merge(
    v7[[c for c in keep if c in v7.columns]]
    .rename(columns=vit),
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

IC = ["m7_" + a for a in TA] + EXTRA
MC = ["mi_" + a if "mi_" + a in mm.columns
      else a for a in TA] + EXTRA
NAMES = TA + EXTRA
print("features:", len(IC))


def mk():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1, max_iter=5000))])


def shap_linear(pipe, X):
    im = pipe.named_steps["im"]
    sc = pipe.named_steps["sc"]
    lr = pipe.named_steps["lr"]
    Z = sc.transform(im.transform(X))
    return Z * lr.coef_[0]


summary = []
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

    Xi = di[IC].values.astype(float)
    Xm = dm[MC].values.astype(float)
    pipe = mk()
    pipe.fit(Xi, yi)

    Si = shap_linear(pipe, Xi)
    Sm = shap_linear(pipe, Xm)
    beta = pipe.named_steps["lr"].coef_[0]

    imp_i = np.abs(Si).mean(axis=0)
    imp_m = np.abs(Sm).mean(axis=0)
    dir_m = Sm.mean(axis=0)

    t = pd.DataFrame({
        "feature": NAMES,
        "beta": beta,
        "shap_inspect": imp_i,
        "shap_mimic": imp_m,
        "signed_mimic": dir_m,
        "rank_ins": (-imp_i).argsort().argsort() + 1,
        "rank_mim": (-imp_m).argsort().argsort() + 1,
    })
    t["rank_shift"] = t["rank_ins"] - t["rank_mim"]
    t = t.sort_values("shap_mimic",
                      ascending=False)
    t.to_csv(OUT + "/shap_" + mcol + ".csv",
             index=False)

    rho = spearmanr(imp_i, imp_m)[0]
    p = pipe.predict_proba(Xm)[:, 1]
    auc = roc_auc_score(ym, p)

    print("")
    print("=" * 62)
    print("%s   MIMIC AUC %.4f" % (mcol, auc))
    print("  source-target SHAP rank rho"
          " = %.3f" % rho)
    print("")
    print("  %-12s %8s %8s %8s %6s"
          % ("feature", "beta", "|SHAP|m",
             "signed", "shift"))
    for _, r in t.head(15).iterrows():
        print("  %-12s %+8.3f %8.3f %+8.3f"
              " %+6d"
              % (r["feature"], r["beta"],
                 r["shap_mimic"],
                 r["signed_mimic"],
                 int(r["rank_shift"])))

    print("")
    print("  LARGEST SOURCE->TARGET SHIFTS")
    s2 = t.reindex(
        t["rank_shift"].abs()
        .sort_values(ascending=False).index)
    for _, r in s2.head(6).iterrows():
        print("  %-12s ins rank %2d -> mim %2d"
              "  (|SHAP| %.3f -> %.3f)"
              % (r["feature"], int(r["rank_ins"]),
                 int(r["rank_mim"]),
                 r["shap_inspect"],
                 r["shap_mimic"]))

    pi = permutation_importance(
        pipe, Xm, ym, n_repeats=20,
        random_state=42, scoring="roc_auc")
    pt = pd.DataFrame({
        "feature": NAMES,
        "perm_drop": pi.importances_mean,
        "perm_sd": pi.importances_std})
    pt = pt.sort_values("perm_drop",
                        ascending=False)
    pt.to_csv(OUT + "/perm_" + mcol + ".csv",
              index=False)
    print("")
    print("  PERMUTATION IMPORTANCE (top 8)")
    for _, r in pt.head(8).iterrows():
        print("  %-12s %.4f +/- %.4f"
              % (r["feature"], r["perm_drop"],
                 r["perm_sd"]))

    print("")
    print("  QUARTILE EFFECT (top 5 by |SHAP|)")
    for f in t.head(5)["feature"]:
        j = NAMES.index(f)
        v = Xm[:, j]
        ok = ~np.isnan(v)
        if ok.sum() < 100:
            continue
        q = pd.qcut(pd.Series(v[ok]), 4,
                    labels=False,
                    duplicates="drop")
        er = [float(ym[ok][q == k].mean())
              for k in sorted(pd.unique(q))
              if not pd.isna(k)]
        print("  %-12s event rate by quartile:"
              " %s" % (f, [round(e, 3)
                           for e in er]))

    summary.append({"outcome": mcol,
                    "auc": auc, "rho": rho})

print("")
print(pd.DataFrame(summary).round(4)
      .to_string(index=False))
print("saved to", OUT)
