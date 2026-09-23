import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
OUT = DATA + "/ceiling_xai"
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 7, 13]
OUTS = ["death_30d", "composite_30d",
        "death_30d_inhosp"]
TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]

# lb_* names that duplicate a transferable mi_*
DUP = {
    "lb_white_blood_cells": "wbc",
    "lb_wbc": "wbc",
    "lb_urea_nitrogen": "bun",
    "lb_platelet_count": "plt",
    "lb_sodium": "sodium",
    "lb_potassium": "potassium",
    "lb_chloride": "chloride",
    "lb_creatinine": "creatinine",
    "lb_glucose": "glucose",
    "lb_bicarbonate": "bicarb",
    "lb_anion_gap": "aniongap",
    "lb_calcium_total": "calcium",
    "lb_hematocrit": "hct",
    "lb_hemoglobin": "hgb",
    "lb_red_blood_cells": "rbc",
    "lb_mcv": "mcv",
    "lb_mch": "mch",
    "lb_mchc": "mchc",
    "lb_rdw": "rdw",
}

lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
allb = pd.read_csv(DATA + "/mimic_all_labs.csv")
v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv"
).drop_duplicates("hadm_id")
vit = {"mean_temp": "temp", "mean_hr": "hr",
       "mean_sbp": "sbp", "mean_dbp": "dbp",
       "mean_rr": "rr", "mean_spo2": "spo2"}
keep = ["hadm_id"] + [c for c in vit
                      if c in v7.columns] \
    + [c for c in FLAGS + ["nlr"]
       if c in v7.columns]
mv = v7[keep].rename(columns=vit)

dem = pd.read_csv(
    P2 + "/mimic_pe_mace_cohort.csv",
    usecols=["hadm_id", "age_at_admit",
             "gender"]).drop_duplicates(
    "hadm_id").rename(
    columns={"age_at_admit": "age"})
dem["sex_f"] = (dem["gender"].astype(str)
                .str.upper().str[0]
                == "F").astype(int)
dem = dem[["hadm_id", "age", "sex_f"]]

mm = pd.read_csv(
    DATA + "/mimic_expanded_feats.csv")
d = lab.merge(mm, on=["subject_id", "hadm_id"],
              how="left")
d = d.merge(mv, on="hadm_id", how="left")
d = d.merge(dem, on="hadm_id", how="left")
d = d.merge(allb, on="hadm_id", how="left")

h = pd.read_csv(
    DATA + "/mimic_history_depth.csv")
h = h.rename(columns={"idx_hadm": "hadm_id"})
hc = [c for c in ["n_prior", "n_codes"]
      if c in h.columns]
d = d.merge(h[["hadm_id"] + hc],
            on="hadm_id", how="left")
for c in hc:
    d[c] = d[c].fillna(0)

TRANS = [c for c in
         (["mi_" + a if "mi_" + a in d.columns
           else a for a in TA]
          + FLAGS + ["age"]) if c in d.columns]

raw_lb = [c for c in d.columns
          if c.startswith("lb_")]
dropped = []
LBC = []
for c in raw_lb:
    base = DUP.get(c)
    if base is not None:
        tn = "mi_" + base
        if tn in TRANS or base in TRANS:
            dropped.append(c)
            continue
    LBC.append(c)

print("lb_ columns:", len(raw_lb))
print("dropped as duplicates:", len(dropped))
for c in dropped:
    print("   ", c, "->", DUP[c])

EXTRA = [c for c in
         ["spo2", "sex_f", "nlr"] + hc
         if c in d.columns]
BAN = ["30d", "first", "label", "composite",
       "death", "mace", "_y"]
ALLF = sorted(set(TRANS + LBC + EXTRA))
ALLF = [c for c in ALLF
        if not any(b in c.lower()
                   for b in BAN)]
TRANS = [c for c in TRANS
         if not any(b in c.lower()
                    for b in BAN)]
IS_T = {c: (c in TRANS) for c in ALLF}
print("")
print("ALL (deduplicated):", len(ALLF))
print("transferable:", len(TRANS))
print("non-transferable:",
      len(ALLF) - len(TRANS))


def mk_gb():
    return HistGradientBoostingClassifier(
        random_state=42, max_depth=3,
        learning_rate=0.05, max_iter=300,
        l2_regularization=1.0)


def cvauc(X, y, grp):
    a = []
    for s in SEEDS:
        cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True,
            random_state=s)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, grp):
            m = mk_gb()
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(
                X[te])[:, 1]
        a.append(roc_auc_score(y, oof))
    return float(np.mean(a))


rows = []
for out in OUTS:
    dd = d
    y = dd[out].values.astype(int)
    grp = dd["subject_id"].values
    X = dd[ALLF].values.astype(float)

    print("")
    print("=" * 64)
    print("%s  n=%d ev=%d  (%d feats)"
          % (out, len(y), int(y.sum()),
             len(ALLF)))

    a_all = cvauc(X, y, grp)
    print("  CV AUC all feats        %.4f"
          % a_all)

    nc = [c for c in ALLF if c != "n_codes"]
    a_nc = cvauc(
        dd[nc].values.astype(float), y, grp)
    print("  CV AUC without n_codes  %.4f"
          "  (delta %+.4f)"
          % (a_nc, a_nc - a_all))

    pl = Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1, max_iter=5000))])
    pl.fit(X, y)
    Z = pl.named_steps["sc"].transform(
        pl.named_steps["im"].transform(X))
    beta = pl.named_steps["lr"].coef_[0]
    shap = np.abs(Z * beta).mean(axis=0)

    m = mk_gb()
    m.fit(X, y)
    pi = permutation_importance(
        m, X, y, n_repeats=10,
        random_state=42, scoring="roc_auc",
        n_jobs=1)

    t = pd.DataFrame({
        "feature": ALLF,
        "transferable": [IS_T[c] for c in ALLF],
        "beta": beta, "shap": shap,
        "perm": pi.importances_mean,
        "perm_sd": pi.importances_std,
        "cov": [float(dd[c].notna().mean())
                for c in ALLF]})
    t = t.sort_values("perm", ascending=False)
    t.to_csv(OUT + "/ceiling_imp_" + out
             + ".csv", index=False)

    print("")
    print("  TOP 20 BY PERMUTATION IMPORTANCE")
    print("  %-26s %-6s %8s %8s %6s"
          % ("feature", "trans", "perm",
             "|SHAP|", "cov"))
    for _, r in t.head(20).iterrows():
        print("  %-26s %-6s %8.4f %8.3f %6.2f"
              % (r["feature"][:26],
                 "yes" if r["transferable"]
                 else "NO", r["perm"],
                 r["shap"], r["cov"]))

    print("")
    print("  HISTORY FEATURES")
    for c in hc:
        rr = t[t["feature"] == c]
        if len(rr):
            rk = int((t["perm"] > float(
                rr["perm"].iloc[0])).sum()) + 1
            print("  %-12s rank %3d of %d"
                  "  perm %+.4f  beta %+.3f"
                  % (c, rk, len(t),
                     float(rr["perm"].iloc[0]),
                     float(rr["beta"].iloc[0])))

    print("")
    print("  TOP 12 NON-TRANSFERABLE"
          " (genuinely missing)")
    for _, r in t[~t["transferable"]].head(
            12).iterrows():
        print("  %-26s perm %+.4f  beta %+.3f"
              "  cov %.2f"
              % (r["feature"][:26], r["perm"],
                 r["beta"], r["cov"]))

    st = t.groupby("transferable")[
        ["perm", "shap"]].sum()
    print("")
    print("  IMPORTANCE SHARE")
    tot_p = float(st["perm"].sum())
    tot_s = float(st["shap"].sum())
    for k in st.index:
        lb = "transferable" if k else \
            "non-transferable"
        print("    %-18s n=%2d  perm %.4f"
              " (%.1f%%)  |SHAP| %.3f (%.1f%%)"
              % (lb, int((t["transferable"]
                          == k).sum()),
                 st.loc[k, "perm"],
                 100 * st.loc[k, "perm"] / tot_p
                 if tot_p else 0,
                 st.loc[k, "shap"],
                 100 * st.loc[k, "shap"] / tot_s
                 if tot_s else 0))

    for _, r in t.iterrows():
        rows.append({"outcome": out,
                     "auc_all": a_all,
                     "auc_no_ncodes": a_nc,
                     **r.to_dict()})

pd.DataFrame(rows).to_csv(
    OUT + "/ceiling_imp_all.csv", index=False)
print("")
print("saved to", OUT)
