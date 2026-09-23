import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

PROJ = os.environ.get("GCP_PROJECT_ID")
BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
LAB = DATA + "/mimic_labels_harmonised.csv"
V7 = P2 + "/mimic_pe_ehr_baseline_v7.csv"
CACHE = DATA + "/mimic_expanded_labs.csv"
OUT = DATA + "/expand_results.csv"

SEEDS = [42, 7, 13]
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]

NEW = {
    51221: "hct", 51279: "rbc", 51250: "mcv",
    51265: "plt", 51277: "rdw", 51248: "mch",
    51249: "mchc", 51006: "bun",
    50971: "potassium", 50983: "sodium",
    50902: "chloride", 50882: "bicarb",
    50868: "aniongap", 50931: "glucose",
    51275: "ptt", 51274: "pt", 51237: "inr",
    50960: "magnesium", 50893: "calcium",
    50970: "phosphate",
}

lab = pd.read_csv(LAB)

if os.path.exists(CACHE):
    ex = pd.read_csv(CACHE)
    print("cached labs:", ex.shape)
else:
    cl = bigquery.Client(project=PROJ)
    hs = ",".join(str(h) for h in
                  sorted(lab["hadm_id"].unique()))
    its = ",".join(str(i) for i in NEW)
    q = """
    SELECT hadm_id, itemid,
           AVG(valuenum) AS v
    FROM `physionet-data.mimiciv_3_1_hosp.labevents`
    WHERE hadm_id IN (%s)
      AND itemid IN (%s)
      AND valuenum IS NOT NULL
    GROUP BY hadm_id, itemid
    """ % (hs, its)
    r = cl.query(q).to_dataframe()
    print("rows:", len(r))
    r["name"] = r["itemid"].map(NEW)
    ex = r.pivot_table(index="hadm_id",
                       columns="name",
                       values="v").reset_index()
    ex.to_csv(CACHE, index=False)
    print("cached:", ex.shape)

v7 = pd.read_csv(V7)
OLD = [c for c in v7.columns
       if c.startswith("mean_")
       or c in ("nlr", "afib", "cancer",
                "copd", "heart_failure")]
print("old features:", len(OLD))

d = lab.merge(v7[["hadm_id"] + OLD].drop_duplicates(
    "hadm_id"), on="hadm_id", how="left")
d = d.merge(ex, on="hadm_id", how="left")
NEWC = [c for c in NEW.values() if c in d.columns]
print("new features:", len(NEWC))
print("merged:", len(d))

print("")
print("new-feature missingness:")
print(d[NEWC].isna().mean().round(3)
      .sort_values().to_string())

SETS = {"old15": OLD,
        "new20": NEWC,
        "combined": OLD + NEWC}


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


def evalcv(X, y, grp, mk):
    aucs = []
    ap = np.nan
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
        aucs.append(roc_auc_score(y, oof))
        if s == 42:
            ap = average_precision_score(y, oof)
    return (float(np.mean(aucs)),
            float(np.std(aucs)), ap)


res = []
for out in OUTS:
    dd = d
    if out == "cv_first":
        dd = d[d["death_first"] == 0]
    y = dd[out].values.astype(int)
    grp = dd["subject_id"].values
    if y.sum() < 20:
        continue
    print("")
    print("=" * 56)
    print(out, "n=%d ev=%d (%.4f)"
          % (len(y), int(y.sum()), y.mean()))
    for nm, cols in SETS.items():
        X = dd[cols].values.astype(float)
        for mn, mk in [("lr", mk_lr),
                       ("gb", mk_gb)]:
            a, s, ap = evalcv(X, y, grp, mk)
            print("  %-9s %s  AUC %.4f +/-"
                  " %.4f  AP %.4f (%d feats)"
                  % (nm, mn, a, s, ap,
                     X.shape[1]))
            res.append({"outcome": out,
                        "set": nm,
                        "learner": mn,
                        "n": len(y),
                        "ev": int(y.sum()),
                        "nfeat": X.shape[1],
                        "auc": a, "sd": s,
                        "ap": ap})

r = pd.DataFrame(res)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string())

print("")
print("EXPANSION EFFECT (combined - old15)")
for out in r["outcome"].unique():
    for mn in ["lr", "gb"]:
        s = r[(r["outcome"] == out)
              & (r["learner"] == mn)]

        def g(k):
            v = s[s["set"] == k]["auc"]
            return float(v.iloc[0]) if len(v) \
                else np.nan
        print("  %-18s %s  %+.4f  (old %.4f"
              " -> new %.4f)"
              % (out, mn,
                 g("combined") - g("old15"),
                 g("old15"), g("combined")))
