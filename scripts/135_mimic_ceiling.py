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
CACHE = DATA + "/mimic_all_labs.csv"
HIST = DATA + "/mimic_history_depth.csv"
OUT = DATA + "/ceiling_results.csv"

SEEDS = [42, 7, 13]
MIN_COV = 0.15
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]

TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]

lab = pd.read_csv(LAB)
n_adm = lab["hadm_id"].nunique()
print("cohort:", n_adm)

if os.path.exists(CACHE):
    allb = pd.read_csv(CACHE)
    print("cached labs:", allb.shape)
else:
    cl = bigquery.Client(project=PROJ)
    hs = ",".join(str(h) for h in
                  sorted(lab["hadm_id"].unique()))
    q = """
    SELECT l.itemid,
           ANY_VALUE(d.label) AS label,
           COUNT(DISTINCT l.hadm_id) AS n_adm
    FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
    JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` d
      ON l.itemid = d.itemid
    WHERE l.hadm_id IN (%s)
      AND l.valuenum IS NOT NULL
    GROUP BY l.itemid
    HAVING COUNT(DISTINCT l.hadm_id) >= %d
    """ % (hs, int(MIN_COV * n_adm))
    it = cl.query(q).to_dataframe()
    it = it[~it["label"].isin(["I", "H", "L"])]
    print("analytes above %.2f coverage: %d"
          % (MIN_COV, len(it)))
    ids = ",".join(str(i) for i in it["itemid"])
    q2 = """
    SELECT hadm_id, itemid, AVG(valuenum) AS v
    FROM `physionet-data.mimiciv_3_1_hosp.labevents`
    WHERE hadm_id IN (%s)
      AND itemid IN (%s)
      AND valuenum IS NOT NULL
    GROUP BY hadm_id, itemid
    """ % (hs, ids)
    r = cl.query(q2).to_dataframe()
    print("rows:", len(r))
    nm = dict(zip(it["itemid"], it["label"]))
    r["nm"] = r["itemid"].map(
        lambda i: "lb_" + str(nm[i]).lower()
        .replace(" ", "_").replace(",", "")
        .replace("(", "").replace(")", "")
        .replace("/", "_").replace("-", "_"))
    allb = r.pivot_table(index="hadm_id",
                         columns="nm",
                         values="v").reset_index()
    allb.to_csv(CACHE, index=False)
    print("cached:", allb.shape)

v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv"
).drop_duplicates("hadm_id")
vit = {"mean_temp": "temp", "mean_hr": "hr",
       "mean_sbp": "sbp", "mean_dbp": "dbp",
       "mean_rr": "rr", "mean_spo2": "spo2"}
keep = ["hadm_id"] + [c for c in vit
                      if c in v7.columns] \
    + [c for c in FLAGS if c in v7.columns] \
    + [c for c in ["nlr"] if c in v7.columns]
mv = v7[keep].rename(columns=vit)

dem = pd.read_csv(
    P2 + "/mimic_pe_mace_cohort.csv",
    usecols=["hadm_id", "age_at_admit",
             "gender"]).drop_duplicates(
    "hadm_id")
dem = dem.rename(
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

if os.path.exists(HIST):
    h = pd.read_csv(HIST)
    h = h.rename(columns={"idx_hadm":
                          "hadm_id"})
    hc = [c for c in ["n_prior", "n_codes"]
          if c in h.columns]
    if hc:
        d = d.merge(h[["hadm_id"] + hc],
                    on="hadm_id", how="left")
        for c in hc:
            d[c] = d[c].fillna(0)
        print("history feats:", hc)

TRANS = [c for c in
         (["mi_" + a if "mi_" + a in d.columns
           else a for a in TA]
          + FLAGS + ["age"])
         if c in d.columns]
LBC = [c for c in d.columns
       if c.startswith("lb_")]
EXTRA = [c for c in
         ["spo2", "sex_f", "nlr",
          "n_prior", "n_codes"]
         if c in d.columns]
ALLF = sorted(set(TRANS + LBC + EXTRA))

BAN = ["30d", "first", "label", "composite",
       "death", "mace", "_y"]
ALLF = [c for c in ALLF
        if not any(b in c.lower()
                   for b in BAN)]
TRANS = [c for c in TRANS
         if not any(b in c.lower()
                    for b in BAN)]
print("")
print("transferable feats:", len(TRANS))
print("ALL feats:", len(ALLF))
print("labs added:", len(LBC),
      " extra:", EXTRA)

SETS = {"transferable": TRANS, "all": ALLF}


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

    zs = np.nan
    eh = "death_30d" if out == \
        "death_30d_inhosp" else out
    zf = DATA + "/p_ehr_harm_" + eh + ".csv"
    if os.path.exists(zf):
        pz = pd.read_csv(zf)
        z = dd[["hadm_id"]].merge(
            pz, on="hadm_id", how="left")
        if z["p_ehr"].notna().all():
            zs = roc_auc_score(
                y, z["p_ehr"].values)

    print("")
    print("=" * 62)
    print("%s  n=%d ev=%d (%.4f)"
          % (out, len(y), int(y.sum()),
             y.mean()))
    print("  zero-shot (transfer)  %.4f" % zs)

    store = {}
    for nm, cols in SETS.items():
        X = dd[cols].values.astype(float)
        for ln, mk in [("lr", mk_lr),
                       ("gb", mk_gb)]:
            a, s, ap = evalcv(X, y, grp, mk)
            store[(nm, ln)] = a
            print("  %-13s %s  AUC %.4f +/-"
                  " %.4f  AP %.4f  (%d feats)"
                  % (nm, ln, a, s, ap,
                     X.shape[1]))
            res.append({
                "outcome": out, "set": nm,
                "learner": ln, "n": len(y),
                "ev": int(y.sum()),
                "nfeat": X.shape[1],
                "auc": a, "sd": s, "ap": ap,
                "zeroshot": zs})

    print("")
    print("  DECOMPOSITION")
    for ln in ["lr", "gb"]:
        tr = store[("transferable", ln)]
        al = store[("all", ln)]
        print("    [%s] transfer cost %+.4f"
              " | feature-restriction cost"
              " %+.4f | total %+.4f"
              % (ln, tr - zs, al - tr,
                 al - zs))

r = pd.DataFrame(res)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string())
print("saved", OUT)
