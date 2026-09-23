import os
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
LX = DATA + "/llm_extract"
OUT = DATA + "/llm_arm_results.csv"

SEEDS = [42, 7, 13]
NBOOT = 2000
POST_H = 24
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]

CTBASE = ["pe_pos", "saddle", "central",
          "lobar", "segmental", "subseg",
          "bilateral", "rv_strain",
          "septal_bow", "reflux",
          "mpa_enlarge", "infarct",
          "effusion", "malignancy",
          "consolid", "atelect", "edema",
          "cardiomeg", "adenopathy",
          "pe_neg", "txt_len", "n_sent"]

# llm field -> regex column it replaces
REPL = {"llm_pe_present": "pe_pos",
        "llm_rv_strain": "rv_strain",
        "llm_pleural_effusion": "effusion",
        "llm_malignancy": "malignancy",
        "llm_consolidation": "consolid",
        "llm_atelectasis": "atelect",
        "llm_pulmonary_oedema": "edema",
        "llm_cardiomegaly": "cardiomeg",
        "llm_lymphadenopathy": "adenopathy",
        "llm_ascites": "cm_ascites",
        "llm_endotracheal_tube": "dv_ett"}

LOC = {"saddle": r"saddle",
       "central": r"central",
       "lobar": r"(?<!sub)(?<!seg)lobar",
       "segmental": r"(?<!sub)segmental",
       "subseg": r"subsegmental"}

lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
fx = pd.read_csv(
    DATA + "/ctpa_comorb_features.csv")
lm = pd.read_csv(LX + "/mimic_llm_raw.csv")
print("llm rows:", len(lm),
      " parsed:", int(lm["parsed"].sum()))

nt = pd.read_csv(DATA + "/ctpa_notes_index.csv")
for c in ["charttime", "admittime"]:
    nt[c] = pd.to_datetime(nt[c],
                           errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm",
                as_index=False).first()
nt["h"] = ((nt["charttime"] - nt["admittime"])
           .dt.total_seconds() / 3600.0)
keep = nt.loc[nt["h"] <= POST_H,
              "idx_hadm"].unique()

for c in lm.columns:
    if c.startswith("llm_") and c != \
            "llm_pe_location":
        lm[c + "_b"] = (
            lm[c].astype(str).str.strip()
            == "yes").astype(int)

loc = lm["llm_pe_location"].astype(
    str).str.lower().fillna("")
for k, pat in LOC.items():
    lm["llm_loc_" + k] = loc.apply(
        lambda s, p=pat: int(
            bool(re.search(p, s))))
lm["llm_loc_any"] = lm[
    ["llm_loc_" + k for k in LOC]].max(axis=1)

LLMB = [c for c in lm.columns
        if c.endswith("_b")]
LLML = ["llm_loc_" + k for k in LOC] \
    + ["llm_loc_any"]
print("llm binary:", len(LLMB),
      " loc:", len(LLML))

CM = sorted([c for c in fx.columns
             if c.startswith("cm_")])
DV = sorted([c for c in fx.columns
             if c.startswith("dv_")])
REGEX_ALL = [c for c in CTBASE + CM + DV
             if c in fx.columns]

d = lab.merge(
    fx[["subject_id", "hadm_id"] + REGEX_ALL],
    on=["subject_id", "hadm_id"],
    how="inner")
d = d[d["hadm_id"].isin(keep)]
d = d.merge(lm[["hadm_id"] + LLMB + LLML],
            on="hadm_id", how="inner")
d = d.reset_index(drop=True)
print("cohort:", len(d))

replaced = set(REPL.values())
HYBRID = [c for c in REGEX_ALL
          if c not in replaced] \
    + [c + "_b" for c in REPL] + LLML
HYBRID = [c for c in HYBRID if c in d.columns]
LLMONLY = LLMB + LLML + ["txt_len", "n_sent"]
LLMONLY = [c for c in LLMONLY
           if c in d.columns]

SETS = {"regex (46)": REGEX_ALL,
        "hybrid": HYBRID,
        "llm only": LLMONLY}
for k, v in SETS.items():
    print("  %-12s %d feats" % (k, len(v)))


def mk():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=10, cv=3, scoring="roc_auc",
            max_iter=5000, n_jobs=-1))])


def oof(X, y, grp, seed):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=seed)
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y, grp):
        m = mk()
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def boot(y, pa, pb, grp):
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(42)
    dd = []
    for _ in range(NBOOT):
        pk = rng.choice(us, len(us),
                        replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        dd.append(roc_auc_score(y[ii], pa[ii])
                  - roc_auc_score(y[ii],
                                  pb[ii]))
    dd = np.array(dd)
    return (float(dd.mean()),
            float(np.percentile(dd, 2.5)),
            float(np.percentile(dd, 97.5)))


rows = []
for out in OUTS:
    dd = d
    if out == "cv_first":
        dd = d[d["death_first"] == 0]
    y = dd[out].values.astype(int)
    grp = dd["subject_id"].values
    if y.sum() < 20:
        continue
    print("")
    print("=" * 60)
    print("%s  n=%d ev=%d (%.4f)"
          % (out, len(y), int(y.sum()),
             y.mean()))

    store = {}
    for nm, cols in SETS.items():
        X = dd[cols].values.astype(float)
        aus, ps = [], None
        for s in SEEDS:
            p = oof(X, y, grp, s)
            aus.append(roc_auc_score(y, p))
            if s == 42:
                ps = p
        store[nm] = ps
        print("  %-12s AUC %.4f +/- %.4f"
              "  AP %.4f  (%d feats)"
              % (nm, float(np.mean(aus)),
                 float(np.std(aus)),
                 average_precision_score(
                     y, ps), X.shape[1]))
        rows.append({
            "outcome": out, "set": nm,
            "nfeat": X.shape[1],
            "auc": float(np.mean(aus)),
            "sd": float(np.std(aus)),
            "ap": average_precision_score(
                y, ps)})

    ref = store["regex (46)"]
    print("")
    for nm in ["hybrid", "llm only"]:
        t = boot(y, store[nm], ref, grp)
        star = " *" if (t[1] > 0
                        or t[2] < 0) else ""
        print("  %-12s vs regex  %+.4f"
              " [%+.4f, %+.4f]%s"
              % (nm, t[0], t[1], t[2], star))
        for r in rows:
            if r["outcome"] == out and \
                    r["set"] == nm:
                r["vs_regex"] = t[0]
                r["lo"] = t[1]
                r["hi"] = t[2]

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string(index=False))
print("")
print("regex CTPA modality reference (script 146):"
      " death 0.7814, composite 0.7378,"
      " in-hosp 0.7593, cv_first 0.5809")
print("saved", OUT)
