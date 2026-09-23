import os
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
OUT = DATA + "/burden_arm_results.csv"

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

YN = ["space_occupying", "any_effusion",
      "airspace_disease", "volume_loss",
      "congestion", "chronic_lung",
      "nodal_enlargement", "organ_chronic",
      "cachexia_frailty", "support_device",
      "pe_present"]
NUM = ["pe_extent", "acute_severity",
       "chronic_burden"]

lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
fx = pd.read_csv(
    DATA + "/ctpa_comorb_features.csv")
bu = pd.read_csv(LX + "/mimic_burden_raw.csv")
print("burden rows:", len(bu))

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

BU = []
for f in YN:
    c = "bu_" + f
    bu[c + "_b"] = (
        bu[c].astype(str).str.strip()
        == "yes").astype(int)
    BU.append(c + "_b")
for f in NUM:
    BU.append("bu_" + f)
drop = [c for c in BU
        if bu[c].nunique(dropna=True) < 2
        or (c.endswith("_b")
            and bu[c].mean() < 0.005)]
if drop:
    print("dropping near-constant:", drop)
    BU = [c for c in BU if c not in drop]
print("burden features:", len(BU))

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
d = d.merge(bu[["hadm_id"] + BU],
            on="hadm_id", how="inner")
d = d.reset_index(drop=True)
print("cohort:", len(d))

SETS = {"regex (46)": REGEX_ALL,
        "burden only": BU,
        "regex+burden": REGEX_ALL + BU}
for k, v in SETS.items():
    print("  %-14s %d feats" % (k, len(v)))


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
        print("  %-14s AUC %.4f +/- %.4f"
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
    for nm in ["burden only", "regex+burden"]:
        t = boot(y, store[nm], ref, grp)
        star = " *" if (t[1] > 0
                        or t[2] < 0) else ""
        print("  %-14s vs regex  %+.4f"
              " [%+.4f, %+.4f]%s"
              % (nm, t[0], t[1], t[2], star))
        for r in rows:
            if r["outcome"] == out and \
                    r["set"] == nm:
                r["vs_regex"] = t[0]
                r["lo"] = t[1]
                r["hi"] = t[2]

    if out == "death_30d":
        X = dd[SETS["regex+burden"]].values \
            .astype(float)
        m = mk()
        m.fit(X, y)
        Z = m.named_steps["sc"].transform(
            m.named_steps["im"].transform(X))
        b = m.named_steps["lr"].coef_[0]
        t2 = pd.DataFrame({
            "feature": SETS["regex+burden"],
            "beta": b,
            "shap": np.abs(Z * b).mean(axis=0)})
        t2["src"] = ["burden"
                     if f.startswith("bu_")
                     else "regex"
                     for f in t2["feature"]]
        t2 = t2.sort_values("shap",
                            ascending=False)
        print("")
        print("  TOP 15 IN COMBINED MODEL")
        for _, r in t2.head(15).iterrows():
            print("    %-22s %-7s %+7.3f"
                  " %7.3f"
                  % (r["feature"][:22],
                     r["src"], r["beta"],
                     r["shap"]))
        g2 = t2.groupby("src")["shap"].sum()
        tot = float(g2.sum())
        print("")
        for k in g2.index:
            print("    %-8s %.3f (%.1f%%)"
                  % (k, g2[k],
                     100 * g2[k] / tot))
        t2.to_csv(LX + "/burden_coef_death.csv",
                  index=False)

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string(index=False))
print("")
print("REF script 149 (first LLM prompt): hybrid -0.0141,"
      " llm-only -0.0250 on death")
print("saved", OUT)
