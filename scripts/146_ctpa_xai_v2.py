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

BASE = "."
DATA = BASE + "/fusion_workspace/data"
OUT = DATA + "/ctpa_xai_v2"
os.makedirs(OUT, exist_ok=True)

FEAT = DATA + "/ctpa_comorb_features.csv"
NOTES = DATA + "/ctpa_notes_index.csv"
POST_H = 24
SEED = 42
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]

PE_DESC = ["pe_pos", "pe_neg", "saddle",
           "central", "lobar", "segmental",
           "subseg", "bilateral", "rv_strain",
           "septal_bow", "reflux",
           "mpa_enlarge", "infarct"]
INCID = ["effusion", "malignancy", "consolid",
         "atelect", "edema", "cardiomeg",
         "adenopathy"]
META = ["txt_len", "n_sent"]


def block_of(f):
    if f.startswith("dv_"):
        return "device"
    if f.startswith("cm_"):
        return "comorbid"
    if f in PE_DESC:
        return "pe_descriptor"
    if f in INCID:
        return "incidental"
    if f in META:
        return "meta"
    return "other"


nt = pd.read_csv(NOTES)
for c in ["charttime", "admittime"]:
    nt[c] = pd.to_datetime(nt[c],
                           errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt1 = nt.groupby("idx_hadm",
                 as_index=False).first()
nt1["h"] = ((nt1["charttime"]
             - nt1["admittime"])
            .dt.total_seconds() / 3600.0)
keep = nt1.loc[nt1["h"] <= POST_H,
               "idx_hadm"].unique()

d = pd.read_csv(FEAT)
d = d[d["hadm_id"].isin(keep)].reset_index(
    drop=True)
XC = [c for c in d.columns
      if block_of(c) != "other"]
print("presentation cohort:", len(d))
print("features:", len(XC))
bl = pd.Series([block_of(c) for c in XC])
print("")
print("BLOCK SIZES")
print(bl.value_counts().to_string())

# ---- check whether `central` also matches central lines ----
print("")
print("=" * 62)
print("`central` CONTAMINATION CHECK")
print("  prevalence in this file: %.3f"
      % float(d["central"].mean()))
print("  (impression-only INSPECT/MIMIC value"
      " in 113-114 was ~0.135-0.149)")
dev = [c for c in d.columns
       if c.startswith("dv_")]
print("")
print("  correlation of `central` with"
      " device flags:")
for c in dev:
    r = float(d["central"].corr(d[c]))
    fl = "  <--" if abs(r) > 0.10 else ""
    print("    %-14s %+0.3f%s" % (c, r, fl))
ov = d.loc[d["dv_cvc"] == 1, "central"].mean()
nv = d.loc[d["dv_cvc"] == 0, "central"].mean()
print("")
print("  central prevalence | cvc=1 %.3f"
      "  | cvc=0 %.3f  (diff %+.3f)"
      % (ov, nv, ov - nv))

# recompute central from raw text if possible
try:
    tx = nt1[["idx_hadm", "text"]].rename(
        columns={"idx_hadm": "hadm_id"})
    tx = tx[tx["hadm_id"].isin(keep)]
    pat_old = re.compile(
        r"\bcentral\b|main pulmonary")
    pat_new = re.compile(
        r"\bcentral\b(?!\s*(?:line|venous"
        r"|cath|access))|main pulmonary")
    lo = tx["text"].astype(str).str.lower()
    o = lo.apply(
        lambda s: int(bool(pat_old.search(s))))
    n2 = lo.apply(
        lambda s: int(bool(pat_new.search(s))))
    print("")
    print("  raw-text check on %d notes:"
          % len(tx))
    print("    old regex prevalence %.3f"
          % float(o.mean()))
    print("    fixed regex prevalence %.3f"
          % float(n2.mean()))
    print("    notes changed: %d (%.1f%%)"
          % (int((o != n2).sum()),
             100.0 * float((o != n2).mean())))
except Exception as e:
    print("  raw-text check skipped:", e)


def mk():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=10, cv=3, scoring="roc_auc",
            max_iter=5000, n_jobs=-1))])


def cvauc(X, y, grp):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y, grp):
        m = mk()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


rows = []
for out in OUTS:
    dd = d
    if out == "cv_first":
        dd = d[d["death_first"] == 0]
    y = dd[out].values.astype(int)
    grp = dd["subject_id"].values
    if y.sum() < 20:
        continue
    X = dd[XC].values.astype(float)

    pipe = mk()
    pipe.fit(X, y)
    Z = pipe.named_steps["sc"].transform(
        pipe.named_steps["im"].transform(X))
    beta = pipe.named_steps["lr"].coef_[0]
    S = Z * beta

    t = pd.DataFrame({
        "feature": XC,
        "block": [block_of(c) for c in XC],
        "prevalence": [
            float(dd[c].mean())
            if c not in META else np.nan
            for c in XC],
        "beta": beta,
        "shap": np.abs(S).mean(axis=0)})
    t = t.sort_values("shap", ascending=False)
    t.to_csv(OUT + "/ctpa_coef_" + out
             + ".csv", index=False)

    print("")
    print("=" * 62)
    print("%s  n=%d ev=%d (%.4f)"
          % (out, len(y), int(y.sum()),
             y.mean()))
    print("  %-18s %-14s %8s %8s"
          % ("feature", "block", "beta",
             "|SHAP|"))
    for _, r in t.head(14).iterrows():
        print("  %-18s %-14s %+8.3f %8.3f"
              % (r["feature"], r["block"],
                 r["beta"], r["shap"]))

    print("")
    print("  BLOCK CONTRIBUTION"
          " (sum |SHAP|, signed mean beta)")
    g = t.groupby("block").agg(
        shap=("shap", "sum"),
        beta=("beta", "mean"),
        n=("feature", "size"))
    tot = float(g["shap"].sum())
    for k in g.sort_values(
            "shap", ascending=False).index:
        print("    %-14s n=%2d  %.3f"
              " (%.1f%%)  mean beta %+.3f"
              % (k, int(g.loc[k, "n"]),
                 g.loc[k, "shap"],
                 100.0 * g.loc[k, "shap"]
                 / tot, g.loc[k, "beta"]))

    a_full = cvauc(X, y, grp)
    nc = [c for c in XC if c != "central"]
    a_noc = cvauc(
        dd[nc].values.astype(float), y, grp)
    npe = [c for c in XC
           if block_of(c) != "pe_descriptor"]
    a_nope = cvauc(
        dd[npe].values.astype(float), y, grp)
    print("")
    print("  CV AUC full            %.4f"
          % a_full)
    print("  CV AUC without central %.4f"
          "  (%+.4f)"
          % (a_noc, a_noc - a_full))
    print("  CV AUC without PE-desc %.4f"
          "  (%+.4f)"
          % (a_nope, a_nope - a_full))

    for _, r in t.iterrows():
        rows.append({
            "outcome": out,
            "auc_full": a_full,
            "auc_no_central": a_noc,
            "auc_no_pedesc": a_nope,
            **r.to_dict()})

pd.DataFrame(rows).to_csv(
    OUT + "/ctpa_coef_all.csv", index=False)
print("")
print("saved to", OUT)
