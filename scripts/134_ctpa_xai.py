import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

BASE = "."
DATA = BASE + "/fusion_workspace/data"
OUT = DATA + "/ctpa_xai"
os.makedirs(OUT, exist_ok=True)

FEAT = DATA + "/ctpa_comorb_features.csv"
NOTES = DATA + "/ctpa_notes_index.csv"
POST_H = 24
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]

BASEC = ["pe_pos", "saddle", "central",
         "lobar", "segmental", "subseg",
         "bilateral", "rv_strain",
         "septal_bow", "reflux",
         "mpa_enlarge", "infarct",
         "effusion", "malignancy",
         "consolid", "atelect", "edema",
         "cardiomeg", "adenopathy",
         "pe_neg", "txt_len", "n_sent"]

nt = pd.read_csv(NOTES)
for c in ["charttime", "admittime"]:
    nt[c] = pd.to_datetime(nt[c],
                           errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm",
                as_index=False).first()
nt["h"] = ((nt["charttime"] - nt["admittime"])
           .dt.total_seconds() / 3600.0)
tm = nt[["idx_hadm", "h"]].rename(
    columns={"idx_hadm": "hadm_id"})

d = pd.read_csv(FEAT).merge(tm, on="hadm_id")
d = d[d["h"] <= POST_H]
CM = sorted([c for c in d.columns
             if c.startswith("cm_")])
DV = sorted([c for c in d.columns
             if c.startswith("dv_")])
XC = [c for c in BASEC + CM + DV
      if c in d.columns]
print("presentation cohort:", len(d))
print("features:", len(XC))

print("")
print("FEATURE PREVALENCE")
for c in XC:
    if c in ("txt_len", "n_sent"):
        print("  %-18s median %.0f"
              % (c, float(d[c].median())))
    else:
        print("  %-18s %.3f"
              % (c, float(d[c].mean())))

rows = []
for out in OUTS:
    dd = d
    if out == "cv_first":
        dd = d[d["death_first"] == 0]
    y = dd[out].values.astype(int)
    if y.sum() < 20:
        continue
    X = dd[XC].values.astype(float)
    pipe = Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=10, cv=3, scoring="roc_auc",
            max_iter=5000, n_jobs=-1))])
    pipe.fit(X, y)
    Z = pipe.named_steps["sc"].transform(
        pipe.named_steps["im"].transform(X))
    beta = pipe.named_steps["lr"].coef_[0]
    S = Z * beta

    prev = []
    for c in XC:
        if c in ("txt_len", "n_sent"):
            prev.append(np.nan)
        else:
            prev.append(float(dd[c].mean()))

    t = pd.DataFrame({
        "feature": XC,
        "prevalence": prev,
        "beta": beta,
        "shap": np.abs(S).mean(axis=0),
        "signed": S.mean(axis=0)})
    t["block"] = [
        "device" if f.startswith("dv_")
        else "comorbid" if f.startswith("cm_")
        else "pe_finding" for f in t["feature"]]
    t = t.sort_values("shap", ascending=False)
    t.to_csv(OUT + "/ctpa_coef_" + out
             + ".csv", index=False)

    print("")
    print("=" * 60)
    print("%s  n=%d ev=%d (%.4f)"
          % (out, len(y), int(y.sum()),
             y.mean()))
    print("  %-18s %-11s %8s %8s"
          % ("feature", "block", "beta",
             "|SHAP|"))
    for _, r in t.head(12).iterrows():
        print("  %-18s %-11s %+8.3f %8.3f"
              % (r["feature"], r["block"],
                 r["beta"], r["shap"]))

    print("")
    print("  STRONGEST NEGATIVE")
    for _, r in t.sort_values(
            "beta").head(5).iterrows():
        print("  %-18s %-11s %+8.3f"
              % (r["feature"], r["block"],
                 r["beta"]))

    print("")
    print("  BLOCK CONTRIBUTION (sum |SHAP|)")
    bs = t.groupby("block")["shap"].sum()
    tot = float(bs.sum())
    for k, v in bs.sort_values(
            ascending=False).items():
        print("    %-11s %.3f  (%.1f%%)"
              % (k, v, 100.0 * v / tot))

    for _, r in t.iterrows():
        rows.append({"outcome": out,
                     **r.to_dict()})

pd.DataFrame(rows).to_csv(
    OUT + "/ctpa_coef_all.csv", index=False)
print("")
print("saved to", OUT)

