import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
FEAT = DATA + "/ctpa_comorb_features.csv"
NOTES = DATA + "/ctpa_notes_index.csv"
OUT = DATA + "/ctpa_window_results.csv"

SEEDS = [42, 7, 13]
POST_H = 24
OUTS = ["cv_first", "composite_30d",
        "death_30d", "death_30d_inhosp"]

BASEC = [
    "pe_pos", "saddle", "central", "lobar",
    "segmental", "subseg", "bilateral",
    "rv_strain", "septal_bow", "reflux",
    "mpa_enlarge", "infarct", "effusion",
    "malignancy", "consolid", "atelect",
    "edema", "cardiomeg", "adenopathy",
    "pe_neg", "txt_len", "n_sent"]

BANNED = ["days", "_30d", "first", "label",
          "composite", "death", "cv_"]

nt = pd.read_csv(NOTES)
for c in ["charttime", "admittime"]:
    nt[c] = pd.to_datetime(nt[c],
                           errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm", as_index=False).first()
nt["h"] = ((nt["charttime"] - nt["admittime"])
           .dt.total_seconds() / 3600.0)
tm = nt[["idx_hadm", "h"]].rename(
    columns={"idx_hadm": "hadm_id"})

d = pd.read_csv(FEAT).merge(tm, on="hadm_id")
print("all admissions:", len(d))

BC = sorted([c for c in d.columns
             if c.startswith("cm_")])
DV = sorted([c for c in d.columns
             if c.startswith("dv_")])

missing = [c for c in BASEC
           if c not in d.columns]
if missing:
    raise SystemExit("missing: %s" % missing)

SETS = {"base": BASEC,
        "base_cm": BASEC + BC,
        "base_cm_dv": BASEC + BC + DV}

for nm, cols in SETS.items():
    bad = [c for c in cols
           if any(b in c.lower()
                  for b in BANNED)]
    if bad:
        raise SystemExit(
            "LEAK in %s: %s" % (nm, bad))
    print("%-11s %d feats" % (nm, len(cols)))
print("base list:", BASEC)


def make_model():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=10, cv=3, scoring="roc_auc",
            max_iter=5000, n_jobs=-1))])


res = []
for win in ["full", "pres"]:
    dw = d if win == "full" else \
        d[d["h"] <= POST_H]
    print("")
    print("#" * 56)
    print("WINDOW:", win, " n=", len(dw))
    for out in OUTS:
        dd = dw
        if out == "cv_first":
            dd = dw[dw["death_first"] == 0]
        y = dd[out].values.astype(int)
        grp = dd["subject_id"].values
        if y.sum() < 20:
            continue
        print("")
        print(out, "n=%d ev=%d (%.4f)"
              % (len(y), int(y.sum()), y.mean()))
        for nm, XC in SETS.items():
            X = dd[XC].values.astype(float)
            aucs = []
            for s in SEEDS:
                cv = StratifiedGroupKFold(
                    n_splits=5, shuffle=True,
                    random_state=s)
                oof = np.zeros(len(y))
                for tr, te in cv.split(X, y, grp):
                    md = make_model()
                    md.fit(X[tr], y[tr])
                    oof[te] = md.predict_proba(
                        X[te])[:, 1]
                aucs.append(
                    roc_auc_score(y, oof))
                if s == 42:
                    o42 = oof.copy()
                    ap = average_precision_score(
                        y, oof)
            mu = float(np.mean(aucs))
            if mu > 0.95:
                print("  !! %s AUC %.4f "
                      "-- CHECK FOR LEAK"
                      % (nm, mu))
            print("  %-11s AUC %.4f +/- %.4f "
                  "AP %.4f"
                  % (nm, mu, float(np.std(aucs)),
                     ap))
            res.append({"window": win,
                        "outcome": out,
                        "set": nm, "n": len(y),
                        "ev": int(y.sum()),
                        "auc": mu, "ap": ap})
            if win == "pres":
                po = pd.DataFrame({
                    "subject_id":
                        dd["subject_id"].values,
                    "hadm_id": dd["hadm_id"].values,
                    "p_ctpa": o42})
                po.to_csv(DATA + "/p_ctpa_pres_"
                          + nm + "_" + out
                          + ".csv", index=False)

rdf = pd.DataFrame(res)
rdf.to_csv(OUT, index=False)
print("")
print(rdf.round(4).to_string())

print("")
print("BLOCK EFFECTS BY WINDOW")
for out in OUTS:
    for win in ["full", "pres"]:
        s = rdf[(rdf["outcome"] == out)
                & (rdf["window"] == win)]
        if len(s) < 3:
            continue

        def g(k):
            return float(
                s[s["set"] == k]["auc"].iloc[0])
        print("%-18s %-5s comorb %+.4f"
              "  device %+.4f"
              % (out, win,
                 g("base_cm") - g("base"),
                 g("base_cm_dv") - g("base_cm")))
print("saved", OUT)
