import glob
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
D = BASE + "/fusion_workspace/data"
OUT = D + "/cohort_convention_check.csv"

lab = pd.read_csv(D + "/mimic_labels_harmonised.csv")
print("labels:", lab.shape)

hits = glob.glob(BASE + "/**/mimic_pe_ehr_baseline_v6.csv",
                 recursive=True)
print("v6 found:", hits[:3])
v6 = pd.read_csv(hits[0]).drop_duplicates(subset=["hadm_id"])
print("v6:", v6.shape)

need = ["age_at_admit", "mean_hr", "mean_sbp",
        "mean_spo2", "cancer", "heart_failure", "copd"]
miss = [c for c in need if c not in v6.columns]
if miss:
    raise SystemExit("missing cols: %s" % miss)

s = (v6["age_at_admit"] > 80).astype(int)
s = s + (v6["mean_hr"] >= 110).astype(int)
s = s + (v6["mean_sbp"] < 100).astype(int)
s = s + (v6["mean_spo2"] < 90).astype(int)
s = s + (v6["cancer"] > 0).astype(int)
hf = (v6["heart_failure"] > 0) | (v6["copd"] > 0)
s = s + hf.astype(int)
sp = pd.DataFrame({"hadm_id": v6["hadm_id"], "spesi6": s})

OUTS = ["composite_30d", "death_30d",
        "death_30d_inhosp", "cv_first"]


def ev(sub, ycol, pcol):
    y = sub[ycol].astype(int)
    p = sub[pcol]
    return {
        "n": len(sub),
        "events": int(y.sum()),
        "auroc": round(roc_auc_score(y, p), 4),
        "ap": round(average_precision_score(y, p), 4),
    }


rows = []
for o in OUTS:
    d = lab[["subject_id", "hadm_id", o]].dropna()
    if o == "cv_first" and "death_first" in lab.columns:
        k = lab.loc[lab["death_first"] == 0, "hadm_id"]
        d = d[d["hadm_id"].isin(k)]

    fe = D + "/p_ecg_harm_%s.csv" % o
    link = None
    if os.path.exists(fe):
        link = set(pd.read_csv(fe)["hadm_id"])

    for model in ["spesi6", "p_ehr"]:
        if model == "spesi6":
            m = d.merge(sp, on="hadm_id", how="inner")
        else:
            fh = D + "/p_ehr_harm_%s.csv" % o
            if not os.path.exists(fh):
                continue
            h = pd.read_csv(fh)[["hadm_id", "p_ehr"]]
            m = d.merge(h, on="hadm_id", how="inner")
        pairs = [("full", m)]
        if link is not None:
            pairs.append(
                ("ecg-linked", m[m["hadm_id"].isin(link)]))
        for tag, sub in pairs:
            r = ev(sub, o, model)
            r["outcome"] = o
            r["model"] = model
            r["cohort"] = tag
            rows.append(r)

cols = ["outcome", "model", "cohort", "n",
        "events", "auroc", "ap"]
res = pd.DataFrame(rows)[cols]
print()
print(res.to_string(index=False))
res.to_csv(OUT, index=False)
print()
print("saved:", OUT)

