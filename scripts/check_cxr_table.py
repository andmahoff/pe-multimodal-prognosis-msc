import glob
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
D = BASE + "/fusion_workspace/data"
LAB = D + "/mimic_labels_harmonised.csv"
OUT = D + "/cxr_table_check.csv"

lab = pd.read_csv(LAB)
print("labels:", lab.shape)
print()

KEYS = ["subject_id", "hadm_id"]

JOBS = {
    "composite_30d": ["composite_30d"],
    "death_30d": ["death_30d", "death_30d_inhosp"],
    "cv_first": ["cv_first"],
}


def score(d, ycol):
    d = d.dropna(subset=["p_cxr", ycol])
    y = d[ycol].astype(int)
    p = d["p_cxr"]
    return {
        "n": len(d),
        "events": int(y.sum()),
        "prev": round(float(y.mean()), 4),
        "auroc": round(roc_auc_score(y, p), 4),
        "ap": round(average_precision_score(y, p), 4),
    }


rows = []
files = sorted(glob.glob(D + "/p_cxr_harm_*.csv"))
for f in files:
    name = os.path.basename(f)
    stem = name.replace("p_cxr_harm_", "")
    stem = stem.replace("_fixed.csv", "")
    stem = stem.replace(".csv", "")
    if stem not in JOBS:
        print("SKIP unknown stem:", name)
        continue
    pred = pd.read_csv(f)
    m = pred.merge(lab, on=KEYS, how="inner")
    print(name, len(pred), "-> merged", len(m))
    for ycol in JOBS[stem]:
        if ycol not in m.columns:
            print("   missing label:", ycol)
            continue
        d = m
        note = ""
        if ycol == "cv_first":
            if "death_first" in d.columns:
                b = len(d)
                d = d[d["death_first"] == 0]
                note = "death_first==0 %d->%d" % (b, len(d))
        r = score(d, ycol)
        r["file"] = name
        r["fixed"] = "_fixed" in name
        r["label"] = ycol
        r["note"] = note
        rows.append(r)

print()
cols = ["file", "fixed", "label", "n", "events",
        "prev", "auroc", "ap", "note"]
res = pd.DataFrame(rows)[cols]
print(res.to_string(index=False))
res.to_csv(OUT, index=False)
print()
print("saved:", OUT)

