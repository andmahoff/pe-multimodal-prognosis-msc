import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

D = "./fusion_workspace/data"
B = 2000
OUT = D + "/unimodal_ci.csv"

lab = pd.read_csv(D + "/mimic_labels_harmonised.csv")
OUTS = ["composite_30d", "death_30d",
        "death_30d_inhosp", "cv_first"]
ARMS = [("EHR", "p_ehr_harm_%s.csv", "p_ehr"),
        ("ECG", "p_ecg_harm_%s.csv", "p_ecg")]


def pick(pat, o):
    f = D + "/" + (pat % o)
    if os.path.exists(f):
        return f, o
    alt = D + "/" + (pat % "death_30d")
    if o == "death_30d_inhosp" and os.path.exists(alt):
        return alt, o
    return None, o


rng = np.random.default_rng(42)
rows = []

for arm, pat, col in ARMS:
    for o in OUTS:
        f, _ = pick(pat, o)
        if f is None:
            print("MISSING", arm, o)
            continue
        p = pd.read_csv(f)[["hadm_id", col]]
        lb = lab[["hadm_id", "subject_id", o]]
        if o == "cv_first":
            keep = lab.loc[lab["death_first"] == 0,
                           "hadm_id"]
            lb = lb[lb["hadm_id"].isin(keep)]
        m = p.merge(lb, on="hadm_id", how="inner")
        # restrict every modality to the ECG-linked cohort
        e = D + "/p_ecg_harm_%s.csv" % (
            "death_30d" if o == "death_30d_inhosp" else o)
        link = set(pd.read_csv(e)["hadm_id"])
        m = m[m["hadm_id"].isin(link)]
        m = m.dropna(subset=[col, o])

        y = m[o].astype(int).to_numpy()
        s = m[col].to_numpy(dtype=float)
        sub, _ = pd.factorize(m["subject_id"])
        idx = [np.where(sub == k)[0]
               for k in range(sub.max() + 1)]

        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
        boot = []
        for _ in range(B):
            pk = rng.integers(0, len(idx), len(idx))
            ii = np.concatenate([idx[j] for j in pk])
            yy = y[ii]
            if yy.min() == yy.max():
                continue
            boot.append(roc_auc_score(yy, s[ii]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "arm": arm, "outcome": o, "n": len(m),
            "events": int(y.sum()),
            "auroc": round(float(auc), 4),
            "lo": round(float(lo), 4),
            "hi": round(float(hi), 4),
            "ap": round(float(ap), 4),
            "file": os.path.basename(f),
        })
        print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("saved:", OUT)

