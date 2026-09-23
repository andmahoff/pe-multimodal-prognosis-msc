import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = "./fusion_workspace/data"
lab = pd.read_csv(D + "/mimic_labels_harmonised.csv")
h = glob.glob("./**/"
              "mimic_pe_ehr_baseline_v6.csv",
              recursive=True)[0]
v6 = pd.read_csv(h).drop_duplicates(subset=["hadm_id"])
s = (v6["age_at_admit"] > 80).astype(int)
s += (v6["mean_hr"] >= 110).astype(int)
s += (v6["mean_sbp"] < 100).astype(int)
s += (v6["mean_spo2"] < 90).astype(int)
s += (v6["cancer"] > 0).astype(int)
s += ((v6["heart_failure"] > 0)
      | (v6["copd"] > 0)).astype(int)
sp = pd.DataFrame({"hadm_id": v6["hadm_id"],
                   "spesi6": s})

JOBS = [
    ("p_wmean2_%s.csv", "p_wmean2", "composite_30d"),
    ("p_wmean2_%s.csv", "p_wmean2", "death_30d"),
    ("p_wmean2_%s.csv", "p_wmean2", "death_30d_inhosp"),
    ("p_wmean2_%s.csv", "p_wmean2", "cv_first"),
    ("p_wmean3_ctpa_%s.csv", "p_wmean3", "composite_30d"),
    ("p_wmean3_ctpa_%s.csv", "p_wmean3", "death_30d"),
]
rng = np.random.default_rng(42)
out = []

for pat, col, o in JOBS:
    f = D + "/" + (pat % o)
    p = pd.read_csv(f)
    if col not in p.columns:
        print("skip", f, list(p.columns))
        continue
    p = p[["hadm_id", col]]
    lb = lab[["hadm_id", "subject_id", o]]
    m = p.merge(lb, on="hadm_id", how="inner")
    m = m.merge(sp, on="hadm_id", how="inner")
    m = m.dropna(subset=[col, o])

    y = m[o].astype(int).to_numpy()
    a = m[col].to_numpy(dtype=float)
    b = m["spesi6"].to_numpy(dtype=float)
    sub, _ = pd.factorize(m["subject_id"])
    idx = [np.where(sub == k)[0]
           for k in range(sub.max() + 1)]

    d0 = roc_auc_score(y, a) - roc_auc_score(y, b)
    boot = []
    for _ in range(2000):
        pk = rng.integers(0, len(idx), len(idx))
        ii = np.concatenate([idx[j] for j in pk])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        boot.append(roc_auc_score(yy, a[ii])
                    - roc_auc_score(yy, b[ii]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("%-18s %-9s n=%5d  fused %.4f  sPESI %.4f"
          "  margin %+.4f (%+.4f to %+.4f)"
          % (o, col, len(m),
             roc_auc_score(y, a), roc_auc_score(y, b),
             d0, lo, hi), flush=True)
    out.append({"outcome": o, "model": col, "n": len(m),
                "margin": round(d0, 4),
                "lo": round(lo, 4), "hi": round(hi, 4)})

pd.DataFrame(out).to_csv(D + "/spesi_margins.csv",
                         index=False)
print("saved:", D + "/spesi_margins.csv")

