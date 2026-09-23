import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = "."
D = BASE + "/fusion_workspace/data"
OUT = D + "/spesi_ci_ecglinked.csv"
B = 2000

lab = pd.read_csv(D + "/mimic_labels_harmonised.csv")
h = glob.glob(BASE + "/**/mimic_pe_ehr_baseline_v6.csv",
              recursive=True)[0]
v6 = pd.read_csv(h).drop_duplicates(subset=["hadm_id"])

s = (v6["age_at_admit"] > 80).astype(int)
s = s + (v6["mean_hr"] >= 110).astype(int)
s = s + (v6["mean_sbp"] < 100).astype(int)
s = s + (v6["mean_spo2"] < 90).astype(int)
s = s + (v6["cancer"] > 0).astype(int)
hf = (v6["heart_failure"] > 0) | (v6["copd"] > 0)
s = s + hf.astype(int)
sp = pd.DataFrame({"hadm_id": v6["hadm_id"],
                   "spesi6": s})


def auc_from_counts(P, N):
    # P, N are (rows, levels) weighted counts,
    # levels sorted ascending by score
    cn = np.cumsum(N, axis=1) - N
    num = (P * (cn + 0.5 * N)).sum(axis=1)
    den = P.sum(axis=1) * N.sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


rng = np.random.default_rng(42)
rows = []

for o in ["composite_30d", "death_30d",
          "death_30d_inhosp", "cv_first"]:
    d = lab[["subject_id", "hadm_id", o]].dropna()
    if o == "cv_first":
        k = lab.loc[lab["death_first"] == 0, "hadm_id"]
        d = d[d["hadm_id"].isin(k)]
    e = pd.read_csv(D + "/p_ecg_harm_%s.csv" % o)
    d = d[d["hadm_id"].isin(set(e["hadm_id"]))]
    m = d.merge(sp, on="hadm_id", how="inner")

    y = m[o].astype(int).to_numpy()
    sc = m["spesi6"].to_numpy(dtype=int)
    sub, _ = pd.factorize(m["subject_id"])
    nsub = sub.max() + 1

    lv = np.sort(np.unique(sc))
    li = np.searchsorted(lv, sc)
    K = len(lv)

    Ppos = np.zeros((nsub, K))
    Pneg = np.zeros((nsub, K))
    np.add.at(Ppos, (sub[y == 1], li[y == 1]), 1.0)
    np.add.at(Pneg, (sub[y == 0], li[y == 0]), 1.0)

    one = np.ones((1, nsub))
    pt = auc_from_counts(one @ Ppos, one @ Pneg)[0]
    chk = roc_auc_score(y, sc)
    print(o, "vectorised %.6f  sklearn %.6f"
          % (pt, chk), flush=True)

    pick = rng.integers(0, nsub, size=(B, nsub))
    W = np.zeros((B, nsub))
    for i in range(B):
        W[i] = np.bincount(pick[i], minlength=nsub)
    boot = auc_from_counts(W @ Ppos, W @ Pneg)
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    rows.append({
        "outcome": o,
        "n": len(m),
        "events": int(y.sum()),
        "auroc": round(float(pt), 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
    })
    print("  ", rows[-1], flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("saved:", OUT)

