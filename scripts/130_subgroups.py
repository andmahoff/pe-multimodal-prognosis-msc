import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
OUT = DATA + "/subgroup_results.csv"

NBOOT = 1000
SEED = 42
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]

lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")

dem = None
for f in ["mimic_pe_mace_cohort.csv",
          "mimic_pe_ehr_baseline_v6.csv",
          "mimic_pe_ehr_baseline.csv"]:
    p = os.path.join(P2, f)
    if not os.path.exists(p):
        continue
    h = pd.read_csv(p, nrows=0)
    cols = ["hadm_id"]
    for c in ["age_at_admit", "gender",
              "sex", "cancer",
              "heart_failure"]:
        if c in h.columns:
            cols.append(c)
    if len(cols) > 1:
        dem = pd.read_csv(p, usecols=cols)
        dem = dem.drop_duplicates("hadm_id")
        print("demographics from", f, cols)
        break

d = lab.merge(dem, on="hadm_id", how="left")
print("cohort:", len(d))
if "age_at_admit" in d.columns:
    d["age_band"] = pd.cut(
        d["age_at_admit"],
        [0, 50, 65, 80, 200],
        labels=["<50", "50-64", "65-79",
                "80+"])
gcol = None
for c in ["gender", "sex"]:
    if c in d.columns:
        gcol = c
        break
if gcol:
    print("gender values:",
          d[gcol].value_counts().to_dict())


def boot_auc(y, p):
    rng = np.random.default_rng(SEED)
    b = []
    n = len(y)
    for _ in range(NBOOT):
        ii = rng.integers(0, n, n)
        if len(np.unique(y[ii])) < 2:
            continue
        b.append(roc_auc_score(y[ii], p[ii]))
    if not b:
        return (np.nan, np.nan)
    return (float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)))


rows = []
for out in OUTS:
    f = DATA + "/p_wmean2_" + out + ".csv"
    if not os.path.exists(f):
        print("missing", f)
        continue
    pw = pd.read_csv(f)
    pc = [c for c in pw.columns
          if "wmean2" in c.lower()
          and "cal" not in c.lower()]
    if not pc:
        continue
    dd = d.merge(pw[["hadm_id", pc[0]]],
                 on="hadm_id", how="inner")
    if out == "cv_first":
        dd = dd[dd["death_first"] == 0]
    y = dd[out].values.astype(int)
    p = dd[pc[0]].values

    print("")
    print("=" * 58)
    print("%s  n=%d ev=%d  overall AUC %.4f"
          % (out, len(y), int(y.sum()),
             roc_auc_score(y, p)))

    groups = {}
    if "age_band" in dd.columns:
        groups["age"] = dd["age_band"].astype(str)
    if gcol:
        groups["sex"] = dd[gcol].astype(str)
    if "cancer" in dd.columns:
        groups["cancer"] = dd["cancer"].astype(
            "Int64").astype(str)
    if "heart_failure" in dd.columns:
        groups["hf"] = dd["heart_failure"].astype(
            "Int64").astype(str)

    for gn, gv in groups.items():
        print("")
        print("  by %s:" % gn)
        for lv in sorted(gv.dropna().unique()):
            m = (gv == lv).values
            ys = y[m]
            ps = p[m]
            if len(ys) < 60 or ys.sum() < 10:
                print("    %-8s n=%4d ev=%3d"
                      "  (too few)"
                      % (lv, len(ys),
                         int(ys.sum())))
                continue
            a = roc_auc_score(ys, ps)
            lo, hi = boot_auc(ys, ps)
            ap = average_precision_score(ys, ps)
            print("    %-8s n=%4d ev=%3d"
                  "  AUC %.4f [%.4f,%.4f]"
                  "  prev %.3f  AP %.3f"
                  % (lv, len(ys), int(ys.sum()),
                     a, lo, hi, ys.mean(), ap))
            rows.append({
                "outcome": out, "var": gn,
                "level": lv, "n": len(ys),
                "ev": int(ys.sum()),
                "prev": float(ys.mean()),
                "auc": a, "lo": lo, "hi": hi,
                "ap": ap})

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string())
print("saved", OUT)

