import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
OUT = DATA + "/subgroup_spesi.csv"

NBOOT = 1000
SEED = 42
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]

lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
dem = pd.read_csv(
    P2 + "/mimic_pe_mace_cohort.csv",
    usecols=["hadm_id", "age_at_admit",
             "gender"]).drop_duplicates(
    "hadm_id")

v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv"
).drop_duplicates("hadm_id")
sp = v7[["hadm_id", "mean_hr", "mean_sbp",
         "mean_spo2", "cancer", "copd",
         "heart_failure"]].merge(
    dem[["hadm_id", "age_at_admit"]],
    on="hadm_id", how="left")
t = np.zeros(len(sp))
t += (sp["mean_hr"] >= 110).fillna(
    False).astype(int).values
t += (sp["mean_sbp"] < 100).fillna(
    False).astype(int).values
t += (sp["mean_spo2"] < 90).fillna(
    False).astype(int).values
t += sp["cancer"].fillna(0).astype(int).values
cp = (sp["heart_failure"].fillna(0).values
      + sp["copd"].fillna(0).values)
t += (cp > 0).astype(int)
t += (sp["age_at_admit"] > 80).fillna(
    False).astype(int).values
sp["spesi"] = t
sp = sp[["hadm_id", "spesi"]]
print("sPESI mean %.3f" % sp["spesi"].mean())


def bci(y, p):
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
    d = lab.merge(pw[["hadm_id", pc[0]]],
                  on="hadm_id")
    d = d.merge(sp, on="hadm_id", how="left")
    d = d.merge(dem, on="hadm_id", how="left")
    if out == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)
    d["band"] = pd.cut(
        d["age_at_admit"],
        [0, 50, 65, 80, 200],
        labels=["<50", "50-64", "65-79",
                "80+"])
    d["sex"] = d["gender"].astype(
        str).str.upper().str[0]

    y = d[out].values.astype(int)
    pm = d[pc[0]].values
    ps = d["spesi"].fillna(
        d["spesi"].median()).values

    print("")
    print("=" * 66)
    print("%s  n=%d ev=%d" % (out, len(y),
                              int(y.sum())))
    am = roc_auc_score(y, pm)
    asp = roc_auc_score(y, ps)
    print("  OVERALL  model %.4f  sPESI"
          " %.4f  gap %+.4f"
          % (am, asp, am - asp))

    for var in ["band", "sex"]:
        print("")
        print("  by %s:" % var)
        print("  %-8s %5s %4s %8s %8s %8s"
              % ("level", "n", "ev", "model",
                 "sPESI", "gap"))
        lv = (d[var].cat.categories
              if var == "band"
              else sorted(d[var].dropna()
                          .unique()))
        for l in lv:
            m = (d[var] == l).values
            ys = y[m]
            if len(ys) < 60 or ys.sum() < 10:
                continue
            a1 = roc_auc_score(ys, pm[m])
            a2 = roc_auc_score(ys, ps[m])
            lo1, hi1 = bci(ys, pm[m])
            lo2, hi2 = bci(ys, ps[m])
            print("  %-8s %5d %4d %8.4f"
                  " %8.4f %+8.4f"
                  % (l, len(ys), int(ys.sum()),
                     a1, a2, a1 - a2))
            print("           %28s %s"
                  % ("[%.3f,%.3f]" % (lo1, hi1),
                     "[%.3f,%.3f]" % (lo2, hi2)))
            rows.append({
                "outcome": out, "var": var,
                "level": str(l), "n": len(ys),
                "ev": int(ys.sum()),
                "prev": float(ys.mean()),
                "model_auc": a1,
                "m_lo": lo1, "m_hi": hi1,
                "spesi_auc": a2,
                "s_lo": lo2, "s_hi": hi2,
                "gap": a1 - a2,
                "model_ap":
                    average_precision_score(
                        ys, pm[m]),
                "spesi_ap":
                    average_precision_score(
                        ys, ps[m])})

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string(index=False))
print("")
print("Cahan et al. 2023 Table 2:"
      " multimodal 0.96 full / 0.95 age70-95;"
      " sPESI 0.77 full / 0.60 age70-95")
print("saved", OUT)

