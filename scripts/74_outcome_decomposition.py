import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

FD = (
    "./"
    "fusion_workspace/data/"
)
KEY = ["subject_id", "hadm_id"]
SEED = 42
NBOOT = 2000


def boot_auc(y, pa, pb, g, n=NBOOT):
    rng = np.random.RandomState(SEED)
    g = np.asarray(g)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    o = []
    for _ in range(n):
        s = rng.choice(uq, len(uq),
                       replace=True)
        t = np.concatenate([idx[u] for u in s])
        if len(np.unique(y[t])) < 2:
            continue
        o.append(
            roc_auc_score(y[t], pa[t])
            - roc_auc_score(y[t], pb[t])
        )
    o = np.array(o)
    return o.mean(), np.percentile(
        o, [2.5, 97.5]
    )


def boot_ci(y, p, g, n=NBOOT):
    rng = np.random.RandomState(SEED)
    g = np.asarray(g)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    o = []
    for _ in range(n):
        s = rng.choice(uq, len(uq),
                       replace=True)
        t = np.concatenate([idx[u] for u in s])
        if len(np.unique(y[t])) < 2:
            continue
        o.append(roc_auc_score(y[t], p[t]))
    return np.percentile(o, [2.5, 97.5])


print("Loading time-to-event...")
tm = pd.read_csv(
    os.path.join(FD, "time_to_mace.csv")
)
tm["death_days"] = pd.to_numeric(
    tm["death_days"], errors="coerce"
)
tm["readmit_days"] = pd.to_numeric(
    tm["readmit_days"], errors="coerce"
)
dd = tm["death_days"]
rd = tm["readmit_days"]
tm["death_first"] = (
    dd.notna() & (rd.isna() | (dd < rd))
).astype(int)
tm["readmit_first"] = (
    rd.notna() & (dd.isna() | (rd <= dd))
).astype(int)
print("  death_first  :", tm["death_first"].sum())
print("  readmit_first:",
      tm["readmit_first"].sum())
print("  composite    :",
      int(tm["event"].sum()))

tm = tm[KEY + ["event", "death_first",
               "readmit_first"]]


def run(name, d, arms, fusions):
    m = d.merge(tm, on=KEY, how="inner")
    print("\n" + "#" * 64)
    print(name, " n=%d" % len(m))
    print("#" * 64)

    outcomes = [
        ("COMPOSITE (orig)",
         m["event"].values,
         np.ones(len(m), dtype=bool)),
        ("A: all-cause death",
         m["death_first"].values,
         np.ones(len(m), dtype=bool)),
        ("B: CV readmission",
         m["readmit_first"].values,
         (m["death_first"] == 0).values),
    ]

    for oname, yv, keep in outcomes:
        y = yv[keep].astype(int)
        g = m["subject_id"].values[keep]
        print("\n" + "=" * 60)
        print("%s   n=%d events=%d (%.2f%%)"
              % (oname, len(y), int(y.sum()),
                 100 * y.mean()))
        print("=" * 60)
        if y.sum() < 20:
            print("  too few events, skipping")
            continue

        allm = {}
        for k, c in arms.items():
            allm[k] = m[c].values[keep]
        for k, c in fusions.items():
            allm[k] = m[c].values[keep]

        print("%-10s %-8s %-20s %-8s"
              % ("model", "AUC", "95% CI", "AUPRC"))
        for k, v in allm.items():
            a = roc_auc_score(y, v)
            ci = boot_ci(y, v, g)
            ap = average_precision_score(y, v)
            print("%-10s %.4f   "
                  "[%.4f, %.4f]     %.4f"
                  % (k, a, ci[0], ci[1], ap))

        uni = [k for k in arms]
        best = max(
            uni,
            key=lambda k: roc_auc_score(
                y, allm[k]
            )
        )
        print("\n  best unimodal: %s" % best)
        for k in fusions:
            mm, ci = boot_auc(
                y, allm[k], allm[best], g
            )
            s = "*" if ci[0] > 0 else " "
            print("  %-8s vs %-6s %+0.4f "
                  "[%+0.4f, %+0.4f] %s"
                  % (k, best, mm,
                     ci[0], ci[1], s))


d2 = pd.read_csv(
    os.path.join(FD, "rh2_twoway_table.csv")
)
run("TWO-WAY COHORT (EHR+ECG)", d2,
    {"EHR": "r_ehr", "ECG": "r_ecg"},
    {"MEAN2": "MEAN2"})

d3 = pd.read_csv(
    os.path.join(FD, "ro4_fusion_table.csv")
)
d3["r_spesi"] = (
    d3["spesi"].fillna(0).rank(pct=True)
)
d3["MEAN4"] = (
    d3["r_ehr"] + d3["r_ecg"]
    + d3["r_cxr"] + d3["r_spesi"]
) / 4
run("THREE-WAY COHORT (EHR+ECG+CXR)", d3,
    {"EHR": "r_ehr", "ECG": "r_ecg",
     "CXR": "r_cxr", "sPESI": "r_spesi"},
    {"MEAN2": "MEAN2", "MEAN3": "MEAN3",
     "MEAN4": "MEAN4"})

print("\nDone.")

