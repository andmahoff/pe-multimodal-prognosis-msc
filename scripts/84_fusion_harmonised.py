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


def rank01(x):
    return pd.Series(x).rank(pct=True).values


def boot(y, pa, pb, g, n=NBOOT):
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


lab = pd.read_csv(
    os.path.join(
        FD, "mimic_labels_harmonised.csv"
    )
)
cx = pd.read_csv(
    os.path.join(FD, "cxr_image_timing.csv")
)
cx = cx[cx["rel"] == "during"]
cxa = cx.groupby(KEY).agg(
    p_cxr=("p_cxr", "mean"),
).reset_index()
print("CXR admissions (index images):",
      len(cxa))

TASKS = [
    ("cv_first", "CV only"),
    ("composite_30d", "composite"),
    ("death_30d", "death (dod)"),
    ("death_30d_inhosp", "death (in-hosp)"),
]

rows = []
for lab_col, nm in TASKS:
    eh_col = lab_col
    if lab_col == "death_30d_inhosp":
        eh_col = "death_30d"
    eh = pd.read_csv(os.path.join(
        FD, "p_ehr_harm_%s.csv" % eh_col
    ))
    ec = pd.read_csv(os.path.join(
        FD, "p_ecg_harm_%s.csv" % lab_col
    ))

    d = lab.merge(eh, on=KEY, how="inner")
    d = d.merge(ec, on=KEY, how="inner")
    if lab_col == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)

    for tag, sub in [
        ("TWO-WAY", d),
        ("THREE-WAY",
         d.merge(cxa, on=KEY, how="inner")),
    ]:
        s = sub.reset_index(drop=True)
        if len(s) == 0:
            continue
        y = s[lab_col].values.astype(int)
        g = s["subject_id"].values
        M = {
            "EHR": rank01(s["p_ehr"]),
            "ECG": rank01(s["p_ecg"]),
        }
        if "p_cxr" in s.columns:
            M["CXR"] = rank01(s["p_cxr"])
        M["MEAN2"] = (
            M["EHR"] + M["ECG"]
        ) / 2
        if "CXR" in M:
            M["MEAN3"] = (
                M["EHR"] + M["ECG"] + M["CXR"]
            ) / 3

        print("\n" + "=" * 62)
        print("%s | %s   n=%d ev=%d (%.4f)"
              % (nm, tag, len(s),
                 int(y.sum()), y.mean()))
        print("=" * 62)
        for k, v in M.items():
            print("  %-7s AUC %.4f  AP %.4f"
                  % (k, roc_auc_score(y, v),
                     average_precision_score(
                         y, v)))
            rows.append({
                "outcome": lab_col,
                "cohort": tag, "model": k,
                "n": len(s),
                "events": int(y.sum()),
                "auc": roc_auc_score(y, v),
                "ap":
                    average_precision_score(
                        y, v),
            })

        uni = [k for k in
               ["EHR", "ECG", "CXR"]
               if k in M]
        best = max(
            uni,
            key=lambda k: roc_auc_score(
                y, M[k])
        )
        print("\n  best unimodal:", best)
        for k in ["MEAN2", "MEAN3"]:
            if k not in M:
                continue
            mm, ci = boot(y, M[k], M[best], g)
            star = "*" if ci[0] > 0 else " "
            print("  %-6s vs %-4s %+0.4f "
                  "[%+0.4f, %+0.4f] %s"
                  % (k, best, mm,
                     ci[0], ci[1], star))
        if "MEAN3" in M:
            mm, ci = boot(
                y, M["MEAN3"], M["MEAN2"], g
            )
            print("  CXR increment "
                  "%+0.4f [%+0.4f, %+0.4f]"
                  % (mm, ci[0], ci[1]))
        print("\n  spearman:")
        cc = pd.DataFrame(M)[uni].corr(
            method="spearman"
        )
        print(cc.to_string())

r = pd.DataFrame(rows)
out = os.path.join(
    FD, "fusion_harmonised_results.csv"
)
r.to_csv(out, index=False)
print("\nSaved", out)

