import os
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

FD = (
    "./"
    "fusion_workspace/data/"
)
KEY = ["subject_id", "hadm_id"]
SEED = 42
NBOOT = 2000

W2 = np.arange(0.0, 1.01, 0.05)
W3 = [
    (a / 10.0, b / 10.0, c / 10.0)
    for a in range(11)
    for b in range(11)
    for c in range(11)
    if a + b + c == 10
]


def rank01(x):
    return pd.Series(x).rank(pct=True).values


def boot(y, pa, pb, g, n=NBOOT):
    rng = np.random.RandomState(SEED)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    o = []
    for _ in range(n):
        s = rng.choice(uq, len(uq),
                       replace=True)
        t = np.concatenate([idx[u] for u in s])
        if len(np.unique(y[t])) < 2:
            continue
        o.append(roc_auc_score(y[t], pa[t])
                 - roc_auc_score(y[t], pb[t]))
    o = np.array(o)
    return o.mean(), np.percentile(
        o, [2.5, 97.5])


lab = pd.read_csv(
    os.path.join(
        FD, "mimic_labels_harmonised.csv"))

TASKS = [
    ("cv_first", "CV only"),
    ("composite_30d", "composite"),
    ("death_30d", "death (dod)"),
    ("death_30d_inhosp", "death (in-hosp)"),
]

rows = []
for lc, nm in TASKS:
    eh_lab = "death_30d" if lc == \
        "death_30d_inhosp" else lc
    cx_lab = eh_lab
    eh = pd.read_csv(os.path.join(
        FD, "p_ehr_harm_%s.csv" % eh_lab))
    ec = pd.read_csv(os.path.join(
        FD, "p_ecg_harm_%s.csv" % lc))
    cx = pd.read_csv(os.path.join(
        FD, "p_cxr_harm_%s.csv" % cx_lab))

    d = lab.merge(eh, on=KEY, how="inner")
    d = d.merge(ec, on=KEY, how="inner")
    d = d.merge(cx, on=KEY, how="inner")
    if lc == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)

    y = d[lc].values.astype(int)
    g = d["subject_id"].values
    re = rank01(d["p_ehr"])
    rc = rank01(d["p_ecg"])
    rx = rank01(d["p_cxr"])

    print("\n" + "=" * 64)
    print("%s   n=%d ev=%d (%.4f)"
          % (nm, len(d), int(y.sum()),
             y.mean()))
    print("=" * 64)

    skf = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    A = np.vstack([re, rc, rx]).T

    w2 = np.zeros(len(d))
    w3 = np.zeros(len(d))
    ws2, ws3 = [], []
    for tr, te in skf.split(A, y, groups=g):
        bw, ba = 0.5, -1
        for w in W2:
            a = roc_auc_score(
                y[tr],
                w * re[tr] + (1 - w) * rc[tr])
            if a > ba:
                ba, bw = a, w
        ws2.append(round(bw, 2))
        w2[te] = bw * re[te] + (1 - bw) * rc[te]

        bt, ba = (0.5, 0.5, 0.0), -1
        for t3 in W3:
            s = (t3[0] * re[tr] + t3[1] * rc[tr]
                 + t3[2] * rx[tr])
            a = roc_auc_score(y[tr], s)
            if a > ba:
                ba, bt = a, t3
        ws3.append(bt)
        w3[te] = (bt[0] * re[te]
                  + bt[1] * rc[te]
                  + bt[2] * rx[te])

    M = {
        "EHR": re, "ECG": rc, "CXR": rx,
        "MEAN3": (re + rc + rx) / 3,
        "WMEAN2": w2, "WMEAN3": w3,
    }
    print("  two-modality weights (EHR):", ws2)
    print("  three-modality weights "
          "(EHR, ECG, CXR):", ws3)
    print()
    for k, v in M.items():
        a = roc_auc_score(y, v)
        p = average_precision_score(y, v)
        print("  %-7s AUC %.4f  AP %.4f"
              % (k, a, p))
        rows.append({
            "outcome": lc, "model": k,
            "n": len(d),
            "events": int(y.sum()),
            "auc": a, "ap": p})

    print("\n  CXR increment "
          "(WMEAN3 vs WMEAN2):")
    m, ci = boot(y, M["WMEAN3"],
                 M["WMEAN2"], g)
    s = "*" if ci[0] > 0 else " "
    print("    %+0.4f [%+0.4f, %+0.4f] %s"
          % (m, ci[0], ci[1], s))

    print("\n  WMEAN3 vs best unimodal:")
    best = max(["EHR", "ECG", "CXR"],
               key=lambda k:
               roc_auc_score(y, M[k]))
    m, ci = boot(y, M["WMEAN3"], M[best], g)
    s = "*" if ci[0] > 0 else " "
    print("    vs %-4s %+0.4f "
          "[%+0.4f, %+0.4f] %s"
          % (best, m, ci[0], ci[1], s))

    cc = pd.DataFrame(
        {"EHR": re, "ECG": rc, "CXR": rx}
    ).corr(method="spearman")
    print("\n  spearman:")
    print(cc.to_string())

r = pd.DataFrame(rows)
out = os.path.join(
    FD, "threeway_weighted_results.csv")
r.to_csv(out, index=False)
print("\nSaved", out)

