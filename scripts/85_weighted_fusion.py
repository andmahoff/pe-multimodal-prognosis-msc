import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
GRID = np.arange(0.0, 1.01, 0.05)


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

    y = d[lab_col].values.astype(int)
    g = d["subject_id"].values
    re = rank01(d["p_ehr"])
    rc = rank01(d["p_ecg"])

    print("\n" + "=" * 64)
    print("%s   n=%d ev=%d (%.4f)"
          % (nm, len(d), int(y.sum()),
             y.mean()))
    print("=" * 64)

    skf = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED
    )
    A = np.vstack([re, rc]).T

    wo = np.zeros(len(d))
    chosen = []
    for tr, te in skf.split(A, y, groups=g):
        best_w, best_a = 0.5, -1
        for w in GRID:
            s = w * re[tr] + (1 - w) * rc[tr]
            a = roc_auc_score(y[tr], s)
            if a > best_a:
                best_a, best_w = a, w
        chosen.append(round(best_w, 2))
        wo[te] = (
            best_w * re[te]
            + (1 - best_w) * rc[te]
        )

    st = np.zeros(len(d))
    for tr, te in skf.split(A, y, groups=g):
        lr = LogisticRegression(
            max_iter=3000, C=1.0
        )
        lr.fit(A[tr], y[tr])
        st[te] = lr.predict_proba(A[te])[:, 1]

    gw, ga = 0.5, -1
    for w in GRID:
        a = roc_auc_score(
            y, w * re + (1 - w) * rc
        )
        if a > ga:
            ga, gw = a, w

    M = {
        "EHR": re,
        "ECG": rc,
        "MEAN2": (re + rc) / 2,
        "WMEAN2": wo,
        "STACK2": st,
    }
    print("  fold weights (EHR):", chosen)
    print("  global-optimal EHR weight "
          "%.2f -> AUC %.4f  "
          "(fitted on all rows, not out-of-fold)"
          % (gw, ga))
    print()
    for k, v in M.items():
        a = roc_auc_score(y, v)
        p = average_precision_score(y, v)
        print("  %-7s AUC %.4f  AP %.4f"
              % (k, a, p))
        rows.append({
            "outcome": lab_col, "model": k,
            "n": len(d),
            "events": int(y.sum()),
            "auc": a, "ap": p,
        })

    best = "EHR" if roc_auc_score(
        y, re) >= roc_auc_score(y, rc) else "ECG"
    print("\n  vs %s:" % best)
    for k in ["MEAN2", "WMEAN2", "STACK2"]:
        m, ci = boot(y, M[k], M[best], g)
        s = "*" if ci[0] > 0 else " "
        print("    %-7s %+0.4f "
              "[%+0.4f, %+0.4f] %s"
              % (k, m, ci[0], ci[1], s))

    print("\n  WMEAN2 vs MEAN2:")
    m, ci = boot(y, M["WMEAN2"],
                 M["MEAN2"], g)
    print("    %+0.4f [%+0.4f, %+0.4f]"
          % (m, ci[0], ci[1]))

r = pd.DataFrame(rows)
out = os.path.join(
    FD, "weighted_fusion_results.csv"
)
r.to_csv(out, index=False)
print("\nSaved", out)

