import os
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
W3 = [
    (a / 10.0, b / 10.0, c / 10.0)
    for a in range(11)
    for b in range(11)
    for c in range(11)
    if a + b + c == 10
]


def rank01(x):
    return pd.Series(x).rank(pct=True).values


def auc(y, p):
    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return roc_auc_score(y, p)
    except ValueError:
        return np.nan


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


def wfuse(a, b, c, y, g, skf):
    A = np.vstack([a, b, c]).T
    out = np.zeros(len(y))
    ws = []
    for tr, te in skf.split(A, y, groups=g):
        bt, ba = (0.34, 0.33, 0.33), -1
        for t3 in W3:
            s = (t3[0] * a[tr] + t3[1] * b[tr]
                 + t3[2] * c[tr])
            v = roc_auc_score(y[tr], s)
            if v > ba:
                ba, bt = v, t3
        ws.append(bt)
        out[te] = (bt[0] * a[te]
                   + bt[1] * b[te]
                   + bt[2] * c[te])
    return out, ws


print("Building metadata...")
tim = pd.read_csv(
    os.path.join(FD, "cxr_image_timing.csv"))
dur = tim[tim["rel"] == "during"]
meta = dur.groupby(KEY).agg(
    frac_port=("port", "mean"),
    n_dur=("port", "size"),
).reset_index()
print("  admissions with index imgs:",
      len(meta))

lab = pd.read_csv(
    os.path.join(
        FD, "mimic_labels_harmonised.csv"))

TASKS = [
    ("cv_first", "CV only"),
    ("composite_30d", "composite"),
    ("death_30d", "death (dod)"),
    ("death_30d_inhosp", "death (in-hosp)"),
]

for lc, nm in TASKS:
    base = "death_30d" if lc == \
        "death_30d_inhosp" else lc
    eh = pd.read_csv(os.path.join(
        FD, "p_ehr_harm_%s.csv" % base))
    ec = pd.read_csv(os.path.join(
        FD, "p_ecg_harm_%s.csv" % lc))
    cx = pd.read_csv(os.path.join(
        FD, "p_cxr_harm_%s.csv" % base))

    d = lab.merge(eh, on=KEY, how="inner")
    d = d.merge(ec, on=KEY, how="inner")
    d = d.merge(cx, on=KEY, how="inner")
    d = d.merge(meta, on=KEY, how="inner")
    if lc == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)

    y = d[lc].values.astype(int)
    g = d["subject_id"].values
    re = rank01(d["p_ehr"])
    rc = rank01(d["p_ecg"])
    rx = rank01(d["p_cxr"])
    rp = rank01(d["frac_port"])
    rn = rank01(d["n_dur"])
    rm = (rp + rn) / 2

    print("\n" + "=" * 64)
    print("%s   n=%d ev=%d (%.4f)"
          % (nm, len(d), int(y.sum()),
             y.mean()))
    print("=" * 64)
    print("  p_cxr          AUC %.4f  AP %.4f"
          % (auc(y, rx),
             average_precision_score(y, rx)))
    print("  frac_portable  AUC %.4f"
          % auc(y, rp))
    print("  n_dur          AUC %.4f"
          % auc(y, rn))
    print("  META combined  AUC %.4f"
          % auc(y, rm))

    m, ci = boot(y, rx, rm, g)
    s = "*" if ci[0] > 0 else " "
    print("\n  p_cxr vs META  %+0.4f "
          "[%+0.4f, %+0.4f] %s"
          % (m, ci[0], ci[1], s))

    print("\n  correlations with p_cxr:")
    for k, v in [("frac_port", rp),
                 ("n_dur", rn)]:
        print("    %-11s spearman %+.3f"
              % (k, pd.Series(rx).corr(
                  pd.Series(v),
                  method="spearman")))

    print("\n  p_cxr within portable strata:")
    fp = d["frac_port"].values
    for lbl, msk in [
        ("all-portable", fp == 1),
        ("none-portable", fp == 0),
        ("mixed", (fp > 0) & (fp < 1)),
    ]:
        if msk.sum() < 30:
            print("    %-14s n=%d (too few)"
                  % (lbl, int(msk.sum())))
            continue
        print("    %-14s n=%-4d ev=%-3d "
              "AUC %.4f"
              % (lbl, int(msk.sum()),
                 int(y[msk].sum()),
                 auc(y[msk], rx[msk])))

    skf = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    f_cxr, w_cxr = wfuse(re, rc, rx, y, g, skf)
    f_met, w_met = wfuse(re, rc, rm, y, g, skf)

    print("\n  FUSION TEST")
    print("    EHR+ECG+CXR   AUC %.4f"
          % auc(y, f_cxr))
    print("      weights:", w_cxr)
    print("    EHR+ECG+META  AUC %.4f"
          % auc(y, f_met))
    print("      weights:", w_met)
    m, ci = boot(y, f_cxr, f_met, g)
    s = "*" if ci[0] > 0 else " "
    print("    CXR-fusion vs META-fusion "
          "%+0.4f [%+0.4f, %+0.4f] %s"
          % (m, ci[0], ci[1], s))

print("\nDone.")
