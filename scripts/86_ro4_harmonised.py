import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import brier_score_loss

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
KEY = ["subject_id", "hadm_id"]
SEED = 42
NBOOT = 2000
GRID = np.arange(0.0, 1.01, 0.05)
CUTS = [0.05, 0.15]


def rank01(x):
    return pd.Series(x).rank(pct=True).values


def midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong(y, p1, p2):
    y = np.asarray(y)
    pos = y == 1
    neg = y == 0
    m = int(pos.sum())
    n = int(neg.sum())
    P = [np.asarray(p1, dtype=float),
         np.asarray(p2, dtype=float)]
    tz = np.array([
        np.concatenate([p[pos], p[neg]])
        for p in P
    ])
    tzr = np.array([midrank(z) for z in tz])
    txr = np.array([midrank(p[pos]) for p in P])
    tyr = np.array([midrank(p[neg]) for p in P])
    a = (tzr[:, :m].sum(axis=1)
         - m * (m + 1) / 2) / (m * n)
    v01 = (tzr[:, :m] - txr) / n
    v10 = 1 - (tzr[:, m:] - tyr) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    L = np.array([1.0, -1.0])
    var = L.dot(S).dot(L)
    if var <= 0:
        return a, np.nan, np.nan
    z = (a[0] - a[1]) / np.sqrt(var)
    return a, z, 2 * (1 - stats.norm.cdf(abs(z)))


def boot(y, pa, pb, g, n=NBOOT):
    rng = np.random.RandomState(SEED)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    o = []
    for _ in range(n):
        s = rng.choice(uq, len(uq), replace=True)
        t = np.concatenate([idx[u] for u in s])
        if len(np.unique(y[t])) < 2:
            continue
        o.append(roc_auc_score(y[t], pa[t])
                 - roc_auc_score(y[t], pb[t]))
    o = np.array(o)
    return o.mean(), np.percentile(o, [2.5, 97.5])


def bstats(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    se = tp / max(tp + fn, 1)
    sp = tn / max(tn + fp, 1)
    pv = tp / max(tp + fp, 1)
    f1 = 2 * pv * se / max(pv + se, 1e-9)
    return se, sp, pv, f1, (tp, fp, tn, fn)


def spec_at_sens(y, p, tgt):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= tgt - 1e-12)[0]
    if len(ok) == 0:
        return np.nan, np.nan
    return 1.0 - fpr[ok[0]], thr[ok[0]]


def rh2_boot(y, pm, spb, g, n=NBOOT):
    rng = np.random.RandomState(SEED)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    o = []
    for _ in range(n):
        s = rng.choice(uq, len(uq), replace=True)
        t = np.concatenate([idx[u] for u in s])
        yy = y[t]
        if len(np.unique(yy)) < 2:
            continue
        se, sp, _, _, _ = bstats(yy, spb[t])
        if se <= 0 or se >= 1:
            continue
        sm, _ = spec_at_sens(yy, pm[t], se)
        if np.isnan(sm):
            continue
        o.append(sm - sp)
    o = np.array(o)
    return (o.mean(), np.median(o),
            np.percentile(o, [2.5, 97.5]),
            (o > 0).mean())


def nri_parts(y, po, pn):
    co = np.digitize(po, CUTS)
    cn = np.digitize(pn, CUTS)
    ev = y == 1
    ne = y == 0
    if ev.sum() == 0 or ne.sum() == 0:
        return np.nan, np.nan, np.nan
    a = ((cn[ev] > co[ev]).sum()
         - (cn[ev] < co[ev]).sum()) / ev.sum()
    b = ((cn[ne] < co[ne]).sum()
         - (cn[ne] > co[ne]).sum()) / ne.sum()
    return a, b, a + b


def nri_boot(y, po, pn, g, n=NBOOT):
    rng = np.random.RandomState(SEED)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    T = []
    for _ in range(n):
        s = rng.choice(uq, len(uq), replace=True)
        t = np.concatenate([idx[u] for u in s])
        _, _, c = nri_parts(y[t], po[t], pn[t])
        if not np.isnan(c):
            T.append(c)
    T = np.array(T)
    return (T.mean(), np.percentile(T, [2.5, 97.5]),
            (T > 0).mean())


def oof_platt(x, y, g, skf):
    x = np.asarray(x).reshape(-1, 1)
    out = np.zeros(len(y))
    for tr, te in skf.split(x, y, groups=g):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(x[tr], y[tr])
        out[te] = lr.predict_proba(x[te])[:, 1]
    return out


print("Loading...")
lab = pd.read_csv(
    os.path.join(FD, "mimic_labels_harmonised.csv")
)
v6 = pd.read_csv(
    os.path.join(P2, "mimic_pe_ehr_baseline_v6.csv")
).drop_duplicates(KEY)
v6["spesi"] = (
    (v6["age_at_admit"] > 80).astype(int)
    + (v6["cancer"] > 0).astype(int)
    + ((v6["heart_failure"] > 0)
       | (v6["copd"] > 0)).astype(int)
    + (v6["mean_hr"] >= 110).fillna(False).astype(int)
    + (v6["mean_sbp"] < 100).fillna(False).astype(int)
    + (v6["mean_spo2"] < 90).fillna(False).astype(int)
)

TASKS = [
    ("cv_first", "CV only"),
    ("composite_30d", "composite"),
    ("death_30d", "death (dod)"),
    ("death_30d_inhosp", "death (in-hosp)"),
]

allrows = []
for lc, nm in TASKS:
    ec_lab = lc
    eh_lab = "death_30d" if lc == \
        "death_30d_inhosp" else lc
    eh = pd.read_csv(os.path.join(
        FD, "p_ehr_harm_%s.csv" % eh_lab))
    ec = pd.read_csv(os.path.join(
        FD, "p_ecg_harm_%s.csv" % ec_lab))
    d = lab.merge(eh, on=KEY, how="inner")
    d = d.merge(ec, on=KEY, how="inner")
    d = d.merge(v6[KEY + ["spesi"]], on=KEY,
                how="left")
    if lc == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)

    y = d[lc].values.astype(int)
    g = d["subject_id"].values
    re = rank01(d["p_ehr"])
    rc = rank01(d["p_ecg"])
    sp_score = d["spesi"].fillna(0).values
    spb = (sp_score >= 1).astype(int)

    skf = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    A = np.vstack([re, rc]).T
    w_oof = np.zeros(len(d))
    ws = []
    for tr, te in skf.split(A, y, groups=g):
        bw, ba = 0.5, -1
        for w in GRID:
            a = roc_auc_score(
                y[tr], w * re[tr] + (1 - w) * rc[tr])
            if a > ba:
                ba, bw = a, w
        ws.append(round(bw, 2))
        w_oof[te] = bw * re[te] + (1 - bw) * rc[te]

    M = {
        "EHR": re, "ECG": rc,
        "sPESI": rank01(sp_score),
        "MEAN2": (re + rc) / 2,
        "WMEAN2": w_oof,
    }

    print("\n" + "=" * 66)
    print("%s   n=%d ev=%d (%.4f)  weights %s"
          % (nm, len(d), int(y.sum()),
             y.mean(), ws))
    print("=" * 66)
    print("  %-8s %-8s %-8s" % ("model", "AUC", "AUPRC"))
    for k, v in M.items():
        a = roc_auc_score(y, v)
        p = average_precision_score(y, v)
        print("  %-8s %.4f   %.4f" % (k, a, p))
        allrows.append({
            "outcome": lc, "model": k,
            "n": len(d), "events": int(y.sum()),
            "auc": a, "ap": p})

    print("\n  WMEAN2 vs:")
    for k in ["EHR", "ECG", "sPESI", "MEAN2"]:
        m, ci = boot(y, M["WMEAN2"], M[k], g)
        _, z, pv = delong(y, M["WMEAN2"], M[k])
        s = "*" if ci[0] > 0 else " "
        print("    %-7s %+0.4f [%+0.4f,%+0.4f] %s"
              "  DeLong p=%.4f" % (k, m, ci[0],
                                   ci[1], s, pv))

    print("\n  Youden thresholds:")
    for k in ["EHR", "WMEAN2"]:
        fpr, tpr, thr = roc_curve(y, M[k])
        t = thr[np.argmax(tpr - fpr)]
        se, sn, pv, f1, cm = bstats(
            y, (M[k] >= t).astype(int))
        print("    %-7s sens %.3f spec %.3f "
              "ppv %.3f f1 %.3f  TP%d FP%d TN%d FN%d"
              % (k, se, sn, pv, f1, cm[0],
                 cm[1], cm[2], cm[3]))
    se, sn, pv, f1, cm = bstats(y, spb)
    print("    %-7s sens %.3f spec %.3f "
          "ppv %.3f f1 %.3f  TP%d FP%d TN%d FN%d"
          % ("sPESI>=1", se, sn, pv, f1,
             cm[0], cm[1], cm[2], cm[3]))

    print("\n  RH2 (spec at sPESI sensitivity):")
    sm, _ = spec_at_sens(y, M["WMEAN2"], se)
    print("    sPESI spec %.3f | WMEAN2 %.3f "
          "| gain %+0.3f" % (sn, sm, sm - sn))
    bm, bmd, bci, bp = rh2_boot(
        y, M["WMEAN2"], spb, g)
    print("    boot %+0.4f (med %+0.4f) "
          "[%+0.4f,%+0.4f] P(>0)=%.3f"
          % (bm, bmd, bci[0], bci[1], bp))

    print("\n  Calibration (Platt OOF):")
    for k in ["EHR", "WMEAN2"]:
        cal = oof_platt(M[k], y, g, skf)
        print("    %-7s Brier %.4f -> %.4f"
              % (k, brier_score_loss(y, M[k]),
                 brier_score_loss(y, cal)))
        if k == "WMEAN2":
            cw = cal
        else:
            ce = cal
    b = pd.qcut(cw, 10, duplicates="drop")
    cb = pd.DataFrame({"b": b, "y": y, "p": cw})
    print(cb.groupby("b", observed=True).agg(
        n=("y", "size"), obs=("y", "mean"),
        pred=("p", "mean")).to_string())

    print("\n  NRI (EHR -> WMEAN2):")
    a, bnr, c = nri_parts(y, ce, cw)
    print("    events %+0.4f  nonevents %+0.4f "
          " total %+0.4f" % (a, bnr, c))
    nm_, nci, npos = nri_boot(y, ce, cw, g)
    print("    total boot %+0.4f [%+0.4f,%+0.4f]"
          "  P(>0)=%.3f" % (nm_, nci[0],
                            nci[1], npos))

    out = d[KEY].copy()
    out["p_wmean2"] = w_oof
    out["p_wmean2_cal"] = cw
    out.to_csv(os.path.join(
        FD, "p_wmean2_%s.csv" % lc), index=False)

pd.DataFrame(allrows).to_csv(
    os.path.join(FD, "ro4_harmonised_results.csv"),
    index=False)
print("\nSaved ro4_harmonised_results.csv "
      "and p_wmean2_*.csv")
