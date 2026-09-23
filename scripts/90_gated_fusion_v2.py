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
EPS = 1e-6
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


def z(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1)


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


def oof_lr(X, y, g, skf):
    X = np.asarray(X, dtype=float)
    out = np.zeros(len(y))
    coefs = []
    for tr, te in skf.split(X, y, groups=g):
        lr = LogisticRegression(
            max_iter=5000, C=1.0)
        lr.fit(X[tr], y[tr])
        out[te] = lr.predict_proba(
            X[te])[:, 1]
        coefs.append(lr.coef_[0])
    return out, np.mean(coefs, axis=0)


tim = pd.read_csv(
    os.path.join(FD, "cxr_image_timing.csv"))
dur = tim[tim["rel"] == "during"]
meta = dur.groupby(KEY).agg(
    frac_port=("port", "mean"),
    n_dur=("port", "size"),
).reset_index()

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
    base_lab = "death_30d" if lc == \
        "death_30d_inhosp" else lc
    eh = pd.read_csv(os.path.join(
        FD, "p_ehr_harm_%s.csv" % base_lab))
    ec = pd.read_csv(os.path.join(
        FD, "p_ecg_harm_%s.csv" % lc))
    cx = pd.read_csv(os.path.join(
        FD, "p_cxr_harm_%s.csv" % base_lab))

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

    print("\n" + "=" * 64)
    print("%s   n=%d ev=%d (%.4f)"
          % (nm, len(d), int(y.sum()),
             y.mean()))
    print("=" * 64)

    skf = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    A2 = np.vstack([re, rc]).T

    base = np.zeros(len(y))
    w3 = np.zeros(len(y))
    for tr, te in skf.split(A2, y, groups=g):
        bw, ba = 0.5, -1
        for w in W2:
            v = roc_auc_score(
                y[tr],
                w * re[tr] + (1 - w) * rc[tr])
            if v > ba:
                ba, bw = v, w
        base[te] = (bw * re[te]
                    + (1 - bw) * rc[te])
        bt, ba = (0.4, 0.3, 0.3), -1
        for t3 in W3:
            s = (t3[0] * re[tr] + t3[1] * rc[tr]
                 + t3[2] * rx[tr])
            v = roc_auc_score(y[tr], s)
            if v > ba:
                ba, bt = v, t3
        w3[te] = (bt[0] * re[te]
                  + bt[1] * rc[te]
                  + bt[2] * rx[te])

    bz = z(base)
    cz = z(rx)
    conf = z(np.abs(bz))
    nd = z(np.log1p(d["n_dur"].values))
    pp = z(d["frac_port"].values)

    MODELS = {}
    MODELS["BASE(EHR+ECG)"] = (base, None)
    MODELS["WMEAN3"] = (w3, None)

    p, c = oof_lr(
        np.vstack([bz, cz]).T, y, g, skf)
    MODELS["RESID"] = (p, c)

    p, c = oof_lr(
        np.vstack([bz, cz, conf,
                   conf * cz]).T, y, g, skf)
    MODELS["GATE_conf"] = (p, c)

    p, c = oof_lr(
        np.vstack([bz, cz, nd,
                   nd * cz]).T, y, g, skf)
    MODELS["GATE_ndur"] = (p, c)

    p, c = oof_lr(
        np.vstack([bz, cz, pp,
                   pp * cz]).T, y, g, skf)
    MODELS["GATE_port"] = (p, c)

    p, c = oof_lr(
        np.vstack([bz, cz, conf, nd, pp,
                   conf * cz, nd * cz,
                   pp * cz]).T, y, g, skf)
    MODELS["GATE_all"] = (p, c)

    print("  %-16s %-8s %-8s"
          % ("model", "AUC", "AUPRC"))
    for k, (v, c) in MODELS.items():
        print("  %-16s %.4f   %.4f"
              % (k, roc_auc_score(y, v),
                 average_precision_score(
                     y, v)))

    print("\n  mean interaction coefficients:")
    for k in ["GATE_conf", "GATE_ndur",
              "GATE_port"]:
        c = MODELS[k][1]
        print("    %-10s cxr %+.3f  "
              "gate %+.3f  interaction %+.3f"
              % (k, c[1], c[2], c[3]))

    print("\n  vs RESID (value of the gate):")
    for k in ["GATE_conf", "GATE_ndur",
              "GATE_port", "GATE_all"]:
        m, ci = boot(y, MODELS[k][0],
                     MODELS["RESID"][0], g)
        s = "*" if ci[0] > 0 else " "
        print("    %-10s %+0.4f "
              "[%+0.4f, %+0.4f] %s"
              % (k, m, ci[0], ci[1], s))

    print("\n  vs WMEAN3:")
    for k in ["RESID", "GATE_conf",
              "GATE_ndur", "GATE_port",
              "GATE_all"]:
        m, ci = boot(y, MODELS[k][0],
                     MODELS["WMEAN3"][0], g)
        s = "*" if ci[0] > 0 else " "
        print("    %-10s %+0.4f "
              "[%+0.4f, %+0.4f] %s"
              % (k, m, ci[0], ci[1], s))

print("\nDone.")
