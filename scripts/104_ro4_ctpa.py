import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import brier_score_loss

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
LAB = DATA + "/mimic_labels_harmonised.csv"
FEAT = DATA + "/ctpa_comorb_features.csv"
OUT = DATA + "/ro4_ctpa_results.csv"

VAR = "base_cm_dv"
SEED = 42
NBOOT = 2000
STEP = 0.05
OUTS = ["composite_30d", "death_30d"]
EHRMAP = {}


def rk(x):
    return rankdata(x) / (len(x) + 1.0)


def grid3():
    g = []
    n = int(round(1.0 / STEP))
    for i in range(n + 1):
        for j in range(n - i + 1):
            a, b = i * STEP, j * STEP
            g.append((a, b, max(1.0 - a - b, 0.0)))
    return g


G3 = grid3()
G2 = [(w, 1.0 - w, 0.0)
      for w in np.arange(0, 1.0001, STEP)]


def wf(cols, w):
    s = np.zeros(len(cols[0]))
    for c, wi in zip(cols, w):
        s = s + wi * c
    return s


def oof_w(cols, y, grp, grid):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    oof = np.zeros(len(y))
    X0 = cols[0].reshape(-1, 1)
    for tr, te in cv.split(X0, y, grp):
        best, bw = None, None
        for w in grid:
            a = roc_auc_score(
                y[tr], wf([c[tr] for c in cols], w))
            if best is None or a > best:
                best, bw = a, w
        oof[te] = wf([c[te] for c in cols], bw)
    return oof


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong(y, p1, p2):
    o = np.argsort(-y)
    y2 = y[o]
    P = np.vstack([p1[o], p2[o]])
    m = int(y2.sum())
    n = len(y2) - m
    tx = np.vstack([_midrank(P[r, :m])
                    for r in range(2)])
    ty = np.vstack([_midrank(P[r, m:])
                    for r in range(2)])
    tz = np.vstack([_midrank(P[r, :])
                    for r in range(2)])
    au = (tz[:, :m].sum(axis=1) / m / n
          - (m + 1.0) / 2.0 / n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    d = au[0] - au[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return d, np.nan
    z = d / np.sqrt(var)
    return d, float(2 * (1 - norm.cdf(abs(z))))


def bdiff(y, pa, pb, grp):
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(NBOOT):
        pk = rng.choice(us, len(us), replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(roc_auc_score(y[ii], pa[ii])
                 - roc_auc_score(y[ii], pb[ii]))
    d = np.array(d)
    return (float(np.mean(d)),
            float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def youden(y, p):
    fpr, tpr, th = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    t = th[j]
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-9)
    return t, sens, spec, ppv, f1, (tp, fp, tn, fn)


def platt(y, p, grp):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    out = np.zeros(len(y))
    X = p.reshape(-1, 1)
    for tr, te in cv.split(X, y, grp):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X[tr], y[tr])
        out[te] = lr.predict_proba(X[te])[:, 1]
    return out


def nri(y, po, pn, grp):
    b = [0.05, 0.15]

    def band(p):
        return np.digitize(p, b)

    def stat(yy, a, c):
        up = (band(c) > band(a))
        dn = (band(c) < band(a))
        e = yy == 1
        ne = yy == 0
        if e.sum() == 0 or ne.sum() == 0:
            return np.nan
        return ((up[e].mean() - dn[e].mean())
                + (dn[ne].mean() - up[ne].mean()))

    obs = stat(y, po, pn)
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(NBOOT):
        pk = rng.choice(us, len(us), replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        v = stat(y[ii], po[ii], pn[ii])
        if not np.isnan(v):
            d.append(v)
    d = np.array(d)
    return (obs, float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)),
            float((d > 0).mean()))


def spec_at_sens(y, p, target):
    fpr, tpr, th = roc_curve(y, p)
    ok = tpr >= target
    if not ok.any():
        return np.nan
    return float(1 - fpr[ok][0])


# ---- sPESI ----
sp = None
for f in ["mimic_pe_ehr_baseline_v7.csv",
          "mimic_pe_ehr_baseline_v6.csv",
          "mimic_pe_ehr_baseline_v5.csv"]:
    pth = os.path.join(P2, f)
    if os.path.exists(pth):
        sp = pd.read_csv(pth)
        print("sPESI source:", f)
        break

spesi = None
if sp is not None:
    print("cols:", [c for c in sp.columns][:30])
    need = {"age": ["age_at_admit", "age"],
            "hr": ["mean_hr"],
            "sbp": ["mean_sbp"],
            "spo2": ["mean_spo2"],
            "ca": ["cancer"],
            "hf": ["heart_failure"],
            "cp": ["copd"]}
    got = {}
    for k, cands in need.items():
        for c in cands:
            if c in sp.columns:
                got[k] = c
                break
    print("sPESI components found:", got)
    if "hadm_id" in sp.columns and len(got) >= 5:
        s = pd.DataFrame()
        s["hadm_id"] = sp["hadm_id"]
        tot = np.zeros(len(sp))
        if "age" in got:
            tot += (sp[got["age"]] > 80).fillna(
                False).astype(int)
        if "ca" in got:
            tot += sp[got["ca"]].fillna(0).astype(int)
        cp = np.zeros(len(sp))
        if "hf" in got:
            cp += sp[got["hf"]].fillna(0).values
        if "cp" in got:
            cp += sp[got["cp"]].fillna(0).values
        tot += (cp > 0).astype(int)
        if "hr" in got:
            tot += (sp[got["hr"]] >= 110).fillna(
                False).astype(int)
        if "sbp" in got:
            tot += (sp[got["sbp"]] < 100).fillna(
                False).astype(int)
        if "spo2" in got:
            tot += (sp[got["spo2"]] < 90).fillna(
                False).astype(int)
        s["spesi"] = tot
        spesi = s.drop_duplicates("hadm_id")
        print("sPESI built, mean score %.3f"
              % float(s["spesi"].mean()))

lab = pd.read_csv(LAB)
fx = pd.read_csv(FEAT)[["hadm_id", "pe_pos"]]

rows = []
for out in OUTS:
    eh = EHRMAP.get(out, out)
    pe = pd.read_csv(DATA + "/p_ehr_harm_" + eh
                     + ".csv")[["hadm_id", "p_ehr"]]
    pg = pd.read_csv(DATA + "/p_ecg_harm_" + out
                     + ".csv")[["hadm_id", "p_ecg"]]
    pc = pd.read_csv(DATA + "/p_ctpa_pres_" + VAR
                     + "_" + out
                     + ".csv")[["hadm_id", "p_ctpa"]]
    d = lab.merge(pe, on="hadm_id")
    d = d.merge(pg, on="hadm_id")
    d = d.merge(pc, on="hadm_id")
    d = d.merge(fx, on="hadm_id")
    if spesi is not None:
        d = d.merge(spesi, on="hadm_id", how="left")
    d = d.reset_index(drop=True)

    y = d[out].values.astype(int)
    grp = d["subject_id"].values
    a = rk(d["p_ehr"].values)
    b = rk(d["p_ecg"].values)
    c = rk(d["p_ctpa"].values)

    w2 = oof_w([a, b], y, grp, G2)
    w3 = oof_w([a, b, c], y, grp, G3)
    m3 = (a + b + c) / 3.0

    print("")
    print("#" * 58)
    print("%s  n=%d ev=%d (%.4f)  [%s]"
          % (out, len(y), int(y.sum()),
             y.mean(), VAR))
    print("")
    print("DISCRIMINATION")
    tbl = {"EHR": a, "ECG": b, "CTPA": c,
           "MEAN3": m3, "WMEAN2": w2,
           "WMEAN3": w3}
    if spesi is not None:
        sv = d["spesi"].fillna(
            d["spesi"].median()).values
        tbl["sPESI"] = sv
    for k, v in tbl.items():
        print("  %-7s %.4f" % (k,
                               roc_auc_score(y, v)))

    print("")
    print("WMEAN3 COMPARISONS")
    for k in ["EHR", "WMEAN2", "sPESI"]:
        if k not in tbl:
            continue
        t = bdiff(y, w3, tbl[k], grp)
        dd, pv = delong(y, w3, tbl[k])
        print("  vs %-7s %+.4f [%+.4f, %+.4f]"
              "  DeLong p=%.4g"
              % (k, t[0], t[1], t[2], pv))
        rows.append({"outcome": out, "var": VAR,
                     "cmp": k, "diff": t[0],
                     "lo": t[1], "hi": t[2],
                     "delong_p": pv})

    print("")
    print("YOUDEN (WMEAN3)")
    t, se, sp2, pv2, f1, cm = youden(y, w3)
    print("  thr %.4f sens %.3f spec %.3f"
          "  ppv %.3f f1 %.3f"
          % (t, se, sp2, pv2, f1))
    print("  TP%d FP%d TN%d FN%d" % cm)

    if spesi is not None:
        print("")
        print("RH2: specificity at sPESI sens")
        sv = tbl["sPESI"]
        fpr, tpr, th = roc_curve(y, sv)
        j = np.argmax(tpr - fpr)
        tgt = tpr[j]
        s_sp = 1 - fpr[j]
        w_sp = spec_at_sens(y, w3, tgt)
        print("  sPESI sens %.3f spec %.3f"
              " -> WMEAN3 spec %.3f (gain %+.3f)"
              % (tgt, s_sp, w_sp, w_sp - s_sp))

    print("")
    print("CALIBRATION (5-fold OOF Platt)")
    cal = platt(y, w3, grp)
    print("  Brier raw %.4f -> platt %.4f"
          " (null %.4f)"
          % (brier_score_loss(y, w3),
             brier_score_loss(y, cal),
             brier_score_loss(
                 y, np.full(len(y), y.mean()))))
    dec = pd.qcut(pd.Series(cal), 10,
                  labels=False, duplicates="drop")
    dt = pd.DataFrame({"d": dec, "y": y,
                       "p": cal})
    print(dt.groupby("d").agg(
        n=("y", "size"), obs=("y", "mean"),
        pred=("p", "mean")).round(4).to_string())

    print("")
    print("NRI (WMEAN2 -> WMEAN3)")
    c2 = platt(y, w2, grp)
    nv = nri(y, c2, cal, grp)
    print("  total %+.4f [%+.4f, %+.4f] P=%.3f"
          % nv)

    po = pd.DataFrame({
        "subject_id": d["subject_id"].values,
        "hadm_id": d["hadm_id"].values,
        "p_wmean2": w2, "p_wmean3": w3,
        "p_wmean3_cal": cal})
    po.to_csv(DATA + "/p_wmean3_ctpa_" + out
              + ".csv", index=False)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("")
print("saved", OUT)
