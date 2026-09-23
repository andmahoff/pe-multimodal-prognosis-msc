import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
LAB = DATA + "/mimic_labels_harmonised.csv"
V7 = P2 + "/mimic_pe_ehr_baseline_v7.csv"
AGEF = P2 + "/mimic_pe_ehr_baseline.csv"
OUT = DATA + "/spesi6_recheck.csv"

VAR = "base_cm_dv"
SEED = 42
NBOOT = 2000
STEP = 0.05
ALL4 = ["cv_first", "composite_30d",
        "death_30d", "death_30d_inhosp"]
CT2 = ["composite_30d", "death_30d"]
EHRMAP = {"death_30d_inhosp": "death_30d"}


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


def rh2boot(y, mdl, sp, grp):
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(NBOOT):
        pk = rng.choice(us, len(us), replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        yb = y[ii]
        if len(np.unique(yb)) < 2:
            continue
        f1, t1, _ = roc_curve(yb, sp[ii])
        j = int(np.argmax(t1 - f1))
        tgt = t1[j]
        ssp = 1.0 - f1[j]
        f2, t2, _ = roc_curve(yb, mdl[ii])
        ok = t2 >= tgt
        if not ok.any():
            continue
        d.append((1.0 - f2[ok][0]) - ssp)
    d = np.array(d)
    return (float(d.mean()),
            float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)),
            float((d > 0).mean()))


def point(y, mdl, sp):
    f1, t1, _ = roc_curve(y, sp)
    j = int(np.argmax(t1 - f1))
    tgt = t1[j]
    ssp = 1.0 - f1[j]
    f2, t2, _ = roc_curve(y, mdl)
    ok = t2 >= tgt
    msp = 1.0 - f2[ok][0] if ok.any() else np.nan
    return tgt, ssp, msp


lab = pd.read_csv(LAB)
age = pd.read_csv(AGEF,
                  usecols=["hadm_id",
                           "age_at_admit"])
age = age.dropna().drop_duplicates("hadm_id")
v7 = pd.read_csv(V7)
sp = v7[["hadm_id", "mean_hr", "mean_sbp",
         "mean_spo2", "cancer", "copd",
         "heart_failure"]].drop_duplicates(
    "hadm_id")
sp = sp.merge(age, on="hadm_id", how="left")

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
sp["spesi6"] = t
sp = sp[["hadm_id", "spesi6"]]
print("spesi6 mean %.3f" % sp["spesi6"].mean())

rows = []

print("")
print("#" * 58)
print("PART A: CTPA cohort, WMEAN3 vs sPESI6")
for out in CT2:
    pe = pd.read_csv(DATA + "/p_ehr_harm_" + out
                     + ".csv")[["hadm_id", "p_ehr"]]
    pg = pd.read_csv(DATA + "/p_ecg_harm_" + out
                     + ".csv")[["hadm_id", "p_ecg"]]
    pc = pd.read_csv(DATA + "/p_ctpa_pres_" + VAR
                     + "_" + out
                     + ".csv")[["hadm_id", "p_ctpa"]]
    d = lab.merge(pe, on="hadm_id")
    d = d.merge(pg, on="hadm_id")
    d = d.merge(pc, on="hadm_id")
    d = d.merge(sp, on="hadm_id", how="left")
    d = d.reset_index(drop=True)
    y = d[out].values.astype(int)
    grp = d["subject_id"].values
    s6 = d["spesi6"].fillna(
        d["spesi6"].median()).values
    w3 = oof_w([rk(d["p_ehr"].values),
                rk(d["p_ecg"].values),
                rk(d["p_ctpa"].values)],
               y, grp, G3)
    dd, pv = delong(y, w3, s6)
    print("")
    print("%s n=%d ev=%d" % (out, len(y),
                             int(y.sum())))
    print("  WMEAN3 %.4f  sPESI6 %.4f"
          % (roc_auc_score(y, w3),
             roc_auc_score(y, s6)))
    print("  diff %+.4f  DeLong p=%.4g"
          % (dd, pv))
    rows.append({"part": "A", "outcome": out,
                 "model": "WMEAN3",
                 "auc": roc_auc_score(y, w3),
                 "spesi6": roc_auc_score(y, s6),
                 "diff": dd, "delong_p": pv})

print("")
print("#" * 58)
print("PART B: full cohort, script 86 recheck")
for out in ALL4:
    f = DATA + "/p_wmean2_" + out + ".csv"
    try:
        pw = pd.read_csv(f)
    except Exception as e:
        print(out, "-- missing:", e)
        continue
    cols = [c for c in pw.columns
            if "wmean2" in c.lower()
            and "cal" not in c.lower()]
    if not cols:
        print(out, "cols:", list(pw.columns))
        continue
    pc = cols[0]
    d = lab.merge(pw[["hadm_id", pc]],
                  on="hadm_id")
    d = d.merge(sp, on="hadm_id", how="left")
    if out == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)
    y = d[out].values.astype(int)
    grp = d["subject_id"].values
    s6 = d["spesi6"].fillna(
        d["spesi6"].median()).values
    w2 = d[pc].values
    dd, pv = delong(y, w2, s6)
    tgt, ssp, msp = point(y, w2, s6)
    bm, lo, hi, pg2 = rh2boot(y, w2, s6, grp)
    print("")
    print("%s n=%d ev=%d  (col %s)"
          % (out, len(y), int(y.sum()), pc))
    print("  WMEAN2 %.4f  sPESI6 %.4f"
          "  diff %+.4f p=%.4g"
          % (roc_auc_score(y, w2),
             roc_auc_score(y, s6), dd, pv))
    print("  RH2: sPESI sens %.3f spec %.3f"
          " -> WMEAN2 spec %.3f (gain %+.3f)"
          % (tgt, ssp, msp, msp - ssp))
    print("       boot %+.4f [%+.4f, %+.4f]"
          "  P(>0)=%.3f" % (bm, lo, hi, pg2))
    rows.append({"part": "B", "outcome": out,
                 "model": "WMEAN2",
                 "auc": roc_auc_score(y, w2),
                 "spesi6": roc_auc_score(y, s6),
                 "diff": dd, "delong_p": pv,
                 "gain": msp - ssp,
                 "boot": bm, "lo": lo,
                 "hi": hi, "p_gt0": pg2})

pd.DataFrame(rows).to_csv(OUT, index=False)
print("")
print(pd.DataFrame(rows).round(4).to_string())
print("saved", OUT)
