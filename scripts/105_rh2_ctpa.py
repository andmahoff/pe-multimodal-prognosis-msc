import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
LAB = DATA + "/mimic_labels_harmonised.csv"
V7 = P2 + "/mimic_pe_ehr_baseline_v7.csv"
OUT = DATA + "/rh2_ctpa_results.csv"

VAR = "base_cm_dv"
SEED = 42
NBOOT = 2000
STEP = 0.05
OUTS = ["composite_30d", "death_30d"]

AGEPAT = ["age_at_admit", "age", "anchor_age"]


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


lab = pd.read_csv(LAB)
need = set(lab["hadm_id"].unique())

print("scanning phase2_mimic for age...")
cands = []
for f in sorted(glob.glob(P2 + "/*.csv")):
    try:
        h = pd.read_csv(f, nrows=0)
    except Exception:
        continue
    cols = list(h.columns)
    if "hadm_id" not in cols:
        continue
    ac = None
    for a in AGEPAT:
        if a in cols:
            ac = a
            break
    if ac is None:
        continue
    try:
        df = pd.read_csv(f, usecols=["hadm_id", ac])
    except Exception:
        continue
    df = df.dropna().drop_duplicates("hadm_id")
    cov = len(set(df["hadm_id"]) & need)
    cands.append((cov, f, ac, df))
    print("  %-52s %-14s cov %d/%d"
          % (os.path.basename(f), ac,
             cov, len(need)))

if not cands:
    raise SystemExit("no age column found")

cands.sort(key=lambda t: -t[0])
cov, fbest, acol, agedf = cands[0]
print("")
print("using:", os.path.basename(fbest),
      "col", acol, "coverage", cov)
agedf = agedf.rename(columns={acol: "age"})

v7 = pd.read_csv(V7)
sp = v7[["hadm_id", "mean_hr", "mean_sbp",
         "mean_spo2", "cancer", "copd",
         "heart_failure"]].drop_duplicates(
    "hadm_id")
sp = sp.merge(agedf, on="hadm_id", how="left")
print("age missing after merge:",
      int(sp["age"].isna().sum()), "of", len(sp))


def build(df, use_age):
    t = np.zeros(len(df))
    t += (df["mean_hr"] >= 110).fillna(
        False).astype(int).values
    t += (df["mean_sbp"] < 100).fillna(
        False).astype(int).values
    t += (df["mean_spo2"] < 90).fillna(
        False).astype(int).values
    t += df["cancer"].fillna(0).astype(int).values
    cp = (df["heart_failure"].fillna(0).values
          + df["copd"].fillna(0).values)
    t += (cp > 0).astype(int)
    if use_age:
        t += (df["age"] > 80).fillna(
            False).astype(int).values
    return t


sp["spesi5"] = build(sp, False)
sp["spesi6"] = build(sp, True)
print("mean spesi5 %.3f  spesi6 %.3f"
      % (sp["spesi5"].mean(),
         sp["spesi6"].mean()))
print("age>80 rate %.3f"
      % float((sp["age"] > 80).mean()))

fx = pd.read_csv(DATA
                 + "/ctpa_comorb_features.csv")
fx = fx[["hadm_id"]]

rows = []
for out in OUTS:
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
    d = d.merge(sp[["hadm_id", "spesi5",
                    "spesi6"]], on="hadm_id",
                how="left")
    d = d.reset_index(drop=True)

    y = d[out].values.astype(int)
    grp = d["subject_id"].values
    a = rk(d["p_ehr"].values)
    b = rk(d["p_ecg"].values)
    c = rk(d["p_ctpa"].values)
    w2 = oof_w([a, b], y, grp, G2)
    w3 = oof_w([a, b, c], y, grp, G3)

    s5 = d["spesi5"].fillna(
        d["spesi5"].median()).values
    s6 = d["spesi6"].fillna(
        d["spesi6"].median()).values

    print("")
    print("#" * 56)
    print("%s n=%d ev=%d" % (out, len(y),
                             int(y.sum())))
    print("  sPESI 5-crit AUC %.4f"
          % roc_auc_score(y, s5))
    print("  sPESI 6-crit AUC %.4f"
          % roc_auc_score(y, s6))
    print("  WMEAN2 %.4f  WMEAN3 %.4f"
          % (roc_auc_score(y, w2),
             roc_auc_score(y, w3)))

    for nm, mdl in [("WMEAN2", w2),
                    ("WMEAN3", w3)]:
        us = np.unique(grp)
        ix = {u: np.where(grp == u)[0]
              for u in us}
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(NBOOT):
            pk = rng.choice(us, len(us),
                            replace=True)
            ii = np.concatenate(
                [ix[u] for u in pk])
            yb = y[ii]
            if len(np.unique(yb)) < 2:
                continue
            f1, t1, _ = roc_curve(yb, s6[ii])
            j = int(np.argmax(t1 - f1))
            tgt = t1[j]
            ssp = 1.0 - f1[j]
            f2, t2, _ = roc_curve(yb, mdl[ii])
            ok = t2 >= tgt
            if not ok.any():
                continue
            msp = 1.0 - f2[ok][0]
            diffs.append(msp - ssp)
        dv = np.array(diffs)

        f1, t1, _ = roc_curve(y, s6)
        j = int(np.argmax(t1 - f1))
        tgt = t1[j]
        ssp = 1.0 - f1[j]
        f2, t2, _ = roc_curve(y, mdl)
        ok = t2 >= tgt
        msp = 1.0 - f2[ok][0] if ok.any() else np.nan

        print("")
        print("  RH2 %s vs sPESI(6)" % nm)
        print("    sPESI sens %.3f spec %.3f"
              % (tgt, ssp))
        print("    %s spec %.3f  gain %+.3f"
              % (nm, msp, msp - ssp))
        print("    boot mean %+.4f "
              "[%+.4f, %+.4f]  P(>0)=%.3f"
              % (float(dv.mean()),
                 float(np.percentile(dv, 2.5)),
                 float(np.percentile(dv, 97.5)),
                 float((dv > 0).mean())))
        rows.append({
            "outcome": out, "model": nm,
            "spesi_sens": tgt,
            "spesi_spec": ssp,
            "model_spec": msp,
            "gain": msp - ssp,
            "boot_mean": float(dv.mean()),
            "lo": float(np.percentile(dv, 2.5)),
            "hi": float(np.percentile(dv, 97.5)),
            "p_gt0": float((dv > 0).mean())})

pd.DataFrame(rows).to_csv(OUT, index=False)
print("")
print(pd.DataFrame(rows).round(4).to_string())
print("saved", OUT)
