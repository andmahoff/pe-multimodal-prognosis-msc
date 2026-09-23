import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
OUT = DATA + "/dca_plots"
os.makedirs(OUT, exist_ok=True)

SEED = 42
STEP = 0.05
VAR = "base_cm_dv"
THR = np.arange(0.01, 0.51, 0.005)
JOBS = ["death_30d", "composite_30d"]


def rk(x):
    return rankdata(x) / (len(x) + 1.0)


def grid3():
    g = []
    n = int(round(1.0 / STEP))
    for i in range(n + 1):
        for j in range(n - i + 1):
            a, b = i * STEP, j * STEP
            g.append((a, b, max(1 - a - b, 0.0)))
    return g


G3 = grid3()


def wf(cols, w):
    s = np.zeros(len(cols[0]))
    for c, wi in zip(cols, w):
        s = s + wi * c
    return s


def oof_w2(a, b, y, folds):
    o = np.zeros(len(y))
    for tr, te in folds:
        best, bw = None, None
        for w in np.arange(0, 1.0001, STEP):
            s = w * a[tr] + (1 - w) * b[tr]
            v = roc_auc_score(y[tr], s)
            if best is None or v > best:
                best, bw = v, w
        o[te] = bw * a[te] + (1 - bw) * b[te]
    return o


def oof_w3(cols, y, folds):
    o = np.zeros(len(y))
    for tr, te in folds:
        best, bw = None, None
        for w in G3:
            v = roc_auc_score(
                y[tr],
                wf([c[tr] for c in cols], w))
            if best is None or v > best:
                best, bw = v, w
        o[te] = wf([c[te] for c in cols], bw)
    return o


def platt(y, p, folds):
    o = np.zeros(len(y))
    X = np.asarray(p, dtype=float).reshape(
        -1, 1)
    for tr, te in folds:
        m = LogisticRegression(max_iter=1000)
        m.fit(X[tr], y[tr])
        o[te] = m.predict_proba(X[te])[:, 1]
    return o


def nb(y, p, th):
    pr = (p >= th)
    tp = int((pr & (y == 1)).sum())
    fp = int((pr & (y == 0)).sum())
    n = len(y)
    return tp / n - (fp / n) * (th / (1 - th))


lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv"
).drop_duplicates("hadm_id")
age = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v6.csv",
    usecols=["hadm_id", "age_at_admit"])
age = age.dropna().drop_duplicates("hadm_id")
sp = v7[["hadm_id", "mean_hr", "mean_sbp",
         "mean_spo2", "cancer", "copd",
         "heart_failure"]].merge(
    age, on="hadm_id", how="left")
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
sp["spesi"] = t
sp = sp[["hadm_id", "spesi"]]

summ = []
for ycol in JOBS:
    pe = pd.read_csv(
        DATA + "/p_ehr_harm_" + ycol
        + ".csv")[["hadm_id", "p_ehr"]]
    pg = pd.read_csv(
        DATA + "/p_ecg_harm_" + ycol
        + ".csv")[["hadm_id", "p_ecg"]]
    pc = pd.read_csv(
        DATA + "/p_ctpa_pres_" + VAR + "_"
        + ycol + ".csv")[["hadm_id",
                          "p_ctpa"]]
    d = lab.merge(pe, on="hadm_id")
    d = d.merge(pg, on="hadm_id")
    d = d.merge(pc, on="hadm_id")
    d = d.merge(sp, on="hadm_id", how="left")
    d = d.reset_index(drop=True)

    y = d[ycol].values.astype(int)
    grp = d["subject_id"].values
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    folds = list(cv.split(
        np.zeros((len(y), 1)), y, grp))

    a = rk(d["p_ehr"].values)
    b = rk(d["p_ecg"].values)
    c = rk(d["p_ctpa"].values)
    w2 = oof_w2(a, b, y, folds)
    w3 = oof_w3([a, b, c], y, folds)

    sv = d["spesi"].fillna(
        d["spesi"].median()).values
    cal_e = platt(y, a, folds)
    cal_2 = platt(y, w2, folds)
    cal_3 = platt(y, w3, folds)
    cal_s = platt(y, sv, folds)

    rows = []
    for th in THR:
        rows.append({
            "outcome": ycol, "thr": th,
            "EHR": nb(y, cal_e, th),
            "WMEAN2": nb(y, cal_2, th),
            "WMEAN3": nb(y, cal_3, th),
            "sPESI": nb(y, cal_s, th),
            "treat_all": nb(
                y, np.ones(len(y)), th),
            "treat_none": 0.0})
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT + "/dca_" + ycol + ".csv",
              index=False)

    print("")
    print("=" * 60)
    print("%s  n=%d ev=%d prev %.4f"
          % (ycol, len(y), int(y.sum()),
             y.mean()))
    print("  AUC  EHR %.4f  WMEAN2 %.4f"
          "  WMEAN3 %.4f  sPESI %.4f"
          % (roc_auc_score(y, a),
             roc_auc_score(y, w2),
             roc_auc_score(y, w3),
             roc_auc_score(y, sv)))
    print("")
    print("  %6s %9s %9s %9s %9s %9s"
          % ("thr", "EHR", "WMEAN2",
             "WMEAN3", "sPESI", "all"))
    for th in [0.05, 0.10, 0.15, 0.20,
               0.30, 0.40]:
        r = t2.iloc[
            (t2["thr"] - th).abs().argmin()]
        print("  %6.2f %9.4f %9.4f %9.4f"
              " %9.4f %9.4f"
              % (r["thr"], r["EHR"],
                 r["WMEAN2"], r["WMEAN3"],
                 r["sPESI"], r["treat_all"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t2["thr"], t2["WMEAN3"],
            lw=2.2, label="WMEAN3")
    ax.plot(t2["thr"], t2["WMEAN2"],
            lw=1.8, ls="--", label="WMEAN2")
    ax.plot(t2["thr"], t2["EHR"], lw=1.5,
            ls=":", label="EHR alone")
    ax.plot(t2["thr"], t2["sPESI"], lw=1.8,
            color="tab:red", label="sPESI")
    ax.plot(t2["thr"], t2["treat_all"],
            lw=1.2, color="grey",
            label="Treat all")
    ax.axhline(0, lw=1.2, color="black",
               label="Treat none")
    lo = float(min(t2["WMEAN3"].min(),
                   t2["sPESI"].min()))
    ax.set_ylim(max(lo, -0.05),
                float(t2["WMEAN3"].max())
                * 1.15)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve: %s"
                 " (n=%d, %d events)"
                 % (ycol, len(y),
                    int(y.sum())))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fn = OUT + "/dca_" + ycol + ".png"
    plt.savefig(fn, dpi=160)
    plt.close()
    print("  figure:", fn)

    gain = (t2["WMEAN3"]
            - t2[["sPESI", "treat_all"]].max(
                axis=1))
    summ.append({
        "outcome": ycol,
        "auc_wmean3": roc_auc_score(y, w3),
        "auc_wmean2": roc_auc_score(y, w2),
        "peak_gain_thr": float(
            t2.loc[gain.idxmax(), "thr"]),
        "peak_gain": float(gain.max())})

s = pd.DataFrame(summ)
s.to_csv(OUT + "/dca_summary.csv",
         index=False)
print("")
print(s.round(4).to_string(index=False))
print("saved to", OUT)
