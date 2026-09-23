#!/usr/bin/env python3
"""Draft decision curve analysis, not used for any
reported result. It expects outcome and calibrated
columns that the script-104 output files do not
contain. The reported decision curves come from
145_dca_plot.py."""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "./fusion_workspace/data"
EHRF = os.path.join(DATA, "mimic_pe_ehr_baseline_v7.csv")
OUTCOMES = ["composite_30d", "death_30d"]
THRESH = np.arange(0.01, 0.51, 0.01)
KEYTH = [0.05, 0.10, 0.15, 0.20, 0.30]
NBOOT = 2000
SEED = 42


def pick(df, want):
    """Find a column by exact then substring match."""
    for c in df.columns:
        if c.lower() == want:
            return c
    for c in df.columns:
        if want in c.lower():
            return c
    return None


def looks_calibrated(p):
    """Rank scores are near-uniform; probs are not."""
    p = np.asarray(p, dtype=float)
    if p.min() < 0.0 or p.max() > 1.0:
        return False
    return abs(np.mean(p) - 0.5) > 0.05


def oof_platt(score, y, groups, seed=SEED):
    """Out-of-fold Platt scaling of a 1-D score."""
    y = np.asarray(y)
    x = np.asarray(score, dtype=float).reshape(-1, 1)
    p = np.zeros(len(y), dtype=float)
    sgkf = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(x, y, groups):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(x[tr], y[tr])
        p[te] = lr.predict_proba(x[te])[:, 1]
    return p


def net_benefit(y, p, t):
    """Vickers net benefit at threshold t."""
    n = len(y)
    if n == 0:
        return np.nan
    pos = p >= t
    tp = np.sum(pos & (y == 1))
    fp = np.sum(pos & (y == 0))
    return tp / n - (fp / n) * (t / (1.0 - t))


def nb_all(y, t):
    """Net benefit of treating everyone."""
    prev = np.mean(y)
    return prev - (1.0 - prev) * (t / (1.0 - t))


def build_spesi6(df):
    """Six-criterion sPESI from the v7 baseline."""
    s = np.zeros(len(df), dtype=float)
    s += (df["age_at_admit"] > 80).astype(float)
    s += (df["mean_hr"] >= 110).astype(float)
    s += (df["mean_sbp"] < 100).astype(float)
    s += (df["mean_spo2"] < 90).astype(float)
    s += (df["cancer"] > 0).astype(float)
    cp = (df["heart_failure"] > 0) | (df["copd"] > 0)
    s += cp.astype(float)
    return s


def curve(y, p, thresholds):
    return np.array([net_benefit(y, p, t) for t in thresholds])


def boot_delta(y, pa, pb, groups, thresholds, nboot=NBOOT):
    """Bootstrap CI for NB(a) - NB(b) by subject."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = np.zeros((nboot, len(thresholds)))
    for b in range(nboot):
        pickg = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in pickg])
        yb = y[rows]
        if yb.sum() == 0 or yb.sum() == len(yb):
            out[b, :] = np.nan
            continue
        ca = curve(yb, pa[rows], thresholds)
        cb = curve(yb, pb[rows], thresholds)
        out[b, :] = ca - cb
    lo = np.nanpercentile(out, 2.5, axis=0)
    hi = np.nanpercentile(out, 97.5, axis=0)
    pgt = np.nanmean(out > 0, axis=0)
    return lo, hi, pgt


# ---- main ----

ehr = pd.read_csv(EHRF)
ehr["spesi6"] = build_spesi6(ehr)
keep = ["hadm_id", "spesi6"]
ehr = ehr[keep].drop_duplicates("hadm_id")

for outcome in OUTCOMES:
    fn = os.path.join(
        DATA, "p_wmean3_ctpa_%s.csv" % outcome)
    if not os.path.exists(fn):
        print("MISSING: %s -- skipping" % fn)
        continue
    df = pd.read_csv(fn)
    df = df.merge(ehr, on="hadm_id", how="left")
    print("\n=== %s: %d rows ===" % (outcome, len(df)))
    print("columns:", list(df.columns))

    ycol = pick(df, "label")
    if ycol is None:
        ycol = pick(df, outcome)
    y = df[ycol].astype(int).values
    grp = df["subject_id"].values
    print("events %d / %d (%.3f)"
          % (y.sum(), len(y), y.mean()))

    models = {}
    for tag, want in [("WMEAN3", "wmean3_cal"),
                      ("WMEAN2", "wmean2_cal"),
                      ("EHR", "ehr_cal")]:
        c = pick(df, want)
        if c is None:
            print("no calibrated col for %s" % tag)
            continue
        if not looks_calibrated(df[c].values):
            print("%s looks uncalibrated -- skipped" % tag)
            continue
        models[tag] = df[c].values.astype(float)

    df["spesi6"] = df["spesi6"].fillna(
        df["spesi6"].median())
    models["sPESI6"] = oof_platt(
        df["spesi6"].values, y, grp)

    if "WMEAN3" not in models:
        print("no WMEAN3 -- cannot run DCA")
        continue

    rows = []
    for t in THRESH:
        rec = {"threshold": t,
               "treat_all": nb_all(y, t),
               "treat_none": 0.0}
        for tag, p in models.items():
            rec["nb_" + tag] = net_benefit(y, p, t)
        rows.append(rec)
    res = pd.DataFrame(rows)
    rf = os.path.join(
        DATA, "dca_curve_%s.csv" % outcome)
    res.to_csv(rf, index=False)
    print("wrote", rf)

    ref = models["sPESI6"]
    lo, hi, pgt = boot_delta(
        y, models["WMEAN3"], ref, grp, np.array(KEYTH))
    drows = []
    for i, t in enumerate(KEYTH):
        d = (net_benefit(y, models["WMEAN3"], t)
             - net_benefit(y, ref, t))
        drows.append({"threshold": t, "delta_nb": d,
                      "lo": lo[i], "hi": hi[i],
                      "p_gt0": pgt[i]})
        print("t=%.2f dNB=%+.4f [%+.4f,%+.4f] P=%.3f"
              % (t, d, lo[i], hi[i], pgt[i]))
    df_d = pd.DataFrame(drows)
    dfn = os.path.join(
        DATA, "dca_delta_%s.csv" % outcome)
    df_d.to_csv(dfn, index=False)
    print("wrote", dfn)

    plt.figure(figsize=(7, 5))
    plt.plot(res["threshold"], res["treat_all"],
             "k--", lw=1, label="Treat all")
    plt.axhline(0, color="k", lw=1, label="Treat none")
    for tag in models:
        plt.plot(res["threshold"], res["nb_" + tag],
                 lw=2, label=tag)
    ymax = max(0.02, float(res["treat_all"].max()))
    plt.ylim(-0.05, ymax * 1.3)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve: %s" % outcome)
    plt.legend()
    plt.grid(alpha=0.3)
    pf = os.path.join(DATA, "dca_%s.png" % outcome)
    plt.savefig(pf, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", pf)

print("\nDONE")
