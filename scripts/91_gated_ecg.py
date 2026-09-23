import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
KEY = ["subject_id", "hadm_id"]
SEED = 42
NBOOT = 2000
W2 = np.arange(0.0, 1.01, 0.05)


def rank01(x):
    return pd.Series(x).rank(pct=True).values


def z(x):
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / (
        s if s > 0 else 1)


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
    X = np.nan_to_num(
        np.asarray(X, dtype=float), nan=0.0)
    out = np.zeros(len(y))
    cf = []
    for tr, te in skf.split(X, y, groups=g):
        lr = LogisticRegression(
            max_iter=5000, C=1.0)
        lr.fit(X[tr], y[tr])
        out[te] = lr.predict_proba(X[te])[:, 1]
        cf.append(lr.coef_[0])
    return out, np.mean(cf, axis=0)


print("Building ECG context vars...")
coh = pd.read_csv(
    os.path.join(
        P2, "mimic_pe_mace_cohort.csv"))
coh["at"] = pd.to_datetime(
    coh["admittime"], errors="coerce")
coh["ct"] = pd.to_datetime(
    coh["ecg_charttime"], errors="coerce")
coh["delay"] = (
    coh["ct"] - coh["at"]
).dt.total_seconds() / 3600.0
ctx = coh.groupby(KEY).agg(
    n_ecg=("ecg_path", "nunique"),
    delay_min=("delay", "min"),
    delay_med=("delay", "median"),
).reset_index()
print("  admissions:", len(ctx))
print(ctx[["n_ecg", "delay_min"]].describe(
).to_string())

lab = pd.read_csv(
    os.path.join(
        FD, "mimic_labels_harmonised.csv"))

TASKS = [
    ("cv_first", "CV only"),
    ("composite_30d", "composite"),
    ("death_30d", "death (dod)"),
    ("death_30d_inhosp", "death (in-hosp)"),
]

signs = {"conf": [], "necg": [],
         "delay": []}
wins = {"conf": 0, "necg": 0, "delay": 0}

for lc, nm in TASKS:
    bl = "death_30d" if lc == \
        "death_30d_inhosp" else lc
    eh = pd.read_csv(os.path.join(
        FD, "p_ehr_harm_%s.csv" % bl))
    ec = pd.read_csv(os.path.join(
        FD, "p_ecg_harm_%s.csv" % lc))

    d = lab.merge(eh, on=KEY, how="inner")
    d = d.merge(ec, on=KEY, how="inner")
    d = d.merge(ctx, on=KEY, how="left")
    if lc == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)

    y = d[lc].values.astype(int)
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
        random_state=SEED)
    A = np.vstack([re, rc]).T
    wm = np.zeros(len(y))
    for tr, te in skf.split(A, y, groups=g):
        bw, ba = 0.5, -1
        for w in W2:
            v = roc_auc_score(
                y[tr],
                w * re[tr] + (1 - w) * rc[tr])
            if v > ba:
                ba, bw = v, w
        wm[te] = (bw * re[te]
                  + (1 - bw) * rc[te])

    bz = z(re)
    cz = z(rc)
    conf = z(np.abs(bz))
    ne = z(np.log1p(
        d["n_ecg"].fillna(1).values))
    dl = z(np.clip(
        d["delay_min"].fillna(0).values,
        -24, 72))

    M = {}
    M["EHR only"] = (re, None)
    M["WMEAN2"] = (wm, None)
    p, c = oof_lr(
        np.vstack([bz, cz]).T, y, g, skf)
    M["RESID"] = (p, c)
    for key, gv in [("conf", conf),
                    ("necg", ne),
                    ("delay", dl)]:
        p, c = oof_lr(
            np.vstack([bz, cz, gv,
                       gv * cz]).T,
            y, g, skf)
        M["GATE_" + key] = (p, c)
    p, c = oof_lr(
        np.vstack([bz, cz, conf, ne, dl,
                   conf * cz, ne * cz,
                   dl * cz]).T, y, g, skf)
    M["GATE_all"] = (p, c)

    print("  %-14s %-8s %-8s"
          % ("model", "AUC", "AUPRC"))
    for k, (v, c) in M.items():
        print("  %-14s %.4f   %.4f"
              % (k, roc_auc_score(y, v),
                 average_precision_score(
                     y, v)))

    print("\n  interaction coefficients:")
    for key in ["conf", "necg", "delay"]:
        c = M["GATE_" + key][1]
        signs[key].append(c[3])
        print("    %-6s ecg %+.3f  gate %+.3f"
              "  interaction %+.3f"
              % (key, c[1], c[2], c[3]))

    print("\n  vs RESID:")
    for key in ["conf", "necg", "delay"]:
        m, ci = boot(y, M["GATE_" + key][0],
                     M["RESID"][0], g)
        s = "*" if ci[0] > 0 else " "
        if ci[0] > 0:
            wins[key] += 1
        print("    GATE_%-6s %+0.4f "
              "[%+0.4f, %+0.4f] %s"
              % (key, m, ci[0], ci[1], s))
    m, ci = boot(y, M["GATE_all"][0],
                 M["RESID"][0], g)
    print("    GATE_all    %+0.4f "
          "[%+0.4f, %+0.4f]"
          % (m, ci[0], ci[1]))

    print("\n  vs WMEAN2:")
    for k in ["RESID", "GATE_conf",
              "GATE_necg", "GATE_delay",
              "GATE_all"]:
        m, ci = boot(y, M[k][0],
                     M["WMEAN2"][0], g)
        s = "*" if ci[0] > 0 else " "
        print("    %-11s %+0.4f "
              "[%+0.4f, %+0.4f] %s"
              % (k, m, ci[0], ci[1], s))

print("\n" + "=" * 64)
print("PRE-SPECIFIED VERDICT")
print("=" * 64)
for key in ["conf", "necg", "delay"]:
    v = signs[key]
    same = (all(x > 0 for x in v)
            or all(x < 0 for x in v))
    print("  %-6s interactions %s | "
          "consistent sign %s | "
          "significant wins %d/4 | %s"
          % (key,
             np.round(v, 3), same,
             wins[key],
             "PASS" if (same and
                        wins[key] >= 2)
             else "FAIL"))
print("\nDone.")
