import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve
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
    aucs = (
        tzr[:, :m].sum(axis=1) - m * (m + 1) / 2
    ) / (m * n)
    v01 = (tzr[:, :m] - txr) / n
    v10 = 1 - (tzr[:, m:] - tyr) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    L = np.array([1.0, -1.0])
    var = L.dot(S).dot(L)
    if var <= 0:
        return aucs, np.nan, np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs, z, p


def binary_stats(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens, spec, (tp, fp, tn, fn)


def spec_at_sens(y, p, target):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target - 1e-12)[0]
    if len(ok) == 0:
        return np.nan
    return 1.0 - fpr[ok[0]]


def rh2_boot(y, p_model, sp_bin, g,
             n=NBOOT):
    rng = np.random.RandomState(SEED)
    g = np.asarray(g)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    d_spec = []
    s_sens = []
    for _ in range(n):
        s = rng.choice(uq, len(uq),
                       replace=True)
        t = np.concatenate([idx[u] for u in s])
        yy = y[t]
        if len(np.unique(yy)) < 2:
            continue
        se, sp, _ = binary_stats(
            yy, sp_bin[t]
        )
        if se <= 0 or se >= 1:
            continue
        sm = spec_at_sens(yy, p_model[t], se)
        if np.isnan(sm):
            continue
        d_spec.append(sm - sp)
        s_sens.append(se)
    d = np.array(d_spec)
    return (
        d.mean(), np.median(d),
        np.percentile(d, [2.5, 97.5]),
        (d > 0).mean(), np.mean(s_sens),
        len(d),
    )


def report(tag, y, g, models, sp_bin):
    print("\n" + "=" * 66)
    print(tag)
    print("=" * 66)
    print("n=%d  events=%d  rate=%.4f"
          % (len(y), int(y.sum()), y.mean()))

    se, sp, cm = binary_stats(y, sp_bin)
    print("\nsPESI (>=1): sens %.3f spec %.3f"
          "  TP%d FP%d TN%d FN%d"
          % (se, sp, cm[0], cm[1],
             cm[2], cm[3]))

    print("\n-- AUC vs sPESI --")
    spc = sp_bin.astype(float)
    for k, v in models.items():
        a, z, p = delong(y, v, spc)
        print("  %-6s AUC %.4f  vs sPESI "
              "%.4f  DeLong z=%.2f p=%.4f"
              % (k, roc_auc_score(y, v),
                 roc_auc_score(y, spc), z, p))

    print("\n-- RH2: specificity at matched "
          "sensitivity (corrected) --")
    for k, v in models.items():
        pt = spec_at_sens(y, v, se)
        m, md, ci, frac, ms, nb = rh2_boot(
            y, v, sp_bin, g
        )
        print("  %-6s point spec %.3f "
              "(sPESI %.3f, gain %+0.3f)"
              % (k, pt, sp, pt - sp))
        print("         boot mean %+0.4f "
              "median %+0.4f "
              "[%+0.4f, %+0.4f] "
              "P(>0)=%.3f  n=%d"
              % (m, md, ci[0], ci[1],
                 frac, nb))

    print("\n-- confusion at matched "
          "sensitivity --")
    for k, v in models.items():
        fpr, tpr, thr = roc_curve(y, v)
        ok = np.where(tpr >= se - 1e-12)[0]
        if len(ok) == 0:
            continue
        t = thr[ok[0]]
        pred = (v >= t).astype(int)
        s2, sp2, cm2 = binary_stats(y, pred)
        print("  %-6s thr %.4f sens %.3f "
              "spec %.3f  TP%d FP%d TN%d FN%d"
              % (k, t, s2, sp2, cm2[0],
                 cm2[1], cm2[2], cm2[3]))


print("Loading three-way table...")
d3 = pd.read_csv(
    os.path.join(FD, "ro4_fusion_table.csv")
)
y3 = d3["label"].values
g3 = d3["subject_id"].values
sp3 = (d3["spesi"].fillna(0) >= 1).astype(
    int).values
m3 = {
    "MEAN3": d3["MEAN3"].values,
    "MEAN2": d3["MEAN2"].values,
}
report("THREE-WAY COHORT (index CXR present)",
       y3, g3, m3, sp3)

print("\n\nBuilding two-way cohort...")
ecg = pd.read_csv(
    os.path.join(P2, "logit_final_pecg.csv")
)
ecga = ecg.groupby(KEY).agg(
    p_ecg=("p_ecg", "mean")
).reset_index()
eh = pd.read_csv(os.path.join(
    FD, "p_ehr_ensemble_v7_vrex.csv"
))[KEY + ["p_ehr_mean", "mace_30d_label"]]
v6 = pd.read_csv(
    os.path.join(
        P2, "mimic_pe_ehr_baseline_v6.csv"
    )
).drop_duplicates(KEY)
v6["spesi"] = (
    (v6["age_at_admit"] > 80).astype(int)
    + (v6["cancer"] > 0).astype(int)
    + (((v6["heart_failure"] > 0)
        | (v6["copd"] > 0))).astype(int)
    + (v6["mean_hr"] >= 110).fillna(
        False).astype(int)
    + (v6["mean_sbp"] < 100).fillna(
        False).astype(int)
    + (v6["mean_spo2"] < 90).fillna(
        False).astype(int)
)

d2 = eh.merge(ecga, on=KEY, how="inner")
d2 = d2.merge(v6[KEY + ["spesi"]], on=KEY,
              how="left")
d2 = d2.dropna(subset=[
    "p_ehr_mean", "p_ecg", "mace_30d_label"
]).reset_index(drop=True)
d2["r_ehr"] = rank01(d2["p_ehr_mean"])
d2["r_ecg"] = rank01(d2["p_ecg"])
d2["MEAN2"] = (d2["r_ehr"] + d2["r_ecg"]) / 2

y2 = d2["mace_30d_label"].values
g2 = d2["subject_id"].values
sp2 = (d2["spesi"].fillna(0) >= 1).astype(
    int).values
report("TWO-WAY COHORT (all admissions)",
       y2, g2,
       {"MEAN2": d2["MEAN2"].values}, sp2)

d2.to_csv(
    os.path.join(FD, "rh2_twoway_table.csv"),
    index=False
)
print("\nSaved rh2_twoway_table.csv")
