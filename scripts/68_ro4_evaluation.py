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
NBOOT = 1000
EHR_VAR = "vrex"


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
    P = [np.asarray(p1), np.asarray(p2)]
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


def boot(y, pa, pb, g, fn, n=NBOOT):
    rng = np.random.RandomState(SEED)
    g = np.asarray(g)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    out = []
    for _ in range(n):
        s = rng.choice(uq, len(uq), replace=True)
        t = np.concatenate([idx[u] for u in s])
        if len(np.unique(y[t])) < 2:
            continue
        out.append(fn(y[t], pa[t]) - fn(y[t], pb[t]))
    o = np.array(out)
    return o.mean(), np.percentile(o, [2.5, 97.5])


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return thr[j]


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-9)
    return {
        "thr": t, "TP": tp, "FP": fp,
        "TN": tn, "FN": fn, "sens": sens,
        "spec": spec, "ppv": ppv, "f1": f1,
    }


def spec_at_sens(y, p, target):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:
        return np.nan
    i = ok[0]
    return 1 - fpr[i]


print("Loading inputs...")
tim = pd.read_csv(
    os.path.join(FD, "cxr_image_timing.csv")
)
dur = tim[tim["rel"] == "during"]
cxa = dur.groupby(KEY).agg(
    p_cxr=("p_cxr", "mean"),
    frac_port=("port", "mean"),
    n_dur=("p_cxr", "size"),
    label=("mace_30d_label", "max"),
).reset_index()
print("  admissions with index CXR:", len(cxa))

ecg = pd.read_csv(
    os.path.join(P2, "logit_final_pecg.csv")
)
ecga = ecg.groupby(KEY).agg(
    p_ecg=("p_ecg", "mean")
).reset_index()

eh = pd.read_csv(os.path.join(
    FD, "p_ehr_ensemble_v7_%s.csv" % EHR_VAR
))[KEY + ["p_ehr_mean"]]

v6 = pd.read_csv(
    os.path.join(P2, "mimic_pe_ehr_baseline_v6.csv")
).drop_duplicates(KEY)
sp = v6[KEY + [
    "age_at_admit", "mean_hr", "mean_sbp",
    "mean_spo2", "cancer", "heart_failure",
    "copd",
]].copy()
sp["spesi"] = (
    (sp["age_at_admit"] > 80).astype(int)
    + (sp["cancer"] > 0).astype(int)
    + (((sp["heart_failure"] > 0)
        | (sp["copd"] > 0))).astype(int)
    + (sp["mean_hr"] >= 110).fillna(
        False).astype(int)
    + (sp["mean_sbp"] < 100).fillna(
        False).astype(int)
    + (sp["mean_spo2"] < 90).fillna(
        False).astype(int)
)

d = cxa.merge(ecga, on=KEY, how="inner")
d = d.merge(eh, on=KEY, how="inner")
d = d.merge(sp[KEY + ["spesi"]], on=KEY,
            how="left")
d = d.dropna(subset=[
    "p_ehr_mean", "p_ecg", "p_cxr"
]).reset_index(drop=True)

y = d["label"].values
g = d["subject_id"].values
print("  cohort:", len(d),
      "events:", int(y.sum()),
      "rate: %.4f" % y.mean())

d["r_ehr"] = rank01(d["p_ehr_mean"])
d["r_ecg"] = rank01(d["p_ecg"])
d["r_cxr"] = rank01(d["p_cxr"])
d["r_port"] = rank01(d["frac_port"])
d["r_ndur"] = rank01(d["n_dur"])
d["MEAN3"] = (
    d["r_ehr"] + d["r_ecg"] + d["r_cxr"]
) / 3
d["MEAN2"] = (d["r_ehr"] + d["r_ecg"]) / 2

M = {
    "EHR": d["r_ehr"].values,
    "ECG": d["r_ecg"].values,
    "CXR": d["r_cxr"].values,
    "META": ((d["r_port"] + d["r_ndur"]) / 2
             ).values,
    "sPESI": d["spesi"].fillna(0).values,
    "MEAN2": d["MEAN2"].values,
    "MEAN3": d["MEAN3"].values,
}

print("\n" + "=" * 66)
print("1. DISCRIMINATION (n=%d, events=%d)"
      % (len(d), int(y.sum())))
print("=" * 66)
print("%-8s %-8s %-8s" % ("model", "AUC", "AUPRC"))
for k, v in M.items():
    print("%-8s %.4f   %.4f"
          % (k, roc_auc_score(y, v),
             average_precision_score(y, v)))

best = max(["EHR", "ECG", "CXR"],
           key=lambda k: roc_auc_score(y, M[k]))
print("\nbest unimodal:", best)

print("\n" + "=" * 66)
print("2. MEAN3 vs each modality")
print("=" * 66)
for k in ["EHR", "ECG", "CXR", "META", "sPESI"]:
    m, ci = boot(y, M["MEAN3"], M[k], g,
                 roc_auc_score)
    a, z, p = delong(y, M["MEAN3"], M[k])
    print("vs %-6s boot %+0.4f [%+0.4f,%+0.4f]"
          "  DeLong z=%.2f p=%.4f"
          % (k, m, ci[0], ci[1], z, p))

print("\nCXR increment (MEAN3 vs MEAN2):")
m, ci = boot(y, M["MEAN3"], M["MEAN2"], g,
             roc_auc_score)
a, z, p = delong(y, M["MEAN3"], M["MEAN2"])
print("  %+0.4f [%+0.4f, %+0.4f]  "
      "DeLong p=%.4f" % (m, ci[0], ci[1], p))

print("\n" + "=" * 66)
print("3. THRESHOLD METRICS (Youden)")
print("=" * 66)
rows = []
for k in ["EHR", "ECG", "CXR", "MEAN2",
          "MEAN3"]:
    t = youden(y, M[k])
    r = metrics_at(y, M[k], t)
    r["model"] = k
    rows.append(r)
    print("%-6s thr %.3f  sens %.3f spec %.3f "
          "ppv %.3f f1 %.3f  "
          "TP%d FP%d TN%d FN%d"
          % (k, r["thr"], r["sens"], r["spec"],
             r["ppv"], r["f1"], r["TP"],
             r["FP"], r["TN"], r["FN"]))

print("\n" + "=" * 66)
print("4. RH2: vs sPESI on SPECIFICITY")
print("=" * 66)
sp_pred = (d["spesi"].fillna(0) >= 1).astype(
    int).values
rs = metrics_at(y, sp_pred.astype(float), 0.5)
print("sPESI (>=1 high risk): sens %.3f "
      "spec %.3f  TP%d FP%d TN%d FN%d"
      % (rs["sens"], rs["spec"], rs["TP"],
         rs["FP"], rs["TN"], rs["FN"]))
tgt = rs["sens"]
for k in ["MEAN2", "MEAN3"]:
    s = spec_at_sens(y, M[k], tgt)
    print("%-6s spec at matched sens %.3f: "
          "%.3f  (gain %+0.3f)"
          % (k, tgt, s, s - rs["spec"]))


def spec_fn_factory(target):
    def f(yy, pp):
        return spec_at_sens(yy, pp, target)
    return f


f = spec_fn_factory(tgt)
m, ci = boot(y, M["MEAN3"],
             sp_pred.astype(float), g, f)
print("\nMEAN3 spec - sPESI spec (bootstrap):")
print("  %+0.4f [%+0.4f, %+0.4f]"
      % (m, ci[0], ci[1]))

print("\n" + "=" * 66)
print("5. CALIBRATION (Platt, 5-fold OOF)")
print("=" * 66)
skf = StratifiedGroupKFold(
    n_splits=5, shuffle=True,
    random_state=SEED
)
for k in ["MEAN3", "MEAN2"]:
    x = M[k].reshape(-1, 1)
    cal = np.zeros(len(d))
    for tr, te in skf.split(x, y, groups=g):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(x[tr], y[tr])
        cal[te] = lr.predict_proba(x[te])[:, 1]
    b0 = brier_score_loss(y, M[k])
    b1 = brier_score_loss(y, cal)
    print("%-6s Brier raw %.4f -> "
          "calibrated %.4f" % (k, b0, b1))
    d["cal_" + k] = cal
    bins = pd.qcut(cal, 10, duplicates="drop")
    cb = pd.DataFrame({
        "bin": bins, "y": y, "p": cal
    }).groupby("bin", observed=True).agg(
        n=("y", "size"),
        obs=("y", "mean"),
        pred=("p", "mean"),
    )
    print(cb.to_string())

print("\n" + "=" * 66)
print("6. SECONDARY: EHR+ECG on ALL admissions")
print("=" * 66)
d2 = ecga.merge(eh, on=KEY, how="inner")
lab = pd.read_csv(
    os.path.join(P2, "mimic_pe_mace_cohort.csv")
)[KEY + ["mace_30d_label"]].drop_duplicates(KEY)
d2 = d2.merge(lab, on=KEY, how="left")
d2 = d2.dropna(subset=[
    "p_ehr_mean", "p_ecg", "mace_30d_label"
]).reset_index(drop=True)
y2 = d2["mace_30d_label"].values
g2 = d2["subject_id"].values
d2["r_ehr"] = rank01(d2["p_ehr_mean"])
d2["r_ecg"] = rank01(d2["p_ecg"])
d2["M2"] = (d2["r_ehr"] + d2["r_ecg"]) / 2
print("n=%d events=%d" % (len(d2), int(y2.sum())))
for k in ["r_ehr", "r_ecg", "M2"]:
    print("  %-6s AUC %.4f  AP %.4f"
          % (k, roc_auc_score(y2, d2[k]),
             average_precision_score(
                 y2, d2[k])))
for k in ["r_ehr", "r_ecg"]:
    m, ci = boot(y2, d2["M2"].values,
                 d2[k].values, g2,
                 roc_auc_score)
    a, z, p = delong(y2, d2["M2"].values,
                     d2[k].values)
    print("  M2 vs %-6s %+0.4f "
          "[%+0.4f,%+0.4f] DeLong p=%.4f"
          % (k, m, ci[0], ci[1], p))

pd.DataFrame(rows).to_csv(
    os.path.join(FD, "ro4_threshold_metrics.csv"),
    index=False
)
d.to_csv(
    os.path.join(FD, "ro4_fusion_table.csv"),
    index=False
)
print("\nSaved ro4_threshold_metrics.csv and "
      "ro4_fusion_table.csv")
