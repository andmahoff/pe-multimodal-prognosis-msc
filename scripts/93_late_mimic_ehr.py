import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
W3 = [
    (a / 10.0, b / 10.0, c / 10.0)
    for a in range(11)
    for b in range(11)
    for c in range(11)
    if a + b + c == 10
]

EHR = [
    "mean_hr", "mean_sbp", "mean_dbp",
    "mean_rr", "mean_temp", "mean_troponin",
    "mean_wbc", "mean_hemoglobin",
    "mean_creatinine", "mean_albumin", "nlr",
    "afib", "cancer", "copd",
    "heart_failure",
]


def norm_path(p):
    return "/".join(
        str(p).rstrip("/").split("/")[-4:])


def rank01(x):
    return pd.Series(x).rank(pct=True).values


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


def oof(X, y, g, skf, kind):
    X = np.asarray(X, dtype=float)
    out = np.zeros(len(y))
    for tr, te in skf.split(X, y, groups=g):
        if kind == "lr":
            im = SimpleImputer(strategy="median")
            sc = StandardScaler()
            a = sc.fit_transform(
                im.fit_transform(X[tr]))
            b = sc.transform(im.transform(X[te]))
            m = LogisticRegression(
                max_iter=5000, C=1.0,
                class_weight="balanced")
        else:
            a, b = X[tr], X[te]
            m = HistGradientBoostingClassifier(
                random_state=SEED, max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0)
        m.fit(a, y[tr])
        out[te] = m.predict_proba(b)[:, 1]
    return out


def wfuse2(a, b, y, g, skf):
    A = np.vstack([a, b]).T
    out = np.zeros(len(y))
    for tr, te in skf.split(A, y, groups=g):
        bw, ba = 0.5, -1
        for w in W2:
            v = roc_auc_score(
                y[tr], w * a[tr] + (1 - w) * b[tr])
            if v > ba:
                ba, bw = v, w
        out[te] = bw * a[te] + (1 - bw) * b[te]
    return out


def wfuse3(a, b, c, y, g, skf):
    A = np.vstack([a, b, c]).T
    out = np.zeros(len(y))
    for tr, te in skf.split(A, y, groups=g):
        bt, ba = (0.4, 0.3, 0.3), -1
        for t3 in W3:
            v = roc_auc_score(
                y[tr],
                t3[0] * a[tr] + t3[1] * b[tr]
                + t3[2] * c[tr])
            if v > ba:
                ba, bt = v, t3
        out[te] = (bt[0] * a[te] + bt[1] * b[te]
                   + bt[2] * c[te])
    return out


print("Assembling...")
v7 = pd.read_csv(
    os.path.join(
        P2, "mimic_pe_ehr_baseline_v7.csv"))
if "mean_troponin_t" in v7.columns:
    v7 = v7.rename(columns={
        "mean_troponin_t": "mean_troponin"})
v7 = v7[KEY + EHR].drop_duplicates(KEY)

bf = pd.read_csv(
    os.path.join(P2, "bench_feats_logit.csv"))
bf["_k"] = bf["ecg_path"].map(norm_path)
bf = bf.drop_duplicates("_k")
coh = pd.read_csv(
    os.path.join(
        P2, "mimic_pe_mace_cohort.csv"))[
    KEY + ["ecg_path"]]
coh["_k"] = coh["ecg_path"].map(norm_path)
coh = coh.drop_duplicates("_k")
bf = bf.merge(coh[["_k"] + KEY], on="_k",
              how="inner")
FEC = [c for c in bf.columns
       if c.startswith("feat_")]
ecgf = bf.groupby(KEY)[FEC].mean().reset_index()

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
    bl = "death_30d" if lc == \
        "death_30d_inhosp" else lc
    eh = pd.read_csv(os.path.join(
        FD, "p_ehr_harm_%s.csv" % bl))
    ec = pd.read_csv(os.path.join(
        FD, "p_ecg_harm_%s.csv" % lc))
    cx = pd.read_csv(os.path.join(
        FD, "p_cxr_harm_%s.csv" % bl))

    d = lab.merge(v7, on=KEY, how="inner")
    d = d.merge(ecgf, on=KEY, how="inner")
    d = d.merge(eh, on=KEY, how="inner")
    d = d.merge(ec, on=KEY, how="inner")
    d = d.merge(cx, on=KEY, how="left")
    if lc == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)
    d["_cxr"] = d["p_cxr"].fillna(
        d["p_cxr"].median())

    y = d[lc].values.astype(int)
    g = d["subject_id"].values
    skf = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)

    zs = rank01(d["p_ehr"])
    rc = rank01(d["p_ecg"])
    rx = rank01(d["_cxr"])

    print("\n" + "=" * 66)
    print("%s   n=%d ev=%d (%.4f)"
          % (nm, len(d), int(y.sum()),
             y.mean()))
    print("=" * 66)

    best_k, best_a, mi = None, -1, None
    for kind in ["lr", "gb"]:
        p = oof(d[EHR].values, y, g, skf, kind)
        a = roc_auc_score(y, p)
        print("  MIMIC-EHR (%s)   AUC %.4f"
              % (kind, a))
        if a > best_a:
            best_a, best_k, mi = a, kind, p
    rm = rank01(mi)
    print("  using %s for MIMIC-EHR" % best_k)
    print("  zero-shot EHR     AUC %.4f"
          % roc_auc_score(y, zs))
    print("  ECG               AUC %.4f"
          % roc_auc_score(y, rc))

    L_zs = wfuse2(zs, rc, y, g, skf)
    L_mi = wfuse2(rm, rc, y, g, skf)
    L_mi3 = wfuse3(rm, rc, rx, y, g, skf)

    E2 = oof(d[EHR + FEC].values, y, g,
             skf, best_k)
    E3 = oof(d[EHR + FEC + ["_cxr"]].values,
             y, g, skf, best_k)

    M = {
        "MIMIC-EHR only": rm,
        "LATE zs (EHR+ECG)": L_zs,
        "LATE mimic (EHR+ECG)": L_mi,
        "LATE mimic (+CXR)": L_mi3,
        "EARLY (EHR+ECG)": E2,
        "EARLY (+CXR)": E3,
    }
    print()
    for k, v in M.items():
        print("  %-22s AUC %.4f  AP %.4f"
              % (k, roc_auc_score(y, v),
                 average_precision_score(y, v)))

    print("\n  Late vs early, same training data:")
    print("  LATE mimic vs EARLY (EHR+ECG):")
    m, ci = boot(y, M["LATE mimic (EHR+ECG)"],
                 M["EARLY (EHR+ECG)"], g)
    s = "*" if (ci[0] > 0 or ci[1] < 0) else " "
    print("    %+0.4f [%+0.4f, %+0.4f] %s"
          % (m, ci[0], ci[1], s))
    print("  LATE mimic +CXR vs EARLY +CXR:")
    m, ci = boot(y, M["LATE mimic (+CXR)"],
                 M["EARLY (+CXR)"], g)
    s = "*" if (ci[0] > 0 or ci[1] < 0) else " "
    print("    %+0.4f [%+0.4f, %+0.4f] %s"
          % (m, ci[0], ci[1], s))

    print("\n  ECG gain over a matched base:")
    for k in ["LATE mimic (EHR+ECG)",
              "EARLY (EHR+ECG)"]:
        m, ci = boot(y, M[k], rm, g)
        s = "*" if ci[0] > 0 else " "
        print("    %-22s %+0.4f "
              "[%+0.4f, %+0.4f] %s"
              % (k, m, ci[0], ci[1], s))

print("\nDone.")
