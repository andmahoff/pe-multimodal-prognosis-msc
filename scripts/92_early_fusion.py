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
            im = SimpleImputer(
                strategy="median")
            sc = StandardScaler()
            a = sc.fit_transform(
                im.fit_transform(X[tr]))
            b = sc.transform(
                im.transform(X[te]))
            m = LogisticRegression(
                max_iter=5000, C=1.0,
                class_weight="balanced")
        else:
            a, b = X[tr], X[te]
            m = HistGradientBoostingClassifier(
                random_state=SEED,
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0)
        m.fit(a, y[tr])
        out[te] = m.predict_proba(b)[:, 1]
    return out


print("Assembling features...")
v7 = pd.read_csv(
    os.path.join(
        P2, "mimic_pe_ehr_baseline_v7.csv"))
if "mean_troponin_t" in v7.columns:
    v7 = v7.rename(columns={
        "mean_troponin_t": "mean_troponin"})
v7 = v7[KEY + EHR].drop_duplicates(KEY)
print("  EHR:", v7.shape)

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
ecg = bf.groupby(KEY)[FEC].mean().reset_index()
print("  ECG:", ecg.shape,
      "(%d logits)" % len(FEC))

lab = pd.read_csv(
    os.path.join(
        FD, "mimic_labels_harmonised.csv"))

TASKS = [
    ("cv_first", "CV only"),
    ("composite_30d", "composite"),
    ("death_30d", "death (dod)"),
    ("death_30d_inhosp", "death (in-hosp)"),
]

rows = []
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
    d = d.merge(ecg, on=KEY, how="inner")
    d = d.merge(eh, on=KEY, how="inner")
    d = d.merge(ec, on=KEY, how="inner")
    d = d.merge(cx, on=KEY, how="left")
    if lc == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)

    y = d[lc].values.astype(int)
    g = d["subject_id"].values
    skf = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)

    re = rank01(d["p_ehr"])
    rc = rank01(d["p_ecg"])
    wm = np.zeros(len(y))
    A = np.vstack([re, rc]).T
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

    d["_cxr"] = d["p_cxr"].fillna(
        d["p_cxr"].median())

    SETS = {
        "EHR15": EHR,
        "ECG71": FEC,
        "EHR+ECG": EHR + FEC,
        "EHR+ECG+CXR": EHR + FEC + ["_cxr"],
    }

    print("\n" + "=" * 64)
    print("%s   n=%d ev=%d (%.4f)"
          % (nm, len(d), int(y.sum()),
             y.mean()))
    print("=" * 64)
    print("  LATE FUSION reference:")
    print("    p_ehr (zero-shot) AUC %.4f"
          % roc_auc_score(y, re))
    print("    p_ecg             AUC %.4f"
          % roc_auc_score(y, rc))
    print("    WMEAN2 (late)     AUC %.4f"
          % roc_auc_score(y, wm))

    print("\n  EARLY FUSION (MIMIC-trained):")
    best_name, best_p, best_a = None, None, -1
    for sn, cols in SETS.items():
        X = d[cols].values
        for kind in ["lr", "gb"]:
            p = oof(X, y, g, skf, kind)
            a = roc_auc_score(y, p)
            ap = average_precision_score(y, p)
            print("    %-14s %-3s AUC %.4f "
                  " AP %.4f"
                  % (sn, kind, a, ap))
            rows.append({
                "outcome": lc, "set": sn,
                "model": kind, "n": len(d),
                "events": int(y.sum()),
                "auc": a, "ap": ap})
            if a > best_a:
                best_a, best_p = a, p
                best_name = "%s/%s" % (sn, kind)

    print("\n  TRANSFER COST "
          "(MIMIC-trained EHR15 vs zero-shot):")
    pe = oof(d[EHR].values, y, g, skf, "lr")
    m, ci = boot(y, pe, re, g)
    print("    lr  %+0.4f [%+0.4f, %+0.4f]"
          % (m, ci[0], ci[1]))

    print("\n  EARLY vs LATE (best early = %s):"
          % best_name)
    m, ci = boot(y, best_p, wm, g)
    s = "*" if ci[0] > 0 else " "
    print("    %+0.4f [%+0.4f, %+0.4f] %s"
          % (m, ci[0], ci[1], s))

r = pd.DataFrame(rows)
out = os.path.join(
    FD, "early_fusion_results.csv")
r.to_csv(out, index=False)
print("\nSaved", out)
