import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
OUT = DATA + "/meta_learners.csv"

SEED = 42
NBOOT = 2000
STEP = 0.05
VAR = "base_cm_dv"
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


def cvsplit(y, grp):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    X0 = np.zeros((len(y), 1))
    return list(cv.split(X0, y, grp))


def oof_wmean(cols, y, grp, folds):
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


def oof_meta(X, y, folds, mk):
    o = np.zeros(len(y))
    for tr, te in folds:
        m = mk()
        m.fit(X[tr], y[tr])
        o[te] = m.predict_proba(X[te])[:, 1]
    return o


def mk_lr():
    return Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000))])


def mk_lr2():
    return Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.5, max_iter=5000))])


def mk_gb():
    return HistGradientBoostingClassifier(
        random_state=SEED, max_depth=2,
        learning_rate=0.05, max_iter=200,
        l2_regularization=1.0,
        min_samples_leaf=40)


def mk_rf():
    return RandomForestClassifier(
        n_estimators=400, max_depth=4,
        min_samples_leaf=30,
        random_state=SEED, n_jobs=-1)


def mk_mlp():
    return Pipeline([
        ("sc", StandardScaler()),
        ("nn", MLPClassifier(
            hidden_layer_sizes=(8,),
            alpha=1.0, max_iter=3000,
            random_state=SEED))])


def mk_mlp16():
    return Pipeline([
        ("sc", StandardScaler()),
        ("nn", MLPClassifier(
            hidden_layer_sizes=(16, 8),
            alpha=1.0, max_iter=3000,
            random_state=SEED))])


def boot(y, pa, pb, grp):
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(NBOOT):
        pk = rng.choice(us, len(us),
                        replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(roc_auc_score(y[ii], pa[ii])
                 - roc_auc_score(y[ii], pb[ii]))
    d = np.array(d)
    return (float(d.mean()),
            float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")

rows = []
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
    d = d.reset_index(drop=True)

    y = d[ycol].values.astype(int)
    grp = d["subject_id"].values
    folds = cvsplit(y, grp)

    a = rk(d["p_ehr"].values)
    b = rk(d["p_ecg"].values)
    c = rk(d["p_ctpa"].values)
    X = np.column_stack([a, b, c])

    print("")
    print("#" * 60)
    print("%s  n=%d ev=%d"
          % (ycol, len(y), int(y.sum())))
    print("  modalities: EHR %.4f  ECG %.4f"
          "  CTPA %.4f"
          % (roc_auc_score(y, a),
             roc_auc_score(y, b),
             roc_auc_score(y, c)))

    ref = oof_wmean([a, b, c], y, grp, folds)
    a_ref = roc_auc_score(y, ref)
    ap_ref = average_precision_score(y, ref)
    mean3 = (a + b + c) / 3.0
    print("")
    print("  %-16s %8s %8s %10s"
          % ("meta-learner", "AUC", "AP",
             "vs WMEAN3"))
    print("  %-16s %8.4f %8.4f %10s"
          % ("MEAN3 (equal)",
             roc_auc_score(y, mean3),
             average_precision_score(
                 y, mean3), ""))
    print("  %-16s %8.4f %8.4f %10s"
          % ("WMEAN3 (ref)", a_ref, ap_ref,
             "--"))

    METAS = [("LR (linear)", mk_lr),
             ("LR C=0.5", mk_lr2),
             ("GB (nonlinear)", mk_gb),
             ("RF (nonlinear)", mk_rf),
             ("MLP 8", mk_mlp),
             ("MLP 16-8", mk_mlp16)]

    for nm, mk in METAS:
        p = oof_meta(X, y, folds, mk)
        au = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        t = boot(y, p, ref, grp)
        star = " *" if (t[1] > 0
                        or t[2] < 0) else ""
        print("  %-16s %8.4f %8.4f  %+.4f"
              " [%+.4f, %+.4f]%s"
              % (nm, au, ap, t[0], t[1],
                 t[2], star))
        rows.append({
            "outcome": ycol, "meta": nm,
            "auc": au, "ap": ap,
            "wmean3": a_ref,
            "diff": t[0], "lo": t[1],
            "hi": t[2]})

    rows.append({
        "outcome": ycol, "meta": "WMEAN3",
        "auc": a_ref, "ap": ap_ref,
        "wmean3": a_ref, "diff": 0.0,
        "lo": 0.0, "hi": 0.0})
    rows.append({
        "outcome": ycol, "meta": "MEAN3",
        "auc": roc_auc_score(y, mean3),
        "ap": average_precision_score(
            y, mean3),
        "wmean3": a_ref, "diff": np.nan,
        "lo": np.nan, "hi": np.nan})

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string())
print("")
print("Cahan et al. 2023: late-average 0.85,"
      " late-TabNet 0.92, late-XGBoost 0.88"
      " (10 test events)")
print("saved", OUT)
