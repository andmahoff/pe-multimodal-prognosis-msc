import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
OUT = DATA + "/supervised_fusion.csv"

SEED = 42
NBOOT = 2000
STEP = 0.05
VAR = "base_cm_dv"
JOBS = ["death_30d", "composite_30d"]

TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]
DUP = ["lb_anion_gap", "lb_bicarbonate",
       "lb_calcium_total", "lb_chloride",
       "lb_creatinine", "lb_glucose",
       "lb_hematocrit", "lb_hemoglobin",
       "lb_mch", "lb_mchc", "lb_mcv",
       "lb_platelet_count", "lb_potassium",
       "lb_rdw", "lb_red_blood_cells",
       "lb_sodium", "lb_urea_nitrogen",
       "lb_wbc", "lb_white_blood_cells"]


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


def oof_w(cols, y, grp):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    oof = np.zeros(len(y))
    W = []
    X0 = cols[0].reshape(-1, 1)
    for tr, te in cv.split(X0, y, grp):
        best, bw = None, None
        for w in G3:
            v = roc_auc_score(
                y[tr],
                wf([c[tr] for c in cols], w))
            if best is None or v > best:
                best, bw = v, w
        oof[te] = wf([c[te] for c in cols], bw)
        W.append(tuple(round(x, 2)
                       for x in bw))
    return oof, W


def oof_pred(X, y, grp, mk):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y, grp):
        m = mk()
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def mk_lr():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1, max_iter=5000))])


def mk_gb():
    return HistGradientBoostingClassifier(
        random_state=SEED, max_depth=3,
        learning_rate=0.05, max_iter=300,
        l2_regularization=1.0)


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
mm = pd.read_csv(
    DATA + "/mimic_expanded_feats.csv")
v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv"
).drop_duplicates("hadm_id")
vit = {"mean_temp": "temp", "mean_hr": "hr",
       "mean_sbp": "sbp", "mean_dbp": "dbp",
       "mean_rr": "rr", "mean_spo2": "spo2"}
keep = ["hadm_id"] + [c for c in vit
                      if c in v7.columns] \
    + [c for c in FLAGS + ["nlr"]
       if c in v7.columns]
mm = mm.merge(v7[keep].rename(columns=vit),
              on="hadm_id", how="left")
age = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v6.csv",
    usecols=["hadm_id", "age_at_admit"])
mm = mm.merge(
    age.dropna().drop_duplicates("hadm_id")
    .rename(columns={"age_at_admit": "age"}),
    on="hadm_id", how="left")
allb = pd.read_csv(DATA + "/mimic_all_labs.csv")
allb = allb[[c for c in allb.columns
             if c == "hadm_id"
             or c not in DUP]]
mm = mm.merge(allb, on="hadm_id", how="left")

C28 = [c for c in
       (["mi_" + a if "mi_" + a in mm.columns
         else a for a in TA] + FLAGS + ["age"])
       if c in mm.columns]
CALL = sorted(set(
    C28 + [c for c in mm.columns
           if c.startswith("lb_")]
    + [c for c in ["spo2", "nlr"]
       if c in mm.columns]))
BAN = ["30d", "first", "label", "composite",
       "death", "mace", "_y"]
CALL = [c for c in CALL
        if not any(b in c.lower()
                   for b in BAN)]
print("EHR28:", len(C28),
      " EHRall:", len(CALL))

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
    d = lab.merge(mm, on=["subject_id",
                          "hadm_id"])
    d = d.merge(pe, on="hadm_id")
    d = d.merge(pg, on="hadm_id")
    d = d.merge(pc, on="hadm_id")
    d = d.reset_index(drop=True)

    y = d[ycol].values.astype(int)
    grp = d["subject_id"].values
    print("")
    print("#" * 60)
    print("%s  n=%d ev=%d (presentation"
          " cohort)" % (ycol, len(y),
                        int(y.sum())))

    zs = rk(d["p_ehr"].values)
    ecg = rk(d["p_ecg"].values)
    ctp = rk(d["p_ctpa"].values)

    m28 = rk(oof_pred(
        d[C28].values.astype(float), y,
        grp, mk_gb))
    mall = rk(oof_pred(
        d[CALL].values.astype(float), y,
        grp, mk_gb))

    print("  MODALITIES")
    print("    EHR zero-shot   %.4f"
          % roc_auc_score(y, zs))
    print("    EHR MIMIC-28    %.4f"
          % roc_auc_score(y, m28))
    print("    EHR MIMIC-all   %.4f"
          % roc_auc_score(y, mall))
    print("    ECG             %.4f"
          % roc_auc_score(y, ecg))
    print("    CTPA            %.4f"
          % roc_auc_score(y, ctp))

    for nm, eh in [("zero-shot", zs),
                   ("MIMIC-28", m28),
                   ("MIMIC-all", mall)]:
        a_e = roc_auc_score(y, eh)
        w3, W = oof_w([eh, ecg, ctp], y, grp)
        a_3 = roc_auc_score(y, w3)
        ap3 = average_precision_score(y, w3)
        t = boot(y, w3, eh, grp)
        we = float(np.mean([w[0] for w in W]))
        wg = float(np.mean([w[1] for w in W]))
        wc = float(np.mean([w[2] for w in W]))
        print("")
        print("  EHR = %s" % nm)
        print("    WMEAN3 %.4f  AP %.4f"
              % (a_3, ap3))
        print("    vs EHR alone %+.4f"
              " [%+.4f, %+.4f]" % t)
        print("    mean weights: EHR %.2f"
              "  ECG %.2f  CTPA %.2f"
              % (we, wg, wc))
        print("    fold weights:", W)
        rows.append({
            "outcome": ycol, "ehr": nm,
            "ehr_auc": a_e, "wmean3": a_3,
            "ap": ap3, "gain": t[0],
            "lo": t[1], "hi": t[2],
            "w_ehr": we, "w_ecg": wg,
            "w_ctpa": wc})

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string())
print("saved", OUT)
