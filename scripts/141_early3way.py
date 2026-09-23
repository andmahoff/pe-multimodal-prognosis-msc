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
OUT = DATA + "/early3way.csv"

SEED = 42
NBOOT = 2000
STEP = 0.05
VAR = "base_cm_dv"
JOBS = ["death_30d", "composite_30d"]
POST_H = 24

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
CTBASE = ["pe_pos", "saddle", "central",
          "lobar", "segmental", "subseg",
          "bilateral", "rv_strain",
          "septal_bow", "reflux",
          "mpa_enlarge", "infarct",
          "effusion", "malignancy",
          "consolid", "atelect", "edema",
          "cardiomeg", "adenopathy",
          "pe_neg", "txt_len", "n_sent"]


def rk(x):
    return rankdata(x) / (len(x) + 1.0)


def norm_path(s):
    s = str(s).rstrip("/")
    return "/".join(s.split("/")[-4:])


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
    o = np.zeros(len(y))
    X0 = cols[0].reshape(-1, 1)
    for tr, te in cv.split(X0, y, grp):
        best, bw = None, None
        for w in G3:
            v = roc_auc_score(
                y[tr],
                wf([c[tr] for c in cols], w))
            if best is None or v > best:
                best, bw = v, w
        o[te] = wf([c[te] for c in cols], bw)
    return o


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
             if c == "hadm_id" or c not in DUP]]
mm = mm.merge(allb, on="hadm_id", how="left")

eg = pd.read_csv(P2 + "/bench_feats_logit.csv")
coh = pd.read_csv(
    P2 + "/mimic_pe_mace_cohort.csv",
    usecols=["subject_id", "hadm_id",
             "ecg_path"])
eg["k"] = eg["ecg_path"].apply(norm_path)
eg = eg.drop_duplicates("k")
coh["k"] = coh["ecg_path"].apply(norm_path)
ecgm = coh.merge(
    eg.drop(columns=["ecg_path"]),
    on="k", how="inner")
LG = [c for c in ecgm.columns
      if c not in ("subject_id", "hadm_id",
                   "ecg_path", "k")]
ecgm = ecgm.groupby("hadm_id",
                    as_index=False)[LG].mean()

fx = pd.read_csv(
    DATA + "/ctpa_comorb_features.csv")
nt = pd.read_csv(
    DATA + "/ctpa_notes_index.csv")
for c in ["charttime", "admittime"]:
    nt[c] = pd.to_datetime(nt[c],
                           errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm",
                as_index=False).first()
nt["h"] = ((nt["charttime"] - nt["admittime"])
           .dt.total_seconds() / 3600.0)
nt = nt[nt["h"] <= POST_H]
fx = fx[fx["hadm_id"].isin(nt["idx_hadm"])]

CT = [c for c in fx.columns
      if c in CTBASE
      or c.startswith("cm_")
      or c.startswith("dv_")]
excl = [c for c in fx.columns
        if c not in CT
        and c not in ("hadm_id", "subject_id")]
print("CTPA whitelist:", len(CT))
print("CTPA excluded:", excl)
print("ECG logits:", len(LG))

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
       "death", "mace", "_y", "_days",
       "inhosp"]
CALL = [c for c in CALL
        if not any(b in c.lower()
                   for b in BAN)]
C28 = [c for c in C28
       if not any(b in c.lower()
                  for b in BAN)]
print("EHR28:", len(C28),
      " EHRall:", len(CALL))

rows = []
for ycol in JOBS:
    pg = pd.read_csv(
        DATA + "/p_ecg_harm_" + ycol
        + ".csv")[["hadm_id", "p_ecg"]]
    pc = pd.read_csv(
        DATA + "/p_ctpa_pres_" + VAR + "_"
        + ycol + ".csv")[["hadm_id",
                          "p_ctpa"]]
    d = lab.merge(mm, on=["subject_id",
                          "hadm_id"])
    d = d.merge(ecgm, on="hadm_id")
    d = d.merge(fx[["hadm_id"] + CT],
                on="hadm_id")
    d = d.merge(pg, on="hadm_id")
    d = d.merge(pc, on="hadm_id")
    d = d.reset_index(drop=True)

    miss = [c for c in C28 + CALL + LG + CT
            if c not in d.columns]
    if miss:
        print("MISSING after merge:",
              sorted(set(miss))[:10])
        raise SystemExit(1)

    y = d[ycol].values.astype(int)
    grp = d["subject_id"].values
    print("")
    print("#" * 60)
    print("%s  n=%d ev=%d"
          % (ycol, len(y), int(y.sum())))

    ecg_r = rk(d["p_ecg"].values)
    ctp_r = rk(d["p_ctpa"].values)

    for enm, EC in [("EHR28", C28),
                    ("EHRall", CALL)]:
        for ln, mk in [("lr", mk_lr),
                       ("gb", mk_gb)]:
            eh = rk(oof_pred(
                d[EC].values.astype(float),
                y, grp, mk))
            a_e = roc_auc_score(y, eh)

            late = oof_w([eh, ecg_r, ctp_r],
                         y, grp)
            a_l = roc_auc_score(y, late)
            ap_l = average_precision_score(
                y, late)

            Xe = d[EC + LG + CT].values.astype(
                float)
            early = oof_pred(Xe, y, grp, mk)
            a_ear = roc_auc_score(y, early)
            ap_e = average_precision_score(
                y, early)

            X2 = d[EC + CT].values.astype(float)
            a_ec = roc_auc_score(
                y, oof_pred(X2, y, grp, mk))

            X3 = d[EC + LG].values.astype(float)
            a_eg = roc_auc_score(
                y, oof_pred(X3, y, grp, mk))

            t = boot(y, late, rk(early), grp)
            print("")
            print("  %s / %s  (EHR alone"
                  " %.4f)" % (enm, ln, a_e))
            print("    EARLY EHR+ECG   %.4f"
                  % a_eg)
            print("    EARLY EHR+CTPA  %.4f"
                  % a_ec)
            print("    EARLY 3-way     %.4f"
                  "  AP %.4f  (%d feats)"
                  % (a_ear, ap_e,
                     Xe.shape[1]))
            print("    LATE  WMEAN3    %.4f"
                  "  AP %.4f" % (a_l, ap_l))
            print("    LATE vs EARLY %+.4f"
                  " [%+.4f, %+.4f]" % t)
            rows.append({
                "outcome": ycol, "ehr": enm,
                "learner": ln, "ehr_auc": a_e,
                "early_ehr_ecg": a_eg,
                "early_ehr_ctpa": a_ec,
                "early3": a_ear,
                "late": a_l, "ap_early": ap_e,
                "ap_late": ap_l,
                "nfeat_early": Xe.shape[1],
                "diff": t[0], "lo": t[1],
                "hi": t[2]})

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string())
print("saved", OUT)
