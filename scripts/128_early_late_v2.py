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
OUT = DATA + "/early_late_v2_results.csv"

SEED = 42
NBOOT = 2000
STEP = 0.05
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]
EHRMAP = {"death_30d_inhosp": "death_30d"}

TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]


def rk(x):
    return rankdata(x) / (len(x) + 1.0)


def norm_path(s):
    s = str(s).rstrip("/")
    return "/".join(s.split("/")[-4:])


# ---------- EHR28 on MIMIC ----------
mm = pd.read_csv(
    DATA + "/mimic_expanded_feats.csv")
v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv")
v7 = v7.drop_duplicates("hadm_id")
vit = {"mean_temp": "temp", "mean_hr": "hr",
       "mean_sbp": "sbp", "mean_dbp": "dbp",
       "mean_rr": "rr"}
keep = ["hadm_id"] + list(vit) + FLAGS
keep = [c for c in keep if c in v7.columns]
mm = mm.merge(v7[keep].rename(columns=vit),
              on="hadm_id", how="left")
age = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v6.csv",
    usecols=["hadm_id", "age_at_admit"])
age = age.dropna().drop_duplicates("hadm_id")
mm = mm.merge(age.rename(
    columns={"age_at_admit": "age"}),
    on="hadm_id", how="left")

EC = ["mi_" + a if "mi_" + a in mm.columns
      else a for a in TA] + FLAGS + ["age"]
EC = [c for c in EC if c in mm.columns]
print("EHR features:", len(EC))

# ---------- ECG71 ----------
eg = pd.read_csv(P2 + "/bench_feats_logit.csv")
coh = pd.read_csv(P2 + "/mimic_pe_mace_cohort.csv",
                  usecols=["subject_id", "hadm_id",
                           "ecg_path"])
eg["k"] = eg["ecg_path"].apply(norm_path)
eg = eg.drop_duplicates("k")
coh["k"] = coh["ecg_path"].apply(norm_path)
ecg = coh.merge(eg.drop(columns=["ecg_path"]),
                on="k", how="inner")
LG = [c for c in ecg.columns
      if c not in ("subject_id", "hadm_id",
                   "ecg_path", "k")]
print("ECG logits:", len(LG),
      " rows:", len(ecg))
ecg = ecg.groupby("hadm_id", as_index=False)[
    LG].mean()

lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
d0 = lab.merge(mm[["hadm_id"] + EC],
               on="hadm_id", how="inner")
d0 = d0.merge(ecg, on="hadm_id", how="inner")
print("merged cohort:", len(d0))


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


def oof(X, y, grp, mk):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y, grp):
        m = mk()
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def wmean_oof(a, b, y, grp):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    o = np.zeros(len(y))
    X0 = a.reshape(-1, 1)
    for tr, te in cv.split(X0, y, grp):
        best, bw = None, None
        for w in np.arange(0, 1.0001, STEP):
            s = w * a[tr] + (1 - w) * b[tr]
            v = roc_auc_score(y[tr], s)
            if best is None or v > best:
                best, bw = v, w
        o[te] = bw * a[te] + (1 - bw) * b[te]
    return o


def boot(y, pa, pb, grp):
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(SEED)
    dd = []
    for _ in range(NBOOT):
        pk = rng.choice(us, len(us),
                        replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        dd.append(roc_auc_score(y[ii], pa[ii])
                  - roc_auc_score(y[ii], pb[ii]))
    dd = np.array(dd)
    return (float(dd.mean()),
            float(np.percentile(dd, 2.5)),
            float(np.percentile(dd, 97.5)))


rows = []
for out in OUTS:
    eh = EHRMAP.get(out, out)
    pe = pd.read_csv(DATA + "/p_ehr_harm_" + eh
                     + ".csv")[["hadm_id",
                                "p_ehr"]]
    pg = pd.read_csv(DATA + "/p_ecg_harm_" + out
                     + ".csv")[["hadm_id",
                                "p_ecg"]]
    d = d0.merge(pe, on="hadm_id")
    d = d.merge(pg, on="hadm_id")
    if out == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)
    y = d[out].values.astype(int)
    grp = d["subject_id"].values
    if y.sum() < 20:
        continue

    Xe = d[EC].values.astype(float)
    Xg = d[LG].values.astype(float)
    Xc = np.hstack([Xe, Xg])

    print("")
    print("=" * 60)
    print("%s  n=%d ev=%d (%.4f)"
          % (out, len(y), int(y.sum()),
             y.mean()))

    zs = rk(d["p_ehr"].values)
    pg_r = rk(d["p_ecg"].values)
    a_zs = roc_auc_score(y, zs)
    print("  zero-shot p_ehr      %.4f" % a_zs)

    for mn, mk in [("lr", mk_lr),
                   ("gb", mk_gb)]:
        pm = oof(Xe, y, grp, mk)
        pe71 = oof(Xg, y, grp, mk)
        pear = oof(Xc, y, grp, mk)
        rm = rk(pm)

        lt_m = wmean_oof(rm, pg_r, y, grp)
        lt_z = wmean_oof(zs, pg_r, y, grp)

        a_m = roc_auc_score(y, pm)
        a_e71 = roc_auc_score(y, pe71)
        a_ear = roc_auc_score(y, pear)
        a_ltm = roc_auc_score(y, lt_m)
        a_ltz = roc_auc_score(y, lt_z)

        tc = boot(y, pm, zs, grp)
        arch = boot(y, lt_m, rk(pear), grp)
        ecg_l = boot(y, lt_m, rm, grp)
        ecg_e = boot(y, rk(pear), rm, grp)

        print("  [%s] MIMIC-EHR28 %.4f | ECG71"
              " %.4f | EARLY %.4f"
              % (mn, a_m, a_e71, a_ear))
        print("       LATE(mimic) %.4f |"
              " LATE(zero-shot) %.4f"
              % (a_ltm, a_ltz))
        print("       transfer cost  %+.4f"
              " [%+.4f, %+.4f]" % tc)
        print("       LATE vs EARLY  %+.4f"
              " [%+.4f, %+.4f]" % arch)
        print("       ECG gain late  %+.4f"
              " [%+.4f, %+.4f]" % ecg_l)
        print("       ECG gain early %+.4f"
              " [%+.4f, %+.4f]" % ecg_e)

        rows.append({
            "outcome": out, "learner": mn,
            "n": len(y), "ev": int(y.sum()),
            "zs_ehr": a_zs, "mimic_ehr": a_m,
            "ecg71": a_e71, "early": a_ear,
            "late_mimic": a_ltm,
            "late_zs": a_ltz,
            "tcost": tc[0], "tc_lo": tc[1],
            "tc_hi": tc[2],
            "arch": arch[0],
            "ar_lo": arch[1], "ar_hi": arch[2],
            "ecg_late": ecg_l[0],
            "el_lo": ecg_l[1], "el_hi": ecg_l[2],
            "ecg_early": ecg_e[0],
            "ee_lo": ecg_e[1], "ee_hi": ecg_e[2]})

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False)
print("")
print(r.round(4).to_string())
print("")
print("REF 15-feature (scripts 92, 93): transfer cost"
      " 0.03-0.04; LATE vs EARLY +0.035 CV,"
      " +0.009 composite, -0.000 death,"
      " +0.016 in-hosp")
