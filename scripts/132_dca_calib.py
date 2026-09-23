import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
OUT = DATA + "/dca_calib"
os.makedirs(OUT, exist_ok=True)

SEED = 42
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]
THR = np.arange(0.01, 0.51, 0.01)

lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")

age = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v6.csv",
    usecols=["hadm_id", "age_at_admit"])
age = age.dropna().drop_duplicates("hadm_id")

v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv"
).drop_duplicates("hadm_id")
sp = v7[["hadm_id", "mean_hr", "mean_sbp",
         "mean_spo2", "cancer", "copd",
         "heart_failure"]].merge(
    age, on="hadm_id", how="left")
t = np.zeros(len(sp))
t += (sp["mean_hr"] >= 110).fillna(
    False).astype(int).values
t += (sp["mean_sbp"] < 100).fillna(
    False).astype(int).values
t += (sp["mean_spo2"] < 90).fillna(
    False).astype(int).values
t += sp["cancer"].fillna(0).astype(int).values
cp = (sp["heart_failure"].fillna(0).values
      + sp["copd"].fillna(0).values)
t += (cp > 0).astype(int)
t += (sp["age_at_admit"] > 80).fillna(
    False).astype(int).values
sp["spesi"] = t
sp = sp[["hadm_id", "spesi"]]


def platt(y, p, grp):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    o = np.zeros(len(y))
    X = p.reshape(-1, 1)
    for tr, te in cv.split(X, y, grp):
        m = LogisticRegression(max_iter=1000)
        m.fit(X[tr], y[tr])
        o[te] = m.predict_proba(X[te])[:, 1]
    return o


def net_benefit(y, p, th):
    pred = (p >= th)
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    n = len(y)
    return tp / n - (fp / n) * (th / (1 - th))


rows = []
crows = []
for out in OUTS:
    f = DATA + "/p_wmean2_" + out + ".csv"
    if not os.path.exists(f):
        continue
    pw = pd.read_csv(f)
    pc = [c for c in pw.columns
          if "wmean2" in c.lower()
          and "cal" not in c.lower()]
    if not pc:
        continue
    d = lab.merge(pw[["hadm_id", pc[0]]],
                  on="hadm_id")
    d = d.merge(sp, on="hadm_id", how="left")
    d = d.merge(age, on="hadm_id", how="left")
    if out == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)
    y = d[out].values.astype(int)
    grp = d["subject_id"].values
    cal = platt(y, d[pc[0]].values, grp)
    d["cal"] = cal

    print("")
    print("=" * 58)
    print("%s n=%d ev=%d prev %.4f"
          % (out, len(y), int(y.sum()),
             y.mean()))
    print("  Brier %.4f"
          % brier_score_loss(y, cal))

    sv = d["spesi"].fillna(
        d["spesi"].median()).values
    smax = float(np.nanmax(sv)) or 1.0
    sp_p = sv / smax

    print("")
    print("  DECISION CURVE (net benefit)")
    print("  %6s %9s %9s %9s %9s"
          % ("thr", "model", "sPESI",
             "treat-all", "gain"))
    for th in [0.02, 0.05, 0.10, 0.15,
               0.20, 0.30, 0.40]:
        nb_m = net_benefit(y, cal, th)
        nb_s = net_benefit(y, sp_p, th)
        nb_a = net_benefit(
            y, np.ones(len(y)), th)
        print("  %6.2f %9.4f %9.4f %9.4f"
              " %+9.4f"
              % (th, nb_m, nb_s, nb_a,
                 nb_m - max(nb_s, nb_a, 0)))

    full = []
    for th in THR:
        full.append({
            "outcome": out, "thr": th,
            "nb_model": net_benefit(y, cal, th),
            "nb_spesi": net_benefit(y, sp_p, th),
            "nb_all": net_benefit(
                y, np.ones(len(y)), th),
            "nb_none": 0.0})
    pd.DataFrame(full).to_csv(
        OUT + "/dca_" + out + ".csv",
        index=False)

    d["band"] = pd.cut(
        d["age_at_admit"],
        [0, 50, 65, 80, 200],
        labels=["<50", "50-64", "65-79",
                "80+"])
    print("")
    print("  CALIBRATION BY AGE BAND")
    print("  %-8s %5s %5s %8s %8s %8s"
          % ("band", "n", "ev", "obs",
             "pred", "Brier"))
    for lv in d["band"].cat.categories:
        m = (d["band"] == lv).values
        if m.sum() < 60 or y[m].sum() < 10:
            continue
        obs = float(y[m].mean())
        pr = float(cal[m].mean())
        br = brier_score_loss(y[m], cal[m])
        print("  %-8s %5d %5d %8.4f %8.4f"
              " %8.4f"
              % (lv, int(m.sum()),
                 int(y[m].sum()), obs, pr, br))
        crows.append({
            "outcome": out, "band": str(lv),
            "n": int(m.sum()),
            "ev": int(y[m].sum()),
            "obs": obs, "pred": pr,
            "ratio": obs / pr if pr else np.nan,
            "brier": br})

pd.DataFrame(crows).to_csv(
    OUT + "/calib_by_age.csv", index=False)
print("")
print(pd.DataFrame(crows).round(4)
      .to_string(index=False))
print("saved to", OUT)
