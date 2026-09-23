import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
OUT = DATA + "/fusion_xai"
os.makedirs(OUT, exist_ok=True)

SEED = 42
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


def oof_w(cols, y, grp):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    oof = np.zeros(len(y))
    W = np.zeros((len(y), 3))
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
        W[te, :] = np.array(bw)
    return oof, W


lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
fx = pd.read_csv(
    DATA + "/ctpa_comorb_features.csv")
CTF = [c for c in fx.columns
       if c.startswith("cm_")
       or c.startswith("dv_")
       or c in ["pe_pos", "saddle", "central",
                "lobar", "segmental", "subseg",
                "bilateral", "rv_strain",
                "infarct", "effusion",
                "malignancy", "consolid",
                "atelect", "edema",
                "cardiomeg", "adenopathy"]]
print("CTPA findings available:", len(CTF))

dem = None
for f in ["mimic_pe_mace_cohort.csv",
          "mimic_pe_ehr_baseline_v6.csv"]:
    p = os.path.join(P2, f)
    if not os.path.exists(p):
        continue
    h = pd.read_csv(p, nrows=0)
    cc = ["hadm_id"] + [
        c for c in ["age_at_admit", "gender",
                    "sex"] if c in h.columns]
    if len(cc) > 1:
        dem = pd.read_csv(
            p, usecols=cc
        ).drop_duplicates("hadm_id")
        print("demographics from", f)
        break

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
    d = d.merge(fx[["hadm_id"] + CTF],
                on="hadm_id", how="left")
    if dem is not None:
        d = d.merge(dem, on="hadm_id",
                    how="left")
    d = d.reset_index(drop=True)

    y = d[ycol].values.astype(int)
    grp = d["subject_id"].values
    a = rk(d["p_ehr"].values)
    b = rk(d["p_ecg"].values)
    c = rk(d["p_ctpa"].values)
    w3, W = oof_w([a, b, c], y, grp)

    print("")
    print("#" * 58)
    print("%s  n=%d ev=%d  WMEAN3 %.4f"
          % (ycol, len(y), int(y.sum()),
             roc_auc_score(y, w3)))

    ct = pd.DataFrame({
        "subject_id": d["subject_id"],
        "hadm_id": d["hadm_id"], "y": y,
        "r_ehr": a, "r_ecg": b, "r_ctpa": c,
        "w_ehr": W[:, 0], "w_ecg": W[:, 1],
        "w_ctpa": W[:, 2],
        "c_ehr": W[:, 0] * a,
        "c_ecg": W[:, 1] * b,
        "c_ctpa": W[:, 2] * c,
        "score": w3})
    ct["pct"] = rankdata(w3) / len(w3) * 100.0
    ct.to_csv(OUT + "/attrib_" + ycol + ".csv",
              index=False)

    tot = ct[["c_ehr", "c_ecg",
              "c_ctpa"]].sum(axis=1)
    print("")
    print("MEAN MODALITY CONTRIBUTION")
    for nm in ["ehr", "ecg", "ctpa"]:
        print("  %-5s weight %.3f   share"
              " of score %.3f"
              % (nm, ct["w_" + nm].mean(),
                 float((ct["c_" + nm]
                        / tot).mean())))

    sd = ct[["r_ehr", "r_ecg",
             "r_ctpa"]].std(axis=1)
    q = pd.qcut(sd, 4, labels=False,
                duplicates="drop")
    print("")
    print("PERFORMANCE BY MODALITY DISAGREEMENT")
    print("  %-5s %5s %4s %7s %7s %7s %7s"
          % ("q", "n", "ev", "EHR", "ECG",
             "CTPA", "WMEAN3"))
    for k in sorted(pd.unique(q)):
        m = (q == k).values
        if y[m].sum() < 10:
            continue
        print("  Q%-4d %5d %4d %7.4f %7.4f"
              " %7.4f %7.4f"
              % (k + 1, int(m.sum()),
                 int(y[m].sum()),
                 roc_auc_score(y[m], a[m]),
                 roc_auc_score(y[m], b[m]),
                 roc_auc_score(y[m], c[m]),
                 roc_auc_score(y[m], w3[m])))

    ehr_pct = rankdata(a) / len(a)
    hi = (ct["pct"] >= 80).values
    resc = ct[hi & (ehr_pct < 0.5)
              & (y == 1)]
    harm = ct[(~hi) & (ehr_pct >= 0.8)
              & (y == 1)]
    print("")
    print("FUSION RESCUE %d | FUSION HARM %d"
          % (len(resc), len(harm)))

    print("")
    print("CASE STUDIES")
    cases = []
    for _, r in resc.sort_values(
            "pct", ascending=False).head(2
                                          ).iterrows():
        cases.append(("RESCUED BY FUSION", r))
    for _, r in ct[y == 1].sort_values(
            "pct", ascending=False).head(1
                                          ).iterrows():
        cases.append(("HIGHEST-RISK EVENT", r))
    for _, r in ct[y == 0].sort_values(
            "pct", ascending=False).head(1
                                          ).iterrows():
        cases.append(("HIGHEST-RISK NON-EVENT",
                      r))

    for title, r in cases:
        h = int(r["hadm_id"])
        row = d[d["hadm_id"] == h]
        print("")
        print("  " + "-" * 50)
        print("  %s   hadm_id %d" % (title, h))
        if dem is not None and \
                "age_at_admit" in row.columns:
            print("    age %s"
                  % row["age_at_admit"].values[0])
        print("    outcome %d   score pct %.1f"
              % (int(r["y"]), r["pct"]))
        print("    modality pct: EHR %.1f  ECG %.1f"
              "  CTPA %.1f"
              % (r["r_ehr"] * 100,
                 r["r_ecg"] * 100,
                 r["r_ctpa"] * 100))
        print("    weights: EHR %.2f  ECG %.2f"
              "  CTPA %.2f"
              % (r["w_ehr"], r["w_ecg"],
                 r["w_ctpa"]))
        pos = [f for f in CTF
               if f in row.columns
               and row[f].notna().all()
               and row[f].values[0] == 1]
        print("    CTPA: %s"
              % (", ".join(pos[:12])
                 if pos else "none extracted"))

print("")
print("saved to", OUT)
