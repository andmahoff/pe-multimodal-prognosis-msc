import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
OUT = DATA + "/ecg_named_xai"
os.makedirs(OUT, exist_ok=True)

SEED = 42
STEP = 0.05
VAR = "base_cm_dv"
JOBS = ["death_30d", "composite_30d"]

MLB = os.path.expanduser(
    "~/ecg_ptbxl_benchmarking/output"
    "/exp0/data/mlb.pkl")
print("mlb.pkl:", MLB, os.path.exists(MLB))

names = None
try:
    with open(MLB, "rb") as f:
        obj = pickle.load(f)
    names = [str(x) for x in obj.classes_]
    print("SCP statements:", len(names))
    print("first 10:", names[:10])
except Exception as e:
    print("mlb load failed:", e)


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


eg = pd.read_csv(P2 + "/bench_feats_logit.csv")
coh = pd.read_csv(
    P2 + "/mimic_pe_mace_cohort.csv",
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
print("logit cols:", len(LG),
      " matched rows:", len(ecg))

if names is not None and len(names) == len(LG):
    LAB = dict(zip(LG, names))
    print("statement names attached")
else:
    LAB = {c: c for c in LG}
    print("NOTE: using raw column names"
          " (%s vs %d)"
          % (len(names) if names else "none",
             len(LG)))

ecg = ecg.groupby("hadm_id",
                  as_index=False)[LG].mean()
lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")

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
    d = d.merge(ecg, on="hadm_id")
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

    X = d[LG].values.astype(float)
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.003, max_iter=5000))])
    pipe.fit(X, y)
    Z = pipe.named_steps["sc"].transform(X)
    beta = pipe.named_steps["lr"].coef_[0]
    S = Z * beta

    t = pd.DataFrame({
        "col": LG,
        "statement": [LAB[c0] for c0 in LG],
        "beta": beta,
        "shap": np.abs(S).mean(axis=0),
        "signed": S.mean(axis=0)})
    t = t.sort_values("shap", ascending=False)
    t.to_csv(OUT + "/ecg_shap_" + ycol
             + ".csv", index=False)

    print("")
    print("TOP SCP-ECG STATEMENTS BY |SHAP|")
    print("  %-14s %8s %8s %8s"
          % ("statement", "beta", "|SHAP|",
             "signed"))
    for _, r in t.head(15).iterrows():
        print("  %-14s %+8.3f %8.3f %+8.3f"
              % (str(r["statement"])[:14],
                 r["beta"], r["shap"],
                 r["signed"]))

    ehr_ct = W[:, 0] * a + W[:, 2] * c
    den = W[:, 0] + W[:, 2]
    base = np.divide(ehr_ct, den,
                     out=np.zeros(len(a)),
                     where=den > 0)
    shift = rankdata(w3) - rankdata(base)
    d["shift"] = shift

    up = d.nlargest(200, "shift")
    dn = d.nsmallest(200, "shift")
    rows = []
    for c0 in LG:
        mu = float(up[c0].mean())
        md = float(dn[c0].mean())
        rows.append((LAB[c0], mu, md, mu - md))
    rr = pd.DataFrame(
        rows, columns=["statement", "up",
                       "down", "diff"])
    rr = rr.sort_values("diff",
                        ascending=False)
    rr.to_csv(OUT + "/ecg_shift_" + ycol
              + ".csv", index=False)

    print("")
    print("STATEMENTS WHERE ECG RAISED RISK")
    print("  %-14s %9s %9s %8s"
          % ("statement", "mean_up",
             "mean_down", "diff"))
    for _, r in rr.head(8).iterrows():
        print("  %-14s %9.3f %9.3f %+8.3f"
              % (str(r["statement"])[:14],
                 r["up"], r["down"],
                 r["diff"]))
    print("  ... where ECG LOWERED risk:")
    for _, r in rr.tail(5).iterrows():
        print("  %-14s %9.3f %9.3f %+8.3f"
              % (str(r["statement"])[:14],
                 r["up"], r["down"],
                 r["diff"]))

    ev = d.index[y == 1]
    print("")
    print("  mean ECG weight %.3f"
          % W[:, 1].mean())
    print("  ECG raised risk for %d of %d"
          " events"
          % (int((shift[ev] > 0).sum()),
             len(ev)))

print("")
print("saved to", OUT)
