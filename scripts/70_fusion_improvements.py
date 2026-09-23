import os
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss

FD = (
    "./"
    "fusion_workspace/data/"
)
SEED = 42
NBOOT = 1000
EPS = 1e-6


def rank01(x):
    return pd.Series(x).rank(pct=True).values


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def oof_platt(x, y, g, skf):
    x = np.asarray(x).reshape(-1, 1)
    out = np.zeros(len(y))
    for tr, te in skf.split(x, y, groups=g):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(x[tr], y[tr])
        out[te] = lr.predict_proba(x[te])[:, 1]
    return out


def oof_stack(X, y, g, skf):
    X = np.asarray(X)
    out = np.zeros(len(y))
    for tr, te in skf.split(X, y, groups=g):
        lr = LogisticRegression(
            max_iter=3000, C=1.0
        )
        lr.fit(X[tr], y[tr])
        out[te] = lr.predict_proba(X[te])[:, 1]
    return out


def boot(y, pa, pb, g, n=NBOOT):
    rng = np.random.RandomState(SEED)
    g = np.asarray(g)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    o = []
    for _ in range(n):
        s = rng.choice(uq, len(uq),
                       replace=True)
        t = np.concatenate([idx[u] for u in s])
        if len(np.unique(y[t])) < 2:
            continue
        o.append(
            roc_auc_score(y[t], pa[t])
            - roc_auc_score(y[t], pb[t])
        )
    o = np.array(o)
    return o.mean(), np.percentile(
        o, [2.5, 97.5]
    )


def nri(y, p_old, p_new, cuts):
    y = np.asarray(y)
    co = np.digitize(p_old, cuts)
    cn = np.digitize(p_new, cuts)
    ev = y == 1
    ne = y == 0
    up_e = (cn[ev] > co[ev]).sum()
    dn_e = (cn[ev] < co[ev]).sum()
    up_n = (cn[ne] > co[ne]).sum()
    dn_n = (cn[ne] < co[ne]).sum()
    nri_e = (up_e - dn_e) / max(ev.sum(), 1)
    nri_n = (dn_n - up_n) / max(ne.sum(), 1)
    return nri_e, nri_n, nri_e + nri_n, {
        "up_event": int(up_e),
        "down_event": int(dn_e),
        "up_nonevent": int(up_n),
        "down_nonevent": int(dn_n),
    }


def cont_nri(y, p_old, p_new):
    y = np.asarray(y)
    d = p_new - p_old
    ev = y == 1
    ne = y == 0
    a = ((d[ev] > 0).mean()
         - (d[ev] < 0).mean())
    b = ((d[ne] < 0).mean()
         - (d[ne] > 0).mean())
    return a, b, a + b


print("Loading...")
d = pd.read_csv(
    os.path.join(FD, "ro4_fusion_table.csv")
)
y = d["label"].values
g = d["subject_id"].values
print("n=%d events=%d" % (len(d), int(y.sum())))

d["r_spesi"] = rank01(d["spesi"].fillna(0))
ARMS = {
    "ehr": d["r_ehr"].values,
    "ecg": d["r_ecg"].values,
    "cxr": d["r_cxr"].values,
    "spesi": d["r_spesi"].values,
}
META = ((d["r_port"] + d["r_ndur"]) / 2).values

skf = StratifiedGroupKFold(
    n_splits=5, shuffle=True,
    random_state=SEED
)

print("\nCalibrating each modality (OOF Platt)...")
CAL = {}
for k, v in ARMS.items():
    CAL[k] = oof_platt(v, y, g, skf)
    print("  %-6s AUC %.4f  Brier %.4f"
          % (k, roc_auc_score(y, CAL[k]),
             brier_score_loss(y, CAL[k])))
CAL["meta"] = oof_platt(META, y, g, skf)

M = {}
M["MEAN3(rank)"] = (
    ARMS["ehr"] + ARMS["ecg"] + ARMS["cxr"]
) / 3
M["MEAN2(rank)"] = (
    ARMS["ehr"] + ARMS["ecg"]
) / 2
M["MEAN4(rank,+sPESI)"] = (
    ARMS["ehr"] + ARMS["ecg"]
    + ARMS["cxr"] + ARMS["spesi"]
) / 4

M["LOGIT3"] = (
    logit(CAL["ehr"]) + logit(CAL["ecg"])
    + logit(CAL["cxr"])
) / 3
M["LOGIT2"] = (
    logit(CAL["ehr"]) + logit(CAL["ecg"])
) / 2
M["LOGIT4(+sPESI)"] = (
    logit(CAL["ehr"]) + logit(CAL["ecg"])
    + logit(CAL["cxr"]) + logit(CAL["spesi"])
) / 4

M["MEDIAN3"] = np.median(
    np.vstack([ARMS["ehr"], ARMS["ecg"],
               ARMS["cxr"]]), axis=0
)

print("\nWeighted rank average "
      "(weights tuned inside folds)...")
grid = [
    w for w in itertools.product(
        [0, 1, 2, 3], repeat=3
    ) if sum(w) > 0
]
wo = np.zeros(len(y))
chosen = []
A = np.vstack([ARMS["ehr"], ARMS["ecg"],
               ARMS["cxr"]]).T
for tr, te in skf.split(A, y, groups=g):
    best, bw = -1, None
    for w in grid:
        wv = np.array(w, dtype=float)
        s = A[tr].dot(wv) / wv.sum()
        a = roc_auc_score(y[tr], s)
        if a > best:
            best, bw = a, wv
    chosen.append(tuple(bw))
    wo[te] = A[te].dot(bw) / bw.sum()
M["WMEAN3"] = wo
print("  chosen weights per fold:", chosen)

print("\nStacks (OOF)...")
M["STACK3"] = oof_stack(
    np.vstack([ARMS["ehr"], ARMS["ecg"],
               ARMS["cxr"]]).T, y, g, skf)
M["STACK4(+sPESI)"] = oof_stack(
    np.vstack([ARMS["ehr"], ARMS["ecg"],
               ARMS["cxr"],
               ARMS["spesi"]]).T, y, g, skf)
M["STACK4+META"] = oof_stack(
    np.vstack([ARMS["ehr"], ARMS["ecg"],
               ARMS["cxr"], ARMS["spesi"],
               META]).T, y, g, skf)

print("\nGated / residual fusion...")
base = oof_stack(
    np.vstack([ARMS["ehr"],
               ARMS["ecg"]]).T, y, g, skf)
bl = logit(base)
blz = (bl - bl.mean()) / bl.std()
cz = ((ARMS["cxr"] - ARMS["cxr"].mean())
      / ARMS["cxr"].std())
Xg = np.vstack([blz, cz, blz * cz]).T
M["GATED(base+cxr+int)"] = oof_stack(
    Xg, y, g, skf)
M["RESID(base+cxr)"] = oof_stack(
    np.vstack([blz, cz]).T, y, g, skf)

print("\n" + "=" * 62)
print("RESULTS (n=%d, events=%d)"
      % (len(d), int(y.sum())))
print("=" * 62)
print("%-22s %-8s %-8s" % ("model", "AUC", "AUPRC"))
for k, v in ARMS.items():
    print("%-22s %.4f   %.4f"
          % ("BASE:" + k, roc_auc_score(y, v),
             average_precision_score(y, v)))
rows = []
for k, v in M.items():
    a = roc_auc_score(y, v)
    p = average_precision_score(y, v)
    rows.append({"model": k, "auc": a, "ap": p})
    print("%-22s %.4f   %.4f" % (k, a, p))

ref = M["MEAN3(rank)"]
print("\n" + "=" * 62)
print("vs MEAN3(rank) baseline")
print("=" * 62)
for k in ["MEAN4(rank,+sPESI)", "LOGIT3",
          "LOGIT4(+sPESI)", "WMEAN3",
          "MEDIAN3", "STACK4(+sPESI)",
          "STACK4+META",
          "GATED(base+cxr+int)"]:
    m, ci = boot(y, M[k], ref, g)
    s = "*" if ci[0] > 0 else " "
    print("  %-22s %+0.4f [%+0.4f,%+0.4f] %s"
          % (k, m, ci[0], ci[1], s))

print("\n" + "=" * 62)
print("GATED vs RESID (value of the gate)")
print("=" * 62)
m, ci = boot(y, M["GATED(base+cxr+int)"],
             M["RESID(base+cxr)"], g)
print("  %+0.4f [%+0.4f, %+0.4f]"
      % (m, ci[0], ci[1]))

print("\n" + "=" * 62)
print("NRI: reclassification from adding CXR "
      "(EHR+ECG -> +CXR)")
print("=" * 62)
old = oof_platt(M["MEAN2(rank)"], y, g, skf)
new = oof_platt(M["MEAN3(rank)"], y, g, skf)
cuts = [0.05, 0.15]
ne, nn, tot, cnt = nri(y, old, new, cuts)
print("categories: <5%, 5-15%, >=15%")
print("  events   NRI %+0.4f" % ne)
print("  nonevent NRI %+0.4f" % nn)
print("  TOTAL    NRI %+0.4f" % tot)
print("  counts:", cnt)
a, b, c = cont_nri(y, old, new)
print("\ncontinuous NRI:")
print("  events %+0.4f  nonevents %+0.4f "
      " total %+0.4f" % (a, b, c))

print("\nNRI: reclassification from adding sPESI "
      "(MEAN3 -> MEAN4)")
new4 = oof_platt(
    M["MEAN4(rank,+sPESI)"], y, g, skf)
ne, nn, tot, cnt = nri(y, new, new4, cuts)
print("  events %+0.4f  nonevent %+0.4f "
      " TOTAL %+0.4f" % (ne, nn, tot))
print("  counts:", cnt)

pd.DataFrame(rows).to_csv(
    os.path.join(FD, "fusion_improvements.csv"),
    index=False
)
print("\nSaved fusion_improvements.csv")
