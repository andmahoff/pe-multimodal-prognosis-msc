import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
CUTS = [0.05, 0.15]


def rank01(x):
    return pd.Series(x).rank(pct=True).values


def oof_platt(x, y, g, skf):
    x = np.asarray(x).reshape(-1, 1)
    out = np.zeros(len(y))
    for tr, te in skf.split(x, y, groups=g):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(x[tr], y[tr])
        out[te] = lr.predict_proba(x[te])[:, 1]
    return out


def nri_parts(y, po, pn, cuts):
    co = np.digitize(po, cuts)
    cn = np.digitize(pn, cuts)
    ev = y == 1
    ne = y == 0
    if ev.sum() == 0 or ne.sum() == 0:
        return np.nan, np.nan, np.nan
    ue = (cn[ev] > co[ev]).sum()
    de = (cn[ev] < co[ev]).sum()
    un = (cn[ne] > co[ne]).sum()
    dn = (cn[ne] < co[ne]).sum()
    a = (ue - de) / ev.sum()
    b = (dn - un) / ne.sum()
    return a, b, a + b


def nri_boot(y, po, pn, g, n=NBOOT):
    rng = np.random.RandomState(SEED)
    g = np.asarray(g)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    E, N, T = [], [], []
    for _ in range(n):
        s = rng.choice(uq, len(uq),
                       replace=True)
        t = np.concatenate([idx[u] for u in s])
        a, b, c = nri_parts(
            y[t], po[t], pn[t], CUTS
        )
        if np.isnan(c):
            continue
        E.append(a)
        N.append(b)
        T.append(c)
    return (np.array(E), np.array(N),
            np.array(T))


def show(tag, arr):
    ci = np.percentile(arr, [2.5, 97.5])
    print("  %-14s %+0.4f "
          "[%+0.4f, %+0.4f]  P(>0)=%.3f"
          % (tag, arr.mean(), ci[0], ci[1],
             (arr > 0).mean()))


def boot_auc(y, pa, pb, g, n=NBOOT):
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


def xtab(y, po, pn, label):
    co = np.digitize(po, CUTS)
    cn = np.digitize(pn, CUTS)
    names = ["<5%", "5-15%", ">=15%"]
    for lab, mask in [("EVENTS", y == 1),
                      ("NON-EVENTS", y == 0)]:
        t = pd.crosstab(
            pd.Series(
                [names[i] for i in co[mask]],
                name="old"
            ),
            pd.Series(
                [names[i] for i in cn[mask]],
                name="new"
            )
        )
        print("\n  %s -- %s" % (label, lab))
        print(t.to_string())


print("=" * 62)
print("PART 1: NRI with bootstrap CIs")
print("=" * 62)
d = pd.read_csv(
    os.path.join(FD, "ro4_fusion_table.csv")
)
y = d["label"].values
g = d["subject_id"].values
print("n=%d events=%d" % (len(d), int(y.sum())))

d["r_spesi"] = rank01(d["spesi"].fillna(0))
m2 = (d["r_ehr"] + d["r_ecg"]) / 2
m3 = (d["r_ehr"] + d["r_ecg"]
      + d["r_cxr"]) / 3
m4 = (d["r_ehr"] + d["r_ecg"]
      + d["r_cxr"] + d["r_spesi"]) / 4

skf = StratifiedGroupKFold(
    n_splits=5, shuffle=True,
    random_state=SEED
)
c2 = oof_platt(m2, y, g, skf)
c3 = oof_platt(m3, y, g, skf)
c4 = oof_platt(m4, y, g, skf)

print("\n--- adding CXR (MEAN2 -> MEAN3) ---")
a, b, c = nri_parts(y, c2, c3, CUTS)
print("  point: events %+0.4f  "
      "nonevents %+0.4f  total %+0.4f"
      % (a, b, c))
E, N, T = nri_boot(y, c2, c3, g)
show("events NRI", E)
show("nonevent NRI", N)
show("TOTAL NRI", T)
xtab(y, c2, c3, "CXR added")

print("\n--- adding sPESI (MEAN3 -> MEAN4) ---")
a, b, c = nri_parts(y, c3, c4, CUTS)
print("  point: events %+0.4f  "
      "nonevents %+0.4f  total %+0.4f"
      % (a, b, c))
E, N, T = nri_boot(y, c3, c4, g)
show("events NRI", E)
show("nonevent NRI", N)
show("TOTAL NRI", T)
xtab(y, c3, c4, "sPESI added")

print("\n\n" + "=" * 62)
print("PART 2: MEAN3 on the FULL 1,797 cohort")
print("=" * 62)

tim = pd.read_csv(
    os.path.join(FD, "cxr_image_timing.csv")
)
cxa = tim.groupby(KEY).agg(
    p_cxr=("p_cxr", "mean"),
    frac_port=("port", "mean"),
    n_all=("p_cxr", "size"),
    label=("mace_30d_label", "max"),
).reset_index()
print("CXR admissions:", len(cxa))

ecg = pd.read_csv(
    os.path.join(P2, "logit_final_pecg.csv")
)
ecga = ecg.groupby(KEY).agg(
    p_ecg=("p_ecg", "mean")
).reset_index()
eh = pd.read_csv(os.path.join(
    FD, "p_ehr_ensemble_v7_vrex.csv"
))[KEY + ["p_ehr_mean"]]
v6 = pd.read_csv(
    os.path.join(
        P2, "mimic_pe_ehr_baseline_v6.csv"
    )
).drop_duplicates(KEY)
v6["spesi"] = (
    (v6["age_at_admit"] > 80).astype(int)
    + (v6["cancer"] > 0).astype(int)
    + (((v6["heart_failure"] > 0)
        | (v6["copd"] > 0))).astype(int)
    + (v6["mean_hr"] >= 110).fillna(
        False).astype(int)
    + (v6["mean_sbp"] < 100).fillna(
        False).astype(int)
    + (v6["mean_spo2"] < 90).fillna(
        False).astype(int)
)

w = cxa.merge(ecga, on=KEY, how="inner")
w = w.merge(eh, on=KEY, how="inner")
w = w.merge(v6[KEY + ["spesi"]], on=KEY,
            how="left")
w = w.dropna(subset=[
    "p_ehr_mean", "p_ecg", "p_cxr"
]).reset_index(drop=True)

yw = w["label"].values
gw = w["subject_id"].values
print("cohort n=%d events=%d rate=%.4f"
      % (len(w), int(yw.sum()), yw.mean()))

w["r_ehr"] = rank01(w["p_ehr_mean"])
w["r_ecg"] = rank01(w["p_ecg"])
w["r_cxr"] = rank01(w["p_cxr"])
w["r_spesi"] = rank01(w["spesi"].fillna(0))

W = {
    "EHR": w["r_ehr"].values,
    "ECG": w["r_ecg"].values,
    "CXR": w["r_cxr"].values,
    "sPESI": w["r_spesi"].values,
}
W["MEAN2"] = (w["r_ehr"] + w["r_ecg"]).values / 2
W["MEAN3"] = (
    w["r_ehr"] + w["r_ecg"] + w["r_cxr"]
).values / 3
W["MEAN4"] = (
    w["r_ehr"] + w["r_ecg"]
    + w["r_cxr"] + w["r_spesi"]
).values / 4

print("\n%-8s %-8s %-8s" % ("model", "AUC", "AUPRC"))
for k, v in W.items():
    print("%-8s %.4f   %.4f"
          % (k, roc_auc_score(yw, v),
             average_precision_score(yw, v)))

best = max(["EHR", "ECG", "CXR"],
           key=lambda k: roc_auc_score(yw, W[k]))
print("\nbest unimodal:", best)
print("\npaired bootstrap vs %s:" % best)
for k in ["MEAN2", "MEAN3", "MEAN4"]:
    m, ci = boot_auc(yw, W[k], W[best], gw)
    s = "*" if ci[0] > 0 else " "
    print("  %-6s %+0.4f [%+0.4f, %+0.4f] %s"
          % (k, m, ci[0], ci[1], s))

print("\nCXR increment (MEAN3 vs MEAN2):")
m, ci = boot_auc(yw, W["MEAN3"], W["MEAN2"], gw)
print("  %+0.4f [%+0.4f, %+0.4f]"
      % (m, ci[0], ci[1]))

print("\ncomparison to primary cohort:")
print("  1,027 (index CXR):  MEAN3 0.7434")
print("  1,797 (any CXR):    MEAN3 %.4f"
      % roc_auc_score(yw, W["MEAN3"]))

w.to_csv(
    os.path.join(FD, "fusion_1797_table.csv"),
    index=False
)
print("\nSaved fusion_1797_table.csv")
