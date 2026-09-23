import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
LAB = DATA + "/mimic_labels_harmonised.csv"
FEAT = DATA + "/mimic_imp_features_v2.csv"
OUT = DATA + "/zeroshot_fusion_results.csv"

SEED = 42
NBOOT = 2000
STEP = 0.05

JOBS = [
    ("death_30d", "pe_native_1m_mortality"),
    ("death_30d", "pe_harm_death_30d"),
    ("composite_30d", "pe_harm_composite_30d"),
    ("cv_first", "pe_harm_cv_first"),
]
EHRMAP = {}


def rk(x):
    return rankdata(x) / (len(x) + 1.0)


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


def wmean_oof(a, b, y, grp):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    oof = np.zeros(len(y))
    ws = []
    X0 = a.reshape(-1, 1)
    for tr, te in cv.split(X0, y, grp):
        best, bw = None, None
        for w in np.arange(0, 1.0001, STEP):
            s = w * a[tr] + (1 - w) * b[tr]
            v = roc_auc_score(y[tr], s)
            if best is None or v > best:
                best, bw = v, w
        oof[te] = bw * a[te] + (1 - bw) * b[te]
        ws.append(round(bw, 2))
    return oof, ws


lab = pd.read_csv(LAB)
fx = pd.read_csv(FEAT)[["hadm_id"]]
print("presentation cohort:", len(fx))

rows = []
for ycol, tag in JOBS:
    zp = DATA + "/p_ctpa_zs_" + tag + ".csv"
    ep = DATA + "/p_ehr_harm_" + \
        EHRMAP.get(ycol, ycol) + ".csv"
    if not os.path.exists(zp):
        print("missing", zp)
        continue
    zs = pd.read_csv(zp)
    eh = pd.read_csv(ep)[["hadm_id", "p_ehr"]]

    d = lab.merge(fx, on="hadm_id")
    d = d.merge(eh, on="hadm_id")
    d = d.merge(zs[["hadm_id", "p_zs_A",
                    "p_zs_B"]], on="hadm_id")
    if ycol == "cv_first":
        d = d[d["death_first"] == 0]
    d = d.reset_index(drop=True)

    y = d[ycol].values.astype(int)
    grp = d["subject_id"].values
    if y.sum() < 20:
        continue

    e = rk(d["p_ehr"].values)
    cA = rk(d["p_zs_A"].values)
    cB = rk(d["p_zs_B"].values)

    print("")
    print("=" * 58)
    print("%s | %s   n=%d ev=%d (%.4f)"
          % (ycol, tag, len(y), int(y.sum()),
             y.mean()))
    print("  p_ehr  (zero-shot) %.4f"
          % roc_auc_score(y, e))
    print("  p_ctpa (zero-shot) %.4f"
          % roc_auc_score(y, cA))

    for nm, c in [("A", cA), ("B", cB)]:
        m2 = (e + c) / 2.0
        aM = roc_auc_score(y, m2)
        apM = average_precision_score(y, m2)
        t1 = boot(y, m2, e, grp)
        print("")
        print("  [%s] MEAN2 (zero-shot) %.4f"
              "  AP %.4f" % (nm, aM, apM))
        print("       vs p_ehr  %+.4f"
              " [%+.4f, %+.4f]" % t1)

        w2, ws = wmean_oof(e, c, y, grp)
        aW = roc_auc_score(y, w2)
        t2 = boot(y, w2, e, grp)
        print("  [%s] WMEAN2 (not zero-shot)"
              " %.4f" % (nm, aW))
        print("       weights(ehr):", ws)
        print("       vs p_ehr  %+.4f"
              " [%+.4f, %+.4f]" % t2)

        rows.append({
            "outcome": ycol, "model": tag,
            "zs_var": nm, "n": len(y),
            "ev": int(y.sum()),
            "ehr": roc_auc_score(y, e),
            "ctpa": roc_auc_score(y, c),
            "mean2_zs": aM, "mean2_ap": apM,
            "mean2_vs_ehr": t1[0],
            "m_lo": t1[1], "m_hi": t1[2],
            "wmean2_sup": aW,
            "wmean2_vs_ehr": t2[0],
            "w_lo": t2[1], "w_hi": t2[2]})

    r = spearmanr(d["p_ehr"], d["p_zs_A"])[0]
    print("")
    print("  spearman ehr-ctpa %.3f" % r)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print("")
print(res.round(4).to_string())

print("")
print("=" * 58)
print("TIER SUMMARY (death_30d)")
s = res[(res["outcome"] == "death_30d")
        & (res["zs_var"] == "A")]
if len(s):
    r0 = s.iloc[0]
    print("  1. EHR alone, zero-shot      %.4f"
          % r0["ehr"])
    print("  2. CTPA alone, zero-shot     %.4f"
          % r0["ctpa"])
    print("  3. MEAN2, fully zero-shot    %.4f"
          % r0["mean2_zs"])
    print("  4. WMEAN2, MIMIC-tuned wts   %.4f"
          % r0["wmean2_sup"])
print("saved", OUT)

