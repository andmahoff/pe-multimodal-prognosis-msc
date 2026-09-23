import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
LAB = DATA + "/mimic_labels_harmonised.csv"
FEAT = DATA + "/ctpa_comorb_features.csv"
OUT = DATA + "/ctpa_fusion_pres.csv"

SEED = 42
NBOOT = 2000
STEP = 0.05
VARS = ["base_cm", "base_cm_dv"]
OUTS = ["cv_first", "composite_30d",
        "death_30d", "death_30d_inhosp"]
EHRMAP = {"death_30d_inhosp": "death_30d"}


def rk(x):
    return rankdata(x) / (len(x) + 1.0)


def grid3():
    g = []
    n = int(round(1.0 / STEP))
    for i in range(n + 1):
        for j in range(n - i + 1):
            a, b = i * STEP, j * STEP
            g.append((a, b, max(1.0 - a - b, 0.0)))
    return g


G3 = grid3()
G2 = [(w, 1.0 - w, 0.0)
      for w in np.arange(0, 1.0001, STEP)]


def wf(cols, w):
    s = np.zeros(len(cols[0]))
    for c, wi in zip(cols, w):
        s = s + wi * c
    return s


def oof_w(cols, y, grp, grid):
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED)
    oof = np.zeros(len(y))
    ws = []
    X0 = cols[0].reshape(-1, 1)
    for tr, te in cv.split(X0, y, grp):
        best, bw = None, None
        for w in grid:
            a = roc_auc_score(
                y[tr], wf([c[tr] for c in cols], w))
            if best is None or a > best:
                best, bw = a, w
        oof[te] = wf([c[te] for c in cols], bw)
        ws.append(tuple(round(x, 2) for x in bw))
    return oof, ws


def boot(y, pa, pb, grp):
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(NBOOT):
        pk = rng.choice(us, len(us), replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(roc_auc_score(y[ii], pa[ii])
                 - roc_auc_score(y[ii], pb[ii]))
    d = np.array(d)
    return (float(np.mean(d)),
            float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


lab = pd.read_csv(LAB)
fx = pd.read_csv(FEAT)[["hadm_id", "txt_len",
                        "pe_pos"]]

rows = []
for out in OUTS:
    eh = EHRMAP.get(out, out)
    pe = pd.read_csv(DATA + "/p_ehr_harm_" + eh
                     + ".csv")[["hadm_id",
                                "p_ehr"]]
    pg = pd.read_csv(DATA + "/p_ecg_harm_" + out
                     + ".csv")[["hadm_id",
                                "p_ecg"]]
    base = lab.merge(pe, on="hadm_id")
    base = base.merge(pg, on="hadm_id")
    base = base.merge(fx, on="hadm_id")

    for v in VARS:
        pth = (DATA + "/p_ctpa_pres_" + v + "_"
               + out + ".csv")
        pc = pd.read_csv(pth)[["hadm_id",
                               "p_ctpa"]]
        d = base.merge(pc, on="hadm_id")
        if out == "cv_first":
            d = d[d["death_first"] == 0]
        d = d.reset_index(drop=True)
        y = d[out].values.astype(int)
        grp = d["subject_id"].values
        if y.sum() < 20:
            continue

        a = rk(d["p_ehr"].values)
        b = rk(d["p_ecg"].values)
        c = rk(d["p_ctpa"].values)
        m = rk(d["txt_len"].values)

        print("")
        print("=" * 58)
        print("%s | %s  n=%d ev=%d (%.4f)"
              % (out, v, len(y), int(y.sum()),
                 y.mean()))
        print("  EHR  %.4f" % roc_auc_score(y, a))
        print("  ECG  %.4f" % roc_auc_score(y, b))
        print("  CTPA %.4f" % roc_auc_score(y, c))
        print("  META %.4f" % roc_auc_score(y, m))

        w2, _ = oof_w([a, b], y, grp, G2)
        w3, ws3 = oof_w([a, b, c], y, grp, G3)
        wm, _ = oof_w([a, b, m], y, grp, G3)
        A2 = roc_auc_score(y, w2)
        A3 = roc_auc_score(y, w3)
        AM = roc_auc_score(y, wm)
        print("  WMEAN2      %.4f" % A2)
        print("  WMEAN3+CTPA %.4f" % A3)
        print("  WMEAN3+META %.4f" % AM)
        print("  w3:", ws3)

        t1 = boot(y, w3, w2, grp)
        t2 = boot(y, w3, wm, grp)
        print("  increment vs WMEAN2 "
              "%+.4f [%+.4f, %+.4f]" % t1)
        print("  vs META fusion      "
              "%+.4f [%+.4f, %+.4f]" % t2)
        r1 = spearmanr(d["p_ehr"],
                       d["p_ctpa"])[0]
        print("  spearman ehr-ctpa %.3f" % r1)

        rows.append({"outcome": out, "var": v,
                     "n": len(y),
                     "ev": int(y.sum()),
                     "ehr": roc_auc_score(y, a),
                     "ecg": roc_auc_score(y, b),
                     "ctpa": roc_auc_score(y, c),
                     "wmean2": A2, "wmean3": A3,
                     "wmean3_meta": AM,
                     "inc": t1[0], "lo": t1[1],
                     "hi": t1[2],
                     "vs_meta": t2[0],
                     "vm_lo": t2[1],
                     "vm_hi": t2[2],
                     "r_ehr": r1})

        ok = (d["pe_pos"] == 1).values
        y2 = y[ok]
        if y2.sum() >= 20:
            g2 = grp[ok]
            v2, _ = oof_w([a[ok], b[ok]], y2,
                          g2, G2)
            v3, _ = oof_w([a[ok], b[ok], c[ok]],
                          y2, g2, G3)
            t3 = boot(y2, v3, v2, g2)
            print("  [pe_pos] n=%d ev=%d "
                  "WMEAN2 %.4f WMEAN3 %.4f"
                  % (len(y2), int(y2.sum()),
                     roc_auc_score(y2, v2),
                     roc_auc_score(y2, v3)))
            print("    increment %+.4f "
                  "[%+.4f, %+.4f]" % t3)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print("")
print(res.round(4).to_string())
print("saved", OUT)
