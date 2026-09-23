import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

BASE = "."
DATA = BASE + "/fusion_workspace/data"
LAB = DATA + "/mimic_labels_harmonised.csv"
FEAT = DATA + "/ctpa_routeA_features.csv"
OUT = DATA + "/ctpa_fusion_results.csv"

SEED = 42
NBOOT = 2000
STEP = 0.05

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
            a = i * STEP
            b = j * STEP
            c = 1.0 - a - b
            if c < -1e-9:
                continue
            g.append((a, b, max(c, 0.0)))
    return g


G3 = grid3()
G2 = [(w, 1.0 - w, 0.0)
      for w in np.arange(0, 1.0001, STEP)]


def wfuse(cols, w):
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
    for tr, te in cv.split(cols[0].reshape(-1, 1),
                           y, grp):
        best = None
        bw = None
        for w in grid:
            s = wfuse([c[tr] for c in cols], w)
            a = roc_auc_score(y[tr], s)
            if best is None or a > best:
                best = a
                bw = w
        oof[te] = wfuse([c[te] for c in cols], bw)
        ws.append(tuple(round(x, 2) for x in bw))
    return oof, ws


def boot(y, pa, pb, grp):
    us = np.unique(grp)
    idx = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(NBOOT):
        pick = rng.choice(us, len(us), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(roc_auc_score(y[ii], pa[ii])
                 - roc_auc_score(y[ii], pb[ii]))
    d = np.array(d)
    return (float(np.mean(d)),
            float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


lab = pd.read_csv(LAB)
fx = pd.read_csv(FEAT)
fx = fx[["hadm_id", "imp_len", "pe_pos"]]

rows = []
for out in OUTS:
    eh = EHRMAP.get(out, out)
    pe = pd.read_csv(DATA + "/p_ehr_harm_"
                     + eh + ".csv")
    pg = pd.read_csv(DATA + "/p_ecg_harm_"
                     + out + ".csv")
    pc = pd.read_csv(DATA + "/p_ctpa_routeA_"
                     + out + ".csv")
    pe = pe[["hadm_id", "p_ehr"]]
    pg = pg[["hadm_id", "p_ecg"]]
    pc = pc[["hadm_id", "p_ctpa"]]

    d = lab.merge(pe, on="hadm_id")
    d = d.merge(pg, on="hadm_id")
    d = d.merge(pc, on="hadm_id")
    d = d.merge(fx, on="hadm_id")
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
    m = rk(d["imp_len"].values)

    print("")
    print("=" * 58)
    print(out, "n=%d ev=%d (%.4f)"
          % (len(y), int(y.sum()), y.mean()))
    print("  EHR  %.4f" % roc_auc_score(y, a))
    print("  ECG  %.4f" % roc_auc_score(y, b))
    print("  CTPA %.4f" % roc_auc_score(y, c))
    print("  META %.4f" % roc_auc_score(y, m))

    w2, ws2 = oof_w([a, b], y, grp, G2)
    w3, ws3 = oof_w([a, b, c], y, grp, G3)
    wm, wsm = oof_w([a, b, m], y, grp, G3)

    A2 = roc_auc_score(y, w2)
    A3 = roc_auc_score(y, w3)
    AM = roc_auc_score(y, wm)
    print("  WMEAN2      %.4f" % A2)
    print("  WMEAN3+CTPA %.4f" % A3)
    print("  WMEAN3+META %.4f" % AM)
    print("  w2:", ws2)
    print("  w3:", ws3)

    t1 = boot(y, w3, w2, grp)
    t2 = boot(y, w3, wm, grp)
    print("  CTPA increment vs WMEAN2 "
          "%+.4f [%+.4f, %+.4f]" % t1)
    print("  CTPA vs META fusion      "
          "%+.4f [%+.4f, %+.4f]" % t2)

    r1 = spearmanr(d["p_ehr"], d["p_ctpa"])[0]
    r2 = spearmanr(d["p_ecg"], d["p_ctpa"])[0]
    print("  spearman ehr-ctpa %.3f"
          "  ecg-ctpa %.3f" % (r1, r2))

    rows.append({
        "outcome": out, "n": len(y),
        "ev": int(y.sum()),
        "ehr": roc_auc_score(y, a),
        "ecg": roc_auc_score(y, b),
        "ctpa": roc_auc_score(y, c),
        "meta": roc_auc_score(y, m),
        "wmean2": A2, "wmean3": A3,
        "wmean3_meta": AM,
        "inc": t1[0], "inc_lo": t1[1],
        "inc_hi": t1[2],
        "vs_meta": t2[0], "vm_lo": t2[1],
        "vm_hi": t2[2],
        "r_ehr_ctpa": r1, "r_ecg_ctpa": r2})

    # pe_pos == 1 only
    mk = (d["pe_pos"] == 1).values
    y2 = y[mk]
    if y2.sum() >= 20:
        g2 = grp[mk]
        a2 = rk(d["p_ehr"].values[mk])
        b2 = rk(d["p_ecg"].values[mk])
        c2 = rk(d["p_ctpa"].values[mk])
        v2, _ = oof_w([a2, b2], y2, g2, G2)
        v3, _ = oof_w([a2, b2, c2], y2, g2, G3)
        t3 = boot(y2, v3, v2, g2)
        print("  [pe_pos only] n=%d ev=%d"
              % (len(y2), int(y2.sum())))
        print("    WMEAN2 %.4f  WMEAN3 %.4f"
              % (roc_auc_score(y2, v2),
                 roc_auc_score(y2, v3)))
        print("    increment %+.4f "
              "[%+.4f, %+.4f]" % t3)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print("")
print(res.round(4).to_string())
print("saved", OUT)
