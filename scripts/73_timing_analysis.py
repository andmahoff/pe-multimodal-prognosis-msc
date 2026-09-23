import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.duration.survfunc import SurvfuncRight
from statsmodels.duration.hazard_regression import PHReg

FD = (
    "./"
    "fusion_workspace/data/"
)
KEY = ["subject_id", "hadm_id"]
SEED = 42
NBOOT = 1000


def auc_sub(y_all, p_all, keep):
    y = y_all[keep]
    p = p_all[keep]
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def boot_auc_sub(y, p, keep, g, n=NBOOT):
    rng = np.random.RandomState(SEED)
    g = np.asarray(g)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    o = []
    for _ in range(n):
        s = rng.choice(uq, len(uq),
                       replace=True)
        t = np.concatenate([idx[u] for u in s])
        kk = keep[t]
        if kk.sum() < 10:
            continue
        yy = y[t][kk]
        if len(np.unique(yy)) < 2:
            continue
        o.append(roc_auc_score(yy, p[t][kk]))
    o = np.array(o)
    if len(o) == 0:
        return np.nan, (np.nan, np.nan)
    return o.mean(), np.percentile(
        o, [2.5, 97.5]
    )


def cindex(t, e, s):
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=int)
    s = np.asarray(s, dtype=float)
    ev = np.where(e == 1)[0]
    conc = 0.0
    perm = 0.0
    for i in ev:
        later = t > t[i]
        k = later.sum()
        if k == 0:
            continue
        perm += k
        conc += (s[i] > s[later]).sum()
        conc += 0.5 * (s[i] == s[later]).sum()
    if perm == 0:
        return np.nan
    return conc / perm


def analyse(name, d, score, tm):
    print("\n" + "=" * 64)
    print(name)
    print("=" * 64)

    m = d.merge(tm, on=KEY, how="left")
    m = m.dropna(
        subset=["event"]
    ).reset_index(drop=True)
    y = m["event"].values.astype(int)
    g = m["subject_id"].values
    p = m[score].values
    days = m["days_to_mace"].fillna(30).values
    days = np.where(days < 0.5, 0.5, days)
    vd = m["via_death"].fillna(0).values

    print("n=%d events=%d overall AUC %.4f"
          % (len(m), int(y.sum()),
             roc_auc_score(y, p)))

    print("\n-- BY ROUTE (vs all non-events) --")
    for lab, mask in [
        ("death route",
         (y == 0) | ((y == 1) & (vd == 1))),
        ("readmit route",
         (y == 0) | ((y == 1) & (vd == 0))),
    ]:
        a = auc_sub(y, p, mask)
        bm, ci = boot_auc_sub(y, p, mask, g)
        ne = int(y[mask].sum())
        print("  %-14s n_ev=%-4d AUC %.4f "
              "[%.4f, %.4f]"
              % (lab, ne, a, ci[0], ci[1]))

    print("\n-- BY TIMING (vs all non-events) --")
    bins = [(-0.01, 7, "0-7"),
            (7, 14, "8-14"),
            (14, 30.01, "15-30")]
    for lo, hi, lab in bins:
        mask = (y == 0) | (
            (y == 1) & (days > lo)
            & (days <= hi)
        )
        a = auc_sub(y, p, mask)
        bm, ci = boot_auc_sub(y, p, mask, g)
        ne = int(y[mask].sum())
        print("  %-6s days n_ev=%-4d AUC %.4f "
              "[%.4f, %.4f]"
              % (lab, ne, a, ci[0], ci[1]))

    print("\n-- ROUTE x TIMING counts --")
    ev = y == 1
    tb = pd.cut(
        days[ev], [-0.01, 7, 14, 30.01],
        labels=["0-7", "8-14", "15-30"]
    )
    rt = np.where(vd[ev] == 1, "death",
                  "readmit")
    print(pd.crosstab(
        pd.Series(rt, name="route"),
        pd.Series(tb, name="days")
    ).to_string())

    print("\n-- mean predicted risk --")
    print("  non-events        %.4f"
          % p[y == 0].mean())
    print("  events via death  %.4f"
          % p[(y == 1) & (vd == 1)].mean())
    print("  events via readm  %.4f"
          % p[(y == 1) & (vd == 0)].mean())
    for lo, hi, lab in bins:
        sel = ev & (days > lo) & (days <= hi)
        print("  events %-6s     %.4f"
              % (lab, p[sel].mean()))

    r = pd.Series(p[ev]).corr(
        pd.Series(days[ev]), method="spearman"
    )
    print("\n  spearman(risk, days) among "
          "events: %+.3f" % r)

    print("\n-- SURVIVAL --")
    ter = pd.qcut(
        pd.Series(p), 3,
        labels=["low", "mid", "high"]
    )
    ter = np.asarray(ter)
    for lab in ["low", "mid", "high"]:
        sel = ter == lab
        if sel.sum() == 0:
            continue
        sf = SurvfuncRight(days[sel], y[sel])
        out = []
        for dd in [7, 14, 30]:
            ix = np.searchsorted(
                sf.surv_times, dd, side="right"
            ) - 1
            v = (sf.surv_prob[ix]
                 if ix >= 0 else 1.0)
            out.append(1 - v)
        print("  %-5s n=%-5d ev=%-4d "
              "cum inc d7 %.3f d14 %.3f "
              "d30 %.3f"
              % (lab, int(sel.sum()),
                 int(y[sel].sum()),
                 out[0], out[1], out[2]))

    z = (p - p.mean()) / p.std()
    mod = PHReg(days, z.reshape(-1, 1),
                status=y)
    res = mod.fit(disp=False)
    hr = np.exp(res.params[0])
    lo_, hi_ = np.exp(res.conf_int()[0])
    print("\n  Cox HR per 1 SD of score: "
          "%.3f [%.3f, %.3f]  p=%.2e"
          % (hr, lo_, hi_, res.pvalues[0]))
    print("  Harrell C-index: %.4f"
          % cindex(days, y, p))


tm = pd.read_csv(
    os.path.join(FD, "time_to_mace.csv")
)[KEY + ["days_to_mace", "event",
         "via_death"]]

d2 = pd.read_csv(
    os.path.join(FD, "rh2_twoway_table.csv")
)
analyse("TWO-WAY COHORT (n=3512) MEAN2",
        d2, "MEAN2", tm)

d3 = pd.read_csv(
    os.path.join(FD, "ro4_fusion_table.csv")
)
analyse("THREE-WAY COHORT (n=1027) MEAN3",
        d3, "MEAN3", tm)

print("\nDone.")
