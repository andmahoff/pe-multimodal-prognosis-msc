"""Baseline characteristics of the INSPECT and MIMIC-IV
cohorts, sorted by absolute standardised mean difference
(SMD), so the variables whose distributions differ most
between the two sites come first. Writes
table06_baseline.csv.
"""
import numpy as np
import pandas as pd
import fig_feats as ff

OUT = "./fusion_workspace/tables"
Xs, ys, Xt, yt, cols = ff.load_pair("death_30d")
BIN = set(ff.FLAGS)

rows = []
for c in cols:
    a = pd.to_numeric(Xs[c], errors="coerce")
    b = pd.to_numeric(Xt[c], errors="coerce")
    av, bv = a.dropna(), b.dropna()
    if c in BIN:
        p1, p2 = av.mean(), bv.mean()
        den = np.sqrt((p1 * (1 - p1)
                       + p2 * (1 - p2)) / 2.0)
        smd = (p1 - p2) / den if den > 0 else np.nan
        s_txt = "%.1f%%" % (100 * p1)
        t_txt = "%.1f%%" % (100 * p2)
    else:
        m1, m2 = av.mean(), bv.mean()
        den = np.sqrt((av.var(ddof=1)
                       + bv.var(ddof=1)) / 2.0)
        smd = (m1 - m2) / den if den > 0 else np.nan
        s_txt = "%.1f [%.1f-%.1f]" % (
            av.median(), av.quantile(.25),
            av.quantile(.75))
        t_txt = "%.1f [%.1f-%.1f]" % (
            bv.median(), bv.quantile(.25),
            bv.quantile(.75))
    rows.append({
        "Variable": ff.NICEF.get(c, c),
        "INSPECT (n=%d)" % len(Xs): s_txt,
        "INSPECT missing": "%.1f%%"
                           % (100 * a.isna().mean()),
        "MIMIC-IV (n=%d)" % len(Xt): t_txt,
        "MIMIC missing": "%.1f%%"
                         % (100 * b.isna().mean()),
        "SMD": round(float(smd), 3)})

t = pd.DataFrame(rows)
t["abs"] = t["SMD"].abs()
t = t.sort_values("abs", ascending=False).drop(
    columns=["abs"])
t.to_csv(OUT + "/table06_baseline.csv", index=False)
print(t.to_string(index=False))
print("\nvariables with |SMD| > 0.10:",
      int((t["SMD"].abs() > 0.10).sum()), "of", len(t))

