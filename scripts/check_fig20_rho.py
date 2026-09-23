import os
import pandas as pd
from scipy.stats import spearmanr

DIRS = ["tables", "scripts/figures"]

for o in ["death_30d", "composite_30d"]:
    for dr in DIRS:
        f = "%s/fig20_importance_%s.csv" % (dr, o)
        if not os.path.exists(f):
            print("missing:", f)
            continue
        d = pd.read_csv(f)
        print("===", f, "| rows", len(d))
        print("cols:", list(d.columns))
        num = d.select_dtypes("number").columns.tolist()
        src = [c for c in num
               if "source" in c or "inspect" in c]
        tgt = [c for c in num
               if "target" in c or "mimic" in c]
        if src and tgt:
            r = spearmanr(d[src[0]], d[tgt[0]]).correlation
            print("rho(%s, %s) = %.4f"
                  % (src[0], tgt[0], r))
        else:
            print("numeric cols:", num)
        txt = d.astype(str)
        m = txt.apply(
            lambda x: x.str.contains("anion", case=False))
        m = m.any(axis=1)
        if m.any():
            print(d[m].to_string(index=False))
        print()

