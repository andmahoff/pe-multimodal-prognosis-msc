import pandas as pd
import numpy as np

BASE = "."
DATA = BASE + "/fusion_workspace/data"

nt = pd.read_csv(DATA + "/ctpa_notes_index.csv")
fx = pd.read_csv(DATA
                 + "/ctpa_comorb_features.csv")

for c in ["charttime", "admittime"]:
    nt[c] = pd.to_datetime(nt[c],
                           errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm", as_index=False).first()
nt["h"] = ((nt["charttime"] - nt["admittime"])
           .dt.total_seconds() / 3600.0)

print("hours from admittime to first CTPA")
print(nt["h"].describe().round(2))
print("")
for lo, hi in [(-48, 0), (0, 6), (6, 24),
               (24, 72), (72, 1e9)]:
    m = (nt["h"] >= lo) & (nt["h"] < hi)
    print("%6.0f to %6.0f h: %5d (%.3f)"
          % (lo, hi, int(m.sum()),
             float(m.mean())))

d = nt[["idx_hadm", "h"]].rename(
    columns={"idx_hadm": "hadm_id"})
d = fx.merge(d, on="hadm_id")
print("")
print("ETT rate by timing band:")
for lo, hi in [(-48, 0), (0, 6), (6, 24),
               (24, 72), (72, 1e9)]:
    m = (d["h"] >= lo) & (d["h"] < hi)
    if m.sum() < 10:
        continue
    print("%6.0f to %6.0f h: n=%4d  ett %.3f"
          % (lo, hi, int(m.sum()),
             float(d.loc[m, "dv_ett"].mean())))

