import sys
import inspect
import numpy as np
import pandas as pd

BASE = "."
FIGS = BASE + "/fusion_workspace/scripts/figures"
OUT = BASE + "/fusion_workspace/data/table41_baseline.csv"
sys.path.insert(0, FIGS)

import fig_feats as ff

print("load_pair signature:")
print("  ", inspect.signature(ff.load_pair))
print()

out = ff.load_pair("death_30d")
if not isinstance(out, tuple):
    out = (out,)
for i, o in enumerate(out):
    print(i, type(o).__name__, getattr(o, "shape", ""))
print()

frames = [o for o in out if hasattr(o, "columns")]
if len(frames) < 2:
    raise SystemExit("expected two feature matrices")

Xs = frames[0]
Xt = frames[1]
print("source:", Xs.shape, " target:", Xt.shape)
print("cols:", list(Xs.columns))
print()

LABELS = {
    "age": "Age, years",
    "age_at_admit": "Age, years",
    "mean_hr": "Heart rate, bpm",
    "hr": "Heart rate, bpm",
    "mean_sbp": "Systolic BP, mmHg",
    "sbp": "Systolic BP, mmHg",
    "mean_dbp": "Diastolic BP, mmHg",
    "dbp": "Diastolic BP, mmHg",
    "mean_rr": "Respiratory rate, /min",
    "rr": "Respiratory rate, /min",
    "mean_temp": "Temperature, C",
    "temp": "Temperature, C",
    "aniongap": "Anion gap, mmol/L",
    "bun": "Urea nitrogen, mg/dL",
    "creatinine": "Creatinine, mg/dL",
    "sodium": "Sodium, mmol/L",
    "chloride": "Chloride, mmol/L",
    "bicarbonate": "Bicarbonate, mmol/L",
    "potassium": "Potassium, mmol/L",
    "calcium": "Calcium, mg/dL",
    "glucose": "Glucose, mg/dL",
    "hct": "Haematocrit, %",
    "hematocrit": "Haematocrit, %",
    "hgb": "Haemoglobin, g/dL",
    "hemoglobin": "Haemoglobin, g/dL",
    "mch": "MCH, pg",
    "mchc": "MCHC, g/dL",
    "mcv": "MCV, fL",
    "rbc": "Erythrocytes, m/uL",
    "platelets": "Platelets, k/uL",
    "wbc": "White cell count, k/uL",
    "cancer": "Malignancy",
    "heart_failure": "Heart failure",
    "copd": "COPD",
    "afib": "Atrial fibrillation",
}


def pretty(c):
    if c in LABELS:
        return LABELS[c]
    return c.replace("_", " ").capitalize()


def is_binary(a, b):
    u = set(pd.concat([a, b]).dropna().unique())
    return u.issubset({0, 1, 0.0, 1.0})


def fmt_cont(s):
    return "%.1f (%.1f to %.1f)" % (
        s.median(), s.quantile(0.25), s.quantile(0.75))


def fmt_bin(s):
    return "%d (%.1f%%)" % (int(s.sum()), 100.0 * s.mean())


def smd_cont(a, b):
    v = (a.var(ddof=1) + b.var(ddof=1)) / 2.0
    if v <= 0:
        return np.nan
    return (b.mean() - a.mean()) / np.sqrt(v)


def smd_bin(a, b):
    p1 = a.mean()
    p2 = b.mean()
    v = (p1 * (1 - p1) + p2 * (1 - p2)) / 2.0
    if v <= 0:
        return np.nan
    return (p2 - p1) / np.sqrt(v)


rows = []
for c in [x for x in Xs.columns if x in Xt.columns]:
    a = Xs[c].dropna()
    b = Xt[c].dropna()
    if len(a) < 10 or len(b) < 10:
        continue
    if is_binary(a, b):
        d = smd_bin(a, b)
        src = fmt_bin(a)
        tgt = fmt_bin(b)
        kind = "binary"
    else:
        d = smd_cont(a, b)
        src = fmt_cont(a)
        tgt = fmt_cont(b)
        kind = "continuous"
    rows.append({
        "feature": pretty(c),
        "raw": c,
        "type": kind,
        "inspect": src,
        "mimic": tgt,
        "smd": round(float(d), 3),
    })

res = pd.DataFrame(rows)
res["k"] = res["smd"].abs()
res = res.sort_values("k", ascending=False)
res = res.drop(columns=["k"])

pd.set_option("display.width", 200)
print(res.to_string(index=False))
res.to_csv(OUT, index=False)
print()
print("n source:", len(Xs), " n target:", len(Xt))
print("saved:", OUT)

