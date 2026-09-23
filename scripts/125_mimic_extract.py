import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJ = os.environ.get("GCP_PROJECT_ID")
BASE = "."
DATA = BASE + "/fusion_workspace/data"
LAB = DATA + "/mimic_labels_harmonised.csv"
ADM = DATA + "/index_admission_times.csv"
OUT = DATA + "/mimic_expanded_feats.csv"

LABS = {
    "creatinine": 50912, "sodium": 50983,
    "potassium": 50971, "bun": 51006,
    "glucose": 50931, "calcium": 50893,
    "bicarb": 50882, "chloride": 50902,
    "hct": 51221, "plt": 51265,
    "wbc": 51301, "hgb": 51222,
    "aniongap": 50868, "rbc": 51279,
    "mchc": 51249, "mch": 51248,
    "mcv": 51250, "rdw": 51277,
    "inr": 51237, "pt": 51274,
    "aptt": 51275, "magnesium": 50960,
    "albumin": 50862, "bili": 50885,
    "alp": 50863, "alt": 50861,
    "ast": 50878, "neut_pct": 51256,
    "lymph_pct": 51244, "mono_pct": 51254,
    "eos_pct": 51200, "baso_pct": 51146,
    "ntprobnp": 50963, "phosphate": 50970,
    "troponin": 51003,
}
CHART = {
    "map": [220052, 220181, 225312],
    "weight": [226512, 224639, 226531],
}

cl = bigquery.Client(project=PROJ)
lab = pd.read_csv(LAB)
adm = pd.read_csv(ADM)
hs = ",".join(str(h) for h in
              sorted(lab["hadm_id"].unique()))
n = lab["hadm_id"].nunique()
print("index admissions:", n)

its = ",".join(str(i) for i in LABS.values())
q = """
SELECT hadm_id, itemid, AVG(valuenum) AS v
FROM `physionet-data.mimiciv_3_1_hosp.labevents`
WHERE hadm_id IN (%s)
  AND itemid IN (%s)
  AND valuenum IS NOT NULL
GROUP BY hadm_id, itemid
""" % (hs, its)
r = cl.query(q).to_dataframe()
print("lab rows:", len(r))
inv = {v: k for k, v in LABS.items()}
r["name"] = r["itemid"].map(inv)
labw = r.pivot_table(index="hadm_id",
                     columns="name",
                     values="v").reset_index()

cits = []
for v in CHART.values():
    cits += v
cs = ",".join(str(i) for i in cits)
q2 = """
SELECT c.hadm_id, c.itemid,
       AVG(c.valuenum) AS v
FROM `physionet-data.mimiciv_3_1_icu.chartevents` c
WHERE c.hadm_id IN (%s)
  AND c.itemid IN (%s)
  AND c.valuenum IS NOT NULL
GROUP BY c.hadm_id, c.itemid
""" % (hs, cs)
try:
    r2 = cl.query(q2).to_dataframe()
    print("chart rows:", len(r2))
    cmap = {}
    for k, v in CHART.items():
        for i in v:
            cmap[i] = k
    r2["name"] = r2["itemid"].map(cmap)
    ch = r2.groupby(["hadm_id", "name"])[
        "v"].mean().unstack().reset_index()
except Exception as e:
    print("chartevents failed:", e)
    ch = pd.DataFrame({"hadm_id": []})

out = lab[["subject_id", "hadm_id"]].copy()
out = out.merge(labw, on="hadm_id", how="left")
if len(ch):
    out = out.merge(ch, on="hadm_id", how="left")
out.columns = ["subject_id", "hadm_id"] + [
    "mi_" + c for c in out.columns[2:]]
out.to_csv(OUT, index=False)
print("saved", OUT, out.shape)

print("")
print("MIMIC COVERAGE (non-null fraction)")
for c in out.columns[2:]:
    print("  %-16s %.3f"
          % (c, out[c].notna().mean()))

