import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import Counter
from google.cloud import bigquery

PROJ = os.environ.get("GCP_PROJECT_ID")
BASE = "."
INS = BASE + "/inspect_files"
DATA = BASE + "/fusion_workspace/data"
SHARDS = INS + "/meds_omop_inspect/data/*.parquet"
OUT = DATA + "/loinc_overlap.csv"

PRE_D = 7

# ---------- MIMIC side ----------
cl = bigquery.Client(project=PROJ)
lab = pd.read_csv(DATA + "/mimic_labels_harmonised.csv")
hs = ",".join(str(h) for h in
              sorted(lab["hadm_id"].unique()))

q = """
SELECT d.loinc_code AS loinc,
       ANY_VALUE(d.label) AS label,
       ANY_VALUE(d.fluid) AS fluid,
       COUNT(DISTINCT l.hadm_id) AS n_adm,
       COUNT(*) AS n_rows
FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` d
  ON l.itemid = d.itemid
WHERE l.hadm_id IN (%s)
  AND l.valuenum IS NOT NULL
  AND d.loinc_code IS NOT NULL
GROUP BY d.loinc_code
""" % hs

mm = cl.query(q).to_dataframe()
n_adm = lab["hadm_id"].nunique()
mm["mimic_cov"] = mm["n_adm"] / n_adm
print("MIMIC index admissions:", n_adm)
print("distinct LOINC in MIMIC labs:", len(mm))
print("with cov >0.5:",
      int((mm["mimic_cov"] > 0.5).sum()),
      " >0.8:",
      int((mm["mimic_cov"] > 0.8).sum()))

# ---------- INSPECT side ----------
coh = pd.read_csv(
    DATA + "/inspect_count_cohort.csv")
mp = pd.read_csv(
    INS + "/study_mapping_20250611.tsv",
    sep="\t")
mp["pdt"] = pd.to_datetime(
    mp["procedure_DATETIME"], errors="coerce")
first = mp.sort_values(
    ["person_id", "pdt"]).groupby(
    "person_id", as_index=False).first()
coh = coh.merge(first[["person_id", "pdt"]],
                on="person_id", how="left")
idx = dict(zip(coh["person_id"], coh["pdt"]))
subs = set(coh["person_id"])
print("")
print("INSPECT cohort:", len(coh))

files = sorted(glob.glob(SHARDS))
win = Counter()
allt = Counter()
for i, f in enumerate(files):
    t = pq.read_table(f, columns=[
        "subject_id", "time", "code",
        "numeric_value"])
    d = t.to_pandas()
    d = d[d["subject_id"].isin(subs)]
    if len(d) == 0:
        continue
    d = d[d["numeric_value"].notna()]
    d["code"] = d["code"].astype(str)
    d = d[d["code"].str.startswith("LOINC/")]
    if len(d) == 0:
        continue
    d["idx"] = d["subject_id"].map(idx)
    d["time"] = pd.to_datetime(d["time"],
                               errors="coerce")
    d = d[d["time"] < d["idx"]]
    if len(d) == 0:
        continue
    d["loinc"] = d["code"].str[6:]
    allt.update(set(zip(d["subject_id"],
                        d["loinc"])))
    lo = d["idx"] - pd.Timedelta(days=PRE_D)
    w = d[d["time"] >= lo]
    win.update(set(zip(w["subject_id"],
                       w["loinc"])))
    if (i + 1) % 20 == 0:
        print("  shard", i + 1, "of",
              len(files))

ca = Counter(c for _, c in allt)
cw = Counter(c for _, c in win)
ni = len(coh)
ins = pd.DataFrame({
    "loinc": list(ca.keys()),
    "ins_all": [ca[k] / ni for k in ca],
    "ins_7d": [cw.get(k, 0) / ni for k in ca]})
print("")
print("distinct LOINC in INSPECT:", len(ins))
print("ins_all >0.5:",
      int((ins["ins_all"] > 0.5).sum()),
      " ins_7d >0.5:",
      int((ins["ins_7d"] > 0.5).sum()))

# ---------- overlap ----------
j = mm[["loinc", "label", "fluid",
        "mimic_cov"]].merge(ins, on="loinc")
j["minc"] = j[["mimic_cov", "ins_7d"]].min(
    axis=1)
j = j.sort_values("minc", ascending=False)
j.to_csv(OUT, index=False)

print("")
print("=" * 60)
print("SHARED LOINC CODES:", len(j))
for t in [0.3, 0.5, 0.7, 0.8, 0.9]:
    n = int(((j["mimic_cov"] >= t)
             & (j["ins_7d"] >= t)).sum())
    n2 = int(((j["mimic_cov"] >= t)
              & (j["ins_all"] >= t)).sum())
    print("  both cov >=%.1f : 7d %3d"
          "   all-time %3d" % (t, n, n2))

print("")
print("TOP 40 BY MIN COVERAGE (7d window)")
print(j.head(40)[["loinc", "label", "fluid",
                  "mimic_cov", "ins_7d",
                  "ins_all"]]
      .round(3).to_string(index=False))
print("")
print("saved", OUT)

