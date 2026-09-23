import os
import pandas as pd
import numpy as np
from google.cloud import bigquery

PROJ = os.environ.get("GCP_PROJECT_ID")
BASE = "."
DATA = BASE + "/fusion_workspace/data"
LAB = DATA + "/mimic_labels_harmonised.csv"
ADM = DATA + "/index_admission_times.csv"

cl = bigquery.Client(project=PROJ)
lab = pd.read_csv(LAB)
idx = pd.read_csv(ADM)
subs = sorted(lab["subject_id"].unique())
ss = ",".join(str(x) for x in subs)

q1 = """
SELECT subject_id, hadm_id, admittime
FROM `physionet-data.mimiciv_3_1_hosp.admissions`
WHERE subject_id IN (%s)
""" % ss
a = cl.query(q1).to_dataframe()
print("all admissions for cohort subs:", len(a))

a["admittime"] = pd.to_datetime(a["admittime"])
idx["admittime"] = pd.to_datetime(
    idx["admittime"])
ix = idx[["subject_id", "hadm_id",
          "admittime"]].rename(
    columns={"hadm_id": "idx_hadm",
             "admittime": "idx_time"})
m = a.merge(ix, on="subject_id")
m = m[m["admittime"] < m["idx_time"]]

pri = m.groupby("idx_hadm").size()
cov = ix[["idx_hadm"]].copy()
cov["n_prior"] = cov["idx_hadm"].map(
    pri).fillna(0)
print("")
print("PRIOR ADMISSIONS before index")
print(cov["n_prior"].describe().round(2)
      .to_string())
print("with zero prior:",
      int((cov["n_prior"] == 0).sum()),
      "of", len(cov),
      "(%.3f)" % (cov["n_prior"] == 0).mean())

hs = m["hadm_id"].unique().tolist()
print("")
print("prior hadm_ids:", len(hs))
if len(hs) == 0:
    raise SystemExit("no prior admissions")

hss = ",".join(str(x) for x in hs)
q2 = """
SELECT hadm_id, COUNT(*) AS n
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
WHERE hadm_id IN (%s)
GROUP BY hadm_id
""" % hss
d = cl.query(q2).to_dataframe()
m2 = m.merge(d, on="hadm_id", how="left")
m2["n"] = m2["n"].fillna(0)
tot = m2.groupby("idx_hadm")["n"].sum()
cov["n_codes"] = cov["idx_hadm"].map(
    tot).fillna(0)

print("")
print("PRIOR DIAGNOSIS CODES per index adm")
print(cov["n_codes"].describe().round(1)
      .to_string())
for t in [0, 5, 20, 50, 100]:
    print("  >%3d codes: %5d (%.3f)"
          % (t, int((cov["n_codes"] > t).sum()),
             float((cov["n_codes"] > t).mean())))

print("")
print("INSPECT reference: median 377 codes")
cov.to_csv(DATA + "/mimic_history_depth.csv",
           index=False)

