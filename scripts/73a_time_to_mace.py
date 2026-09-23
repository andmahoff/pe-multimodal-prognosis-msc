import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
OUT = os.path.join(FD, "time_to_mace.csv")

coh = pd.read_csv(
    os.path.join(
        P2, "mimic_pe_mace_cohort.csv"
    )
).drop_duplicates(["subject_id", "hadm_id"])
print("index admissions:", len(coh))

hadms = ",".join(
    str(int(x))
    for x in sorted(coh["hadm_id"].unique())
)

SQL = """
WITH idx AS (
  SELECT subject_id, hadm_id,
         admittime, deathtime
  FROM
`physionet-data.mimiciv_3_1_hosp.admissions`
  WHERE hadm_id IN ({h})
),
mace_ev AS (
  SELECT DISTINCT a.subject_id, a.hadm_id,
         a.admittime
  FROM
`physionet-data.mimiciv_3_1_hosp.admissions` a
  JOIN
`physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
    ON a.hadm_id = d.hadm_id
  WHERE (d.icd_version = 10
         AND SUBSTR(d.icd_code, 1, 3) IN
             ('I21','I22','I50','I46',
              'I47','I48','I49'))
     OR (d.icd_version = 9
         AND SUBSTR(d.icd_code, 1, 3) IN
             ('410','428','427'))
),
readm AS (
  SELECT i.hadm_id AS idx_hadm,
         MIN(DATETIME_DIFF(m.admittime,
             i.admittime, DAY)) AS rd
  FROM idx i
  JOIN mace_ev m
    ON m.subject_id = i.subject_id
   AND m.hadm_id != i.hadm_id
   AND m.admittime >= i.admittime
   AND DATETIME_DIFF(m.admittime,
       i.admittime, DAY) <= 30
  GROUP BY 1
)
SELECT
  i.subject_id,
  i.hadm_id,
  CASE WHEN i.deathtime IS NOT NULL
        AND DATETIME_DIFF(i.deathtime,
            i.admittime, DAY) <= 30
       THEN DATETIME_DIFF(i.deathtime,
            i.admittime, DAY)
  END AS death_days,
  r.rd AS readmit_days
FROM idx i
LEFT JOIN readm r
  ON r.idx_hadm = i.hadm_id
""".format(h=hadms)

client = bigquery.Client(
    project=os.environ.get("GCP_PROJECT_ID")
)
print("querying...")
t = client.query(SQL).to_dataframe()
print("rows:", len(t))

t["death_days"] = pd.to_numeric(
    t["death_days"], errors="coerce"
)
t["readmit_days"] = pd.to_numeric(
    t["readmit_days"], errors="coerce"
)
t["days_to_mace"] = t[
    ["death_days", "readmit_days"]
].min(axis=1)
t["event"] = t["days_to_mace"].notna().astype(int)
t["via_death"] = (
    t["death_days"].notna()
    & (t["death_days"] <= t["readmit_days"]
       .fillna(1e9))
).astype(int)

lab = coh[[
    "subject_id", "hadm_id", "mace_30d_label"
]]
t = t.merge(
    lab, on=["subject_id", "hadm_id"],
    how="left"
)

print("\nagreement with mace_30d_label:")
print(pd.crosstab(
    t["mace_30d_label"], t["event"]
).to_string())

print("\nevents by route:")
print("  via death:   %d"
      % int(t.loc[t["event"] == 1,
                  "via_death"].sum()))
print("  via readmit: %d"
      % int((t["event"] == 1).sum()
            - t.loc[t["event"] == 1,
                    "via_death"].sum()))

print("\ndays_to_mace describe:")
print(t.loc[t["event"] == 1,
            "days_to_mace"].describe())

print("\nbinned:")
b = pd.cut(
    t.loc[t["event"] == 1, "days_to_mace"],
    [-0.01, 7, 14, 30],
    labels=["0-7", "8-14", "15-30"]
)
print(b.value_counts().sort_index().to_string())

t.to_csv(OUT, index=False)
print("\nSaved", OUT)

