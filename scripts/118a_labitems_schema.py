import os
import pandas as pd
from google.cloud import bigquery

PROJ = os.environ.get("GCP_PROJECT_ID")
cl = bigquery.Client(project=PROJ)

DSETS = ["mimiciv_3_1_hosp", "mimiciv_hosp",
         "mimiciv_2_2_hosp", "mimiciv_2_0_hosp"]

for ds in DSETS:
    q = """
    SELECT table_name, column_name, data_type
    FROM `physionet-data.%s.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'd_labitems'
    ORDER BY ordinal_position
    """ % ds
    print("")
    print("=" * 50)
    print(ds)
    try:
        d = cl.query(q).to_dataframe()
        if len(d) == 0:
            print("  no d_labitems")
            continue
        print(d[["column_name",
                 "data_type"]].to_string(
            index=False))
    except Exception as e:
        print("  failed:", type(e).__name__)
        print(" ", str(e)[:160])

print("")
print("=" * 50)
print("d_labitems sample (3_1)")
q2 = """
SELECT *
FROM `physionet-data.mimiciv_3_1_hosp.d_labitems`
LIMIT 5
"""
try:
    print(cl.query(q2).to_dataframe().to_string())
except Exception as e:
    print("failed:", str(e)[:200])

print("")
print("=" * 50)
print("most frequent lab items in cohort")
BASE = "."
DATA = BASE + "/fusion_workspace/data"
lab = pd.read_csv(
    DATA + "/mimic_labels_harmonised.csv")
hs = ",".join(str(h) for h in
              sorted(lab["hadm_id"].unique()))
n = lab["hadm_id"].nunique()
q3 = """
SELECT l.itemid,
       ANY_VALUE(d.label) AS label,
       ANY_VALUE(d.fluid) AS fluid,
       ANY_VALUE(d.category) AS category,
       COUNT(DISTINCT l.hadm_id) AS n_adm
FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` d
  ON l.itemid = d.itemid
WHERE l.hadm_id IN (%s)
  AND l.valuenum IS NOT NULL
GROUP BY l.itemid
ORDER BY n_adm DESC
LIMIT 60
""" % hs
d3 = cl.query(q3).to_dataframe()
d3["cov"] = d3["n_adm"] / n
print("index admissions:", n)
print(d3[["itemid", "label", "fluid",
          "category", "cov"]].round(3)
      .to_string(index=False))

