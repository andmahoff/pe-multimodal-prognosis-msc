import os
import pandas as pd
from google.cloud import bigquery

PROJ = os.environ.get("GCP_PROJECT_ID")
BASE = "."
DATA = BASE + "/fusion_workspace/data"
LAB = DATA + "/mimic_labels_harmonised.csv"
OUT = DATA + "/index_admission_times.csv"

client = bigquery.Client(project=PROJ)
lab = pd.read_csv(LAB)
print("cohort admissions:", len(lab))

hadms = sorted(lab["hadm_id"].unique().tolist())
hs = ",".join(str(h) for h in hadms)

q = """
SELECT hadm_id, subject_id, admittime, dischtime
FROM `physionet-data.mimiciv_3_1_hosp.admissions`
WHERE hadm_id IN (%s)
""" % hs

adm = client.query(q).to_dataframe()
print("admissions pulled:", len(adm))
print(adm.head())

adm.to_csv(OUT, index=False)
print("saved", OUT)

