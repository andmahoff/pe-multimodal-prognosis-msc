import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
OUT = os.path.join(FD, "cxr_image_timing.csv")

c = pd.read_csv(
    os.path.join(FD, "cxr_oof_predictions.csv")
)
meta = pd.concat([
    pd.read_csv(P2 + "train.csv"),
    pd.read_csv(P2 + "val.csv"),
    pd.read_csv(P2 + "test.csv"),
], ignore_index=True)
meta = meta[[
    "dicom_id", "StudyDate",
    "PerformedProcedureStepDescription",
]].drop_duplicates("dicom_id")
m = c.merge(meta, on="dicom_id", how="left")

m["sd"] = pd.to_datetime(
    m["StudyDate"].astype("Int64").astype(str),
    format="%Y%m%d", errors="coerce"
)
m["port"] = (
    m["PerformedProcedureStepDescription"]
    .fillna("").str.upper()
    .str.contains("PORT").astype(int)
)

hadms = ",".join(
    str(int(x))
    for x in sorted(m["hadm_id"].unique())
)
client = bigquery.Client(
    project=os.environ.get("GCP_PROJECT_ID")
)
q = """
SELECT hadm_id, admittime, dischtime
FROM
`physionet-data.mimiciv_3_1_hosp.admissions`
WHERE hadm_id IN ({h})
""".format(h=hadms)
adm = client.query(q).to_dataframe()
adm["a_d"] = pd.to_datetime(
    adm["admittime"]
).dt.tz_localize(None).dt.normalize()
adm["d_d"] = pd.to_datetime(
    adm["dischtime"]
).dt.tz_localize(None).dt.normalize()

m = m.merge(
    adm[["hadm_id", "a_d", "d_d"]],
    on="hadm_id", how="left"
)
m["rel"] = np.where(
    m["sd"] < m["a_d"], "before",
    np.where(m["sd"] > m["d_d"],
             "after", "during")
)

out = m[[
    "dicom_id", "subject_id", "hadm_id",
    "mace_30d_label", "p_cxr", "rel", "port",
]]
out.to_csv(OUT, index=False)
print("saved", OUT, out.shape)
print(out["rel"].value_counts())

