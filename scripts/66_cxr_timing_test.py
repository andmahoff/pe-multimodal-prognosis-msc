import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
OUT = os.path.join(FD, "cxr_timing_test.csv")
KEY = ["subject_id", "hadm_id"]

PROJECT = os.environ.get("GCP_PROJECT_ID")


def auc(y, p):
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().values
    return (
        r[y == 1].sum() - n1 * (n1 + 1) / 2
    ) / (n1 * n0)


print("Loading predictions...")
c = pd.read_csv(
    os.path.join(FD, "cxr_oof_predictions.csv")
)
print("  images:", len(c))

print("Loading metadata (StudyDate)...")
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
print("  merged:", m.shape)
print("  StudyDate nulls:",
      m["StudyDate"].isna().sum())

m["sd"] = pd.to_datetime(
    m["StudyDate"].astype("Int64").astype(str),
    format="%Y%m%d", errors="coerce"
)
print("  parsed nulls:", m["sd"].isna().sum())
print("  date range:",
      m["sd"].min(), "->", m["sd"].max())

hadms = ",".join(
    str(int(x))
    for x in sorted(m["hadm_id"].unique())
)
client = bigquery.Client(project=PROJECT)
print("\nQuerying admission times...")
q = """
SELECT hadm_id, admittime, dischtime
FROM
`physionet-data.mimiciv_3_1_hosp.admissions`
WHERE hadm_id IN ({h})
""".format(h=hadms)
adm = client.query(q).to_dataframe()
print("  admissions:", len(adm))

adm["a_d"] = pd.to_datetime(
    adm["admittime"]
).dt.tz_localize(None).dt.normalize()
adm["d_d"] = pd.to_datetime(
    adm["dischtime"]
).dt.tz_localize(None).dt.normalize()
adm = adm[["hadm_id", "a_d", "d_d"]]

m = m.merge(adm, on="hadm_id", how="left")

m["rel"] = np.where(
    m["sd"] < m["a_d"], "before",
    np.where(m["sd"] > m["d_d"],
             "after", "during")
)

print("\n" + "=" * 55)
print("IMAGE TIMING vs INDEX ADMISSION")
print("=" * 55)
print(m["rel"].value_counts())
print("\nfraction:")
print(m["rel"].value_counts(normalize=True))

print("\nlength of stay (days):")
los = (m["d_d"] - m["a_d"]).dt.days
print(los.describe())

print("\ndays from discharge (after only):")
da = (
    m.loc[m["rel"] == "after", "sd"]
    - m.loc[m["rel"] == "after", "d_d"]
).dt.days
print(da.describe())
print("within 30d of discharge: %d"
      % (da <= 30).sum())

a = m.groupby(KEY).agg(
    y=("mace_30d_label", "max"),
    p_all=("p_cxr", "mean"),
    n_all=("p_cxr", "size"),
).reset_index()

dur = m[m["rel"] == "during"].groupby(
    KEY
).agg(
    p_dur=("p_cxr", "mean"),
    n_dur=("p_cxr", "size"),
).reset_index()

aft = m[m["rel"] == "after"].groupby(
    KEY
).agg(n_aft=("p_cxr", "size")).reset_index()

bef = m[m["rel"] == "before"].groupby(
    KEY
).agg(n_bef=("p_cxr", "size")).reset_index()

a = a.merge(dur, on=KEY, how="left")
a = a.merge(aft, on=KEY, how="left")
a = a.merge(bef, on=KEY, how="left")
for col in ["n_dur", "n_aft", "n_bef"]:
    a[col] = a[col].fillna(0).astype(int)

print("\n" + "=" * 55)
print("ADMISSION-LEVEL COVERAGE")
print("=" * 55)
print("admissions:", len(a))
print("with any during-image: %d"
      % (a["n_dur"] > 0).sum())
print("with any after-image:  %d"
      % (a["n_aft"] > 0).sum())
print("with any before-image: %d"
      % (a["n_bef"] > 0).sum())
print("\nn_all describe:")
print(a["n_all"].describe())
print("\nn_dur describe:")
print(a["n_dur"].describe())

print("\n" + "=" * 55)
print("MACE RATE BY POST-DISCHARGE IMAGING")
print("=" * 55)
a["has_aft"] = (a["n_aft"] > 0).astype(int)
print(
    a.groupby("has_aft")["y"]
    .agg(["count", "sum", "mean"]).to_string()
)
print("\nbase rate: %.4f" % a["y"].mean())

print("\n" + "=" * 55)
print("p_cxr AUC: ALL images vs DURING only")
print("=" * 55)
print("all images     n=%d  AUC %.4f"
      % (len(a), auc(a["y"], a["p_all"])))
sub = a[a["n_dur"] > 0]
print("during only    n=%d  AUC %.4f"
      % (len(sub), auc(sub["y"], sub["p_dur"])))
print("  (same rows, all-image score) AUC %.4f"
      % auc(sub["y"], sub["p_all"]))

print("\nn_images AUC:")
print("  n_all  %.4f" % auc(a["y"], a["n_all"]))
print("  n_dur  %.4f" % auc(a["y"], a["n_dur"]))
print("  n_aft  %.4f" % auc(a["y"], a["n_aft"]))

a.to_csv(OUT, index=False)
print("\nSaved", OUT)

