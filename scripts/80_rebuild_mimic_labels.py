import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
OUT = os.path.join(
    FD, "mimic_labels_harmonised.csv"
)
KEY = ["subject_id", "hadm_id"]
pd.set_option("display.width", 200)

coh = pd.read_csv(
    os.path.join(
        P2, "mimic_pe_mace_cohort.csv"
    )
).drop_duplicates(KEY)
print("index admissions:", len(coh))
hadms = ",".join(
    str(int(x))
    for x in sorted(coh["hadm_id"].unique())
)

SQL = """
WITH idx AS (
  SELECT subject_id, hadm_id, admittime,
         dischtime, deathtime
  FROM
`physionet-data.mimiciv_3_1_hosp.admissions`
  WHERE hadm_id IN ({h})
),
mace_adm AS (
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
              'I48','I63'))
     OR (d.icd_version = 9
         AND (SUBSTR(d.icd_code, 1, 3) IN
              ('410','428','433','434')
           OR SUBSTR(d.icd_code, 1, 4) IN
              ('4275','4273')))
),
cv AS (
  SELECT i.hadm_id AS ih,
         MIN(DATE_DIFF(DATE(m.admittime),
             DATE(i.admittime), DAY)) AS cvd
  FROM idx i
  JOIN mace_adm m
    ON m.subject_id = i.subject_id
   AND m.hadm_id != i.hadm_id
   AND DATE_DIFF(DATE(m.admittime),
       DATE(i.admittime), DAY)
       BETWEEN 0 AND 30
  GROUP BY 1
)
SELECT
  i.subject_id,
  i.hadm_id,
  DATE_DIFF(DATE(p.dod), DATE(i.admittime),
            DAY) AS dod_days,
  CASE WHEN i.deathtime IS NOT NULL
       THEN DATE_DIFF(DATE(i.deathtime),
            DATE(i.admittime), DAY)
  END AS inhosp_days,
  cv.cvd AS cv_days
FROM idx i
LEFT JOIN
`physionet-data.mimiciv_3_1_hosp.patients` p
  ON p.subject_id = i.subject_id
LEFT JOIN cv ON cv.ih = i.hadm_id
""".format(h=hadms)

client = bigquery.Client(
    project=os.environ.get("GCP_PROJECT_ID")
)
print("querying...")
t = client.query(SQL).to_dataframe()
print("rows:", len(t))

for c in ["dod_days", "inhosp_days",
          "cv_days"]:
    t[c] = pd.to_numeric(
        t[c], errors="coerce"
    ).astype("float64")

n = len(t)
dod30 = (
    (t["dod_days"] >= 0)
    & (t["dod_days"] <= 30)
).fillna(False).to_numpy()
ih30 = (
    (t["inhosp_days"] >= 0)
    & (t["inhosp_days"] <= 30)
).fillna(False).to_numpy()

print("\n" + "=" * 62)
print("DEATH CAPTURE: dod vs deathtime")
print("=" * 62)
print("  in-hospital (old) %5d (%.4f)"
      % (ih30.sum(), ih30.sum() / n))
print("  dod (new)         %5d (%.4f)"
      % (dod30.sum(), dod30.sum() / n))
print("  dod but not in-hosp: %d"
      % int((dod30 & ~ih30).sum()))
print("  in-hosp but not dod: %d"
      % int((ih30 & ~dod30).sum()))
print("  subjects with any dod: %d"
      % int(t["dod_days"].notna().sum()))

dodv = t["dod_days"].to_numpy(dtype=float)
ihv = t["inhosp_days"].to_numpy(dtype=float)
cvv = t["cv_days"].to_numpy(dtype=float)

both = np.fmin(
    np.where(dod30, dodv, np.inf),
    np.where(ih30, ihv, np.inf)
)
t["death_days"] = np.where(
    np.isinf(both), np.nan, both
)
t["death_30d"] = (
    ~np.isnan(t["death_days"].to_numpy())
).astype(int)
t["death_30d_inhosp"] = ih30.astype(int)
t["cv_30d"] = (~np.isnan(cvv)).astype(int)

dd = t["death_days"].to_numpy(dtype=float)
dfirst = (
    (t["death_30d"].to_numpy() == 1)
    & (np.isnan(cvv) | (dd < cvv))
)
t["death_first"] = dfirst.astype(int)
t["cv_first"] = (
    (t["cv_30d"].to_numpy() == 1) & ~dfirst
).astype(int)
t["composite_30d"] = (
    (t["death_30d"].to_numpy() == 1)
    | (t["cv_30d"].to_numpy() == 1)
).astype(int)
t["composite_inhosp"] = (
    (ih30) | (t["cv_30d"].to_numpy() == 1)
).astype(int)

print("\n" + "=" * 62)
print("HARMONISED LABELS (n=%d)" % n)
print("=" * 62)
for c, lab in [
    ("cv_first", "LABEL 1 CV only"),
    ("composite_30d", "LABEL 2 composite"),
    ("death_30d", "LABEL 3 death (dod)"),
    ("death_30d_inhosp", "  death (in-hosp)"),
    ("composite_inhosp", "  composite (in-h)"),
]:
    print("  %-22s %5d (%.4f)"
          % (lab, t[c].sum(), t[c].mean()))
print("\n  death_first %d  cv_first %d"
      % (t["death_first"].sum(),
         t["cv_first"].sum()))
print("  CV-only cohort: %d"
      % int((t["death_first"] == 0).sum()))
print("\n  INSPECT ref: CV 0.0809  "
      "composite 0.1346  death 0.0610")

print("\n" + "=" * 62)
print("VS ORIGINAL LABELS")
print("=" * 62)
old = pd.read_csv(
    os.path.join(FD, "time_to_mace.csv")
)
keep = KEY + ["event", "via_death"]
old = old[keep].copy()
old["old_comp"] = old["event"].astype(int)
old["old_death"] = (
    old["via_death"].fillna(0).astype(int)
)
old["old_cv"] = (
    (old["old_comp"] == 1)
    & (old["old_death"] == 0)
).astype(int)
old = old[KEY + ["old_comp", "old_death",
                 "old_cv"]]

j = t.merge(old, on=KEY, how="inner")
print("  old composite %5d (%.4f)"
      % (j["old_comp"].sum(),
         j["old_comp"].mean()))
print("  new composite %5d (%.4f)"
      % (j["composite_30d"].sum(),
         j["composite_30d"].mean()))
for a, b, nm in [
    ("old_comp", "composite_30d", "composite"),
    ("old_cv", "cv_first", "CV"),
    ("old_death", "death_first", "death"),
]:
    print("\n  crosstab %s (rows=old):" % nm)
    print(pd.crosstab(
        j[a], j[b]
    ).to_string())

t.to_csv(OUT, index=False)
print("\nSaved", OUT, t.shape)
