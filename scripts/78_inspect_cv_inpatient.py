import os
import numpy as np
import pandas as pd
import redivis

FD = (
    "./"
    "fusion_workspace/data/"
)
OUT = os.path.join(
    FD, "inspect_labels_final.csv"
)
pd.set_option("display.width", 200)

ds = redivis.user("shahlab").dataset(
    "inspect_ehr:dzc6:v1_2"
)
PE = 440417
MACE = [4329847, 316139, 321042,
        313217, 443454]
mace_str = ",".join(str(i) for i in MACE)

print("=" * 70)
print("1. VISIT TYPE DISTRIBUTION")
print("=" * 70)
q = ds.query("""
SELECT vo.visit_concept_id,
       c.concept_name,
       COUNT(*) AS n_visits,
       COUNT(DISTINCT vo.person_id) AS n_pers
FROM visit_occurrence vo
LEFT JOIN concept c
  ON c.concept_id = vo.visit_concept_id
GROUP BY 1, 2
ORDER BY n_visits DESC
LIMIT 20
""")
print(q.to_pandas_dataframe(
    dtype_backend="numpy").to_string(
    index=False))

PE_CTE = """
WITH pe_all AS (
  SELECT co.person_id,
         co.visit_occurrence_id AS ivisit,
         co.condition_start_DATE AS dt,
         ROW_NUMBER() OVER (
           PARTITION BY co.person_id
           ORDER BY co.condition_start_DATE
         ) AS rn
  FROM condition_occurrence co
  JOIN concept_ancestor ca
    ON ca.descendant_concept_id
       = co.condition_concept_id
  WHERE ca.ancestor_concept_id = %d
),
pe AS (
  SELECT person_id, ivisit, dt AS pe_date
  FROM pe_all WHERE rn = 1
)
""" % PE

print("\n" + "=" * 70)
print("2. INDEX VISIT TYPE")
print("=" * 70)
q = ds.query(PE_CTE + """
SELECT vo.visit_concept_id,
       c.concept_name,
       COUNT(*) AS n
FROM pe
LEFT JOIN visit_occurrence vo
  ON vo.visit_occurrence_id = pe.ivisit
LEFT JOIN concept c
  ON c.concept_id = vo.visit_concept_id
GROUP BY 1, 2
ORDER BY n DESC
""")
print(q.to_pandas_dataframe(
    dtype_backend="numpy").to_string(
    index=False))

print("\n" + "=" * 70)
print("3. MACE EVENTS WITH VISIT TYPE")
print("=" * 70)
q = ds.query(PE_CTE + """
, mc AS (
  SELECT co.person_id,
         co.visit_occurrence_id AS mvisit,
         co.condition_start_DATE AS dt
  FROM condition_occurrence co
  JOIN concept_ancestor ca
    ON ca.descendant_concept_id
       = co.condition_concept_id
  WHERE ca.ancestor_concept_id IN (%s)
),
prior AS (
  SELECT DISTINCT co.person_id
  FROM condition_occurrence co
  JOIN concept_ancestor ca
    ON ca.descendant_concept_id
       = co.condition_concept_id
  JOIN pe ON pe.person_id = co.person_id
  WHERE ca.ancestor_concept_id IN (%s)
    AND co.condition_start_DATE < pe.pe_date
)
SELECT pe.person_id, pe.ivisit, mc.mvisit,
       vo.visit_concept_id AS vtype,
       DATE_DIFF(mc.dt, pe.pe_date, DAY) AS d,
       CASE WHEN p.person_id IS NULL
            THEN 0 ELSE 1 END AS had_prior
FROM pe
JOIN mc ON mc.person_id = pe.person_id
LEFT JOIN visit_occurrence vo
  ON vo.visit_occurrence_id = mc.mvisit
LEFT JOIN prior p
  ON p.person_id = pe.person_id
WHERE DATE_DIFF(mc.dt, pe.pe_date, DAY)
      BETWEEN 0 AND 30
""" % (mace_str, mace_str))
ev = q.to_pandas_dataframe(
    dtype_backend="numpy"
)
ev["d"] = pd.to_numeric(ev["d"],
                        errors="coerce")
print("rows:", len(ev),
      " persons:", ev["person_id"].nunique())
print("\nby visit type:")
print(ev.groupby("vtype").agg(
    rows=("d", "size"),
    persons=("person_id", "nunique"),
).to_string())

IP = [9201, 262, 8717]
print("\ninpatient concept ids used:", IP)

n = 4524
ev["is_ip"] = ev["vtype"].isin(IP).astype(int)
ev["not_index"] = (
    ev["mvisit"] != ev["ivisit"]
).astype(int)

print("\n" + "=" * 70)
print("4. CV RATE UNDER RULES (n=%d)" % n)
print("=" * 70)
rules = {}
rules["any visit, excl index"] = ev[
    ev["not_index"] == 1]
rules["INPATIENT, excl index"] = ev[
    (ev["is_ip"] == 1)
    & (ev["not_index"] == 1)]
rules["INPATIENT, excl index, >0d"] = ev[
    (ev["is_ip"] == 1)
    & (ev["not_index"] == 1)
    & (ev["d"] > 0)]
rules["INPATIENT, excl index, >2d"] = ev[
    (ev["is_ip"] == 1)
    & (ev["not_index"] == 1)
    & (ev["d"] > 2)]
rules["INPATIENT, excl idx, no prior"] = ev[
    (ev["is_ip"] == 1)
    & (ev["not_index"] == 1)
    & (ev["had_prior"] == 0)]

for k, v in rules.items():
    p = v["person_id"].nunique()
    print("  %-32s %5d (%.4f)"
          % (k, p, p / n))
print("\n  MIMIC reference CV rate: 0.0541")

print("\n" + "=" * 70)
print("5. FINAL LABELS")
print("=" * 70)
PRIM = "INPATIENT, excl index, >0d"
cv = rules[PRIM].groupby(
    "person_id")["d"].min()
cv = cv.rename("days_to_cv").reset_index()

idx = pd.read_csv(
    os.path.join(
        FD, "inspect_pe_index_labels.csv"
    ),
    usecols=["person_id", "pe_date",
             "ivisit", "days_to_death"]
)
m = idx.merge(cv, on="person_id", how="left")
dd = m["days_to_death"]
dm = m["days_to_cv"]
m["death_30d"] = (
    dd.notna() & (dd >= 0) & (dd <= 30)
).astype(int)
m["cv_30d"] = dm.notna().astype(int)
m["death_first"] = (
    (m["death_30d"] == 1)
    & (dm.isna() | (dd < dm))
).astype(int)
m["cv_first"] = (
    (m["cv_30d"] == 1)
    & (m["death_first"] == 0)
).astype(int)
m["composite_30d"] = (
    (m["death_30d"] == 1)
    | (m["cv_30d"] == 1)
).astype(int)

print("primary rule:", PRIM)
print("n = %d\n" % len(m))
for c, lab in [
    ("cv_first", "LABEL 1 CV only"),
    ("composite_30d", "LABEL 2 composite"),
    ("death_30d", "LABEL 3 death"),
]:
    print("  %-20s %5d (%.4f)"
          % (lab, m[c].sum(), m[c].mean()))
print("\n  MIMIC: CV 0.0541  "
      "composite 0.1230  death 0.0689")
print("\n  CV-only training cohort: %d"
      % int((m["death_first"] == 0).sum()))

m.to_csv(OUT, index=False)
print("\nSaved", OUT, m.shape)
