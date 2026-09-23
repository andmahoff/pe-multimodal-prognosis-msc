import os
import numpy as np
import pandas as pd
import redivis

FD = (
    "./"
    "fusion_workspace/data/"
)
OUT = os.path.join(
    FD, "inspect_pe_index_labels.csv"
)
pd.set_option("display.width", 200)

ds = redivis.user("shahlab").dataset(
    "inspect_ehr:dzc6:v1_2"
)

PE = 440417
MACE = [4329847, 316139, 321042,
        313217, 443454]
mace_str = ",".join(str(i) for i in MACE)

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

print("=" * 70)
print("1. INDEX PE + DEATH")
print("=" * 70)
q = ds.query(PE_CTE + """
, d AS (
  SELECT person_id,
         MIN(death_DATE) AS death_date
  FROM death GROUP BY person_id
)
SELECT pe.person_id, pe.pe_date, pe.ivisit,
       DATE_DIFF(d.death_date, pe.pe_date,
                 DAY) AS days_to_death
FROM pe
LEFT JOIN d ON d.person_id = pe.person_id
""")
idx = q.to_pandas_dataframe(
    dtype_backend="numpy"
)
n = len(idx)
idx["days_to_death"] = pd.to_numeric(
    idx["days_to_death"], errors="coerce"
)
print("PE-positive persons:", n)
print("with death record:",
      int(idx["days_to_death"].notna().sum()))
for w in [30, 90, 365]:
    k = int((
        (idx["days_to_death"] >= 0)
        & (idx["days_to_death"] <= w)
    ).sum())
    print("  death within %3dd: %5d (%.4f)"
          % (w, k, k / n))

print("\n" + "=" * 70)
print("2. MACE EVENTS 0-30d WITH VISIT IDS")
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
)
SELECT pe.person_id, pe.ivisit, mc.mvisit,
       DATE_DIFF(mc.dt, pe.pe_date, DAY) AS d
FROM pe
JOIN mc ON mc.person_id = pe.person_id
WHERE DATE_DIFF(mc.dt, pe.pe_date, DAY)
      BETWEEN 0 AND 30
""" % mace_str)
ev = q.to_pandas_dataframe(
    dtype_backend="numpy"
)
ev["d"] = pd.to_numeric(
    ev["d"], errors="coerce"
)
print("MACE event rows 0-30d:", len(ev))
print("persons involved:",
      ev["person_id"].nunique())

ev["same_visit"] = (
    ev["mvisit"] == ev["ivisit"]
).astype(int)
print("\nrows on the INDEX visit: %d (%.3f)"
      % (ev["same_visit"].sum(),
         ev["same_visit"].mean()))
print("\nday distribution, same-visit rows:")
print(ev.loc[ev["same_visit"] == 1,
             "d"].describe())
print("\nday distribution, other-visit rows:")
print(ev.loc[ev["same_visit"] == 0,
             "d"].describe())

print("\n" + "=" * 70)
print("3. CV LABEL UNDER DIFFERENT RULES")
print("=" * 70)
rules = {}
oth = ev[ev["same_visit"] == 0]
rules["excl index visit, 0-30d"] = (
    oth.groupby("person_id")["d"].min()
)
for b in [0, 3, 7, 14]:
    s = ev[ev["d"] > b]
    rules["blank >%dd (any visit)" % b] = (
        s.groupby("person_id")["d"].min()
    )
s = oth[oth["d"] > 7]
rules["excl index visit AND >7d"] = (
    s.groupby("person_id")["d"].min()
)
for k, v in rules.items():
    print("  %-28s %5d (%.4f)"
          % (k, len(v), len(v) / n))

print("\n" + "=" * 70)
print("4. BUILD THE THREE LABELS")
print("=" * 70)
PRIMARY = "excl index visit, 0-30d"
cv = rules[PRIMARY].rename(
    "days_to_cv"
).reset_index()
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
    (m["death_30d"] == 1) | (m["cv_30d"] == 1)
).astype(int)

print("primary CV rule:", PRIMARY)
print("cohort n = %d\n" % n)
print("  LABEL 1 cv_first (CV only)  "
      "%5d (%.4f)"
      % (m["cv_first"].sum(),
         m["cv_first"].mean()))
print("  LABEL 2 composite_30d       "
      "%5d (%.4f)"
      % (m["composite_30d"].sum(),
         m["composite_30d"].mean()))
print("  LABEL 3 death_30d           "
      "%5d (%.4f)"
      % (m["death_30d"].sum(),
         m["death_30d"].mean()))
print("\n  death_first %d"
      % m["death_first"].sum())
print("  CV-only training cohort "
      "(excl death_first): %d"
      % int((m["death_first"] == 0).sum()))
print("\ncrosstab death_30d x cv_30d:")
print(pd.crosstab(
    m["death_30d"], m["cv_30d"]
).to_string())

print("\n" + "=" * 70)
print("5. OVERLAP WITH FEATURE MATRIX")
print("=" * 70)
fm = pd.read_csv(
    "./"
    "inspect_workspace/data/processed/"
    "final_feature_matrix_v2_labeled.csv",
    usecols=["person_id", "mace_30d"]
)
print("feature matrix persons:", len(fm))
j = m.merge(fm, on="person_id", how="inner")
print("overlap (usable training set):", len(j))
print("  old mace_30d rate:  %.4f"
      % j["mace_30d"].mean())
print("  new composite rate: %.4f"
      % j["composite_30d"].mean())
print("  new death rate:     %.4f"
      % j["death_30d"].mean())
print("  new cv_first rate:  %.4f"
      % j["cv_first"].mean())

m.to_csv(OUT, index=False)
print("\nSaved", OUT, m.shape)
