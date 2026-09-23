import pandas as pd
import redivis

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

ds = redivis.user("shahlab").dataset(
    "inspect_ehr:dzc6:v1_2"
)

CAND = {
    440417: "pulmonary embolism",
    4329847: "myocardial infarction",
    316139: "heart failure",
    321042: "cardiac arrest",
    313217: "atrial fibrillation",
    372924: "cerebral infarction",
    443392: "malignant neoplasm (cancer)",
    255573: "COPD",
}
ids = ",".join(str(i) for i in CAND)

print("=" * 70)
print("1. VERIFY CANDIDATE CONCEPT IDS")
print("=" * 70)
q = ds.query("""
SELECT concept_id, concept_name, domain_id,
       vocabulary_id, concept_class_id,
       standard_concept
FROM concept
WHERE concept_id IN (%s)
""" % ids)
c = q.to_pandas_dataframe()
print(c.to_string(index=False))
print("\nexpected labels:")
for k, v in CAND.items():
    print("  %-10d %s" % (k, v))

print("\n" + "=" * 70)
print("2. SEARCH FOR STANDARD CONDITION CONCEPTS")
print("=" * 70)
q = ds.query("""
SELECT concept_id, concept_name,
       vocabulary_id, concept_class_id
FROM concept
WHERE domain_id = 'Condition'
  AND standard_concept = 'S'
  AND (
    LOWER(concept_name) = 'pulmonary embolism'
    OR LOWER(concept_name) = 'myocardial infarction'
    OR LOWER(concept_name) = 'cardiac arrest'
    OR LOWER(concept_name) = 'heart failure'
    OR LOWER(concept_name) = 'cerebral infarction'
    OR LOWER(concept_name) = 'atrial fibrillation'
  )
ORDER BY concept_name
""")
s = q.to_pandas_dataframe()
print(s.to_string(index=False))

print("\n" + "=" * 70)
print("3. PE COVERAGE IN condition_occurrence")
print("=" * 70)
q = ds.query("""
SELECT
  COUNT(DISTINCT co.person_id) AS n_persons,
  COUNT(*) AS n_rows,
  MIN(co.condition_start_DATE) AS first_dt,
  MAX(co.condition_start_DATE) AS last_dt
FROM condition_occurrence co
JOIN concept_ancestor ca
  ON ca.descendant_concept_id
     = co.condition_concept_id
WHERE ca.ancestor_concept_id = 440417
""")
print(q.to_pandas_dataframe().to_string(
    index=False))

print("\n" + "=" * 70)
print("4. MACE COMPONENT COVERAGE")
print("=" * 70)
for cid, nm in [
    (4329847, "MI"), (316139, "HF"),
    (321042, "cardiac arrest"),
    (313217, "AF"),
    (372924, "cerebral infarction"),
]:
    q = ds.query("""
    SELECT COUNT(DISTINCT co.person_id) AS n_p,
           COUNT(*) AS n_rows
    FROM condition_occurrence co
    JOIN concept_ancestor ca
      ON ca.descendant_concept_id
         = co.condition_concept_id
    WHERE ca.ancestor_concept_id = %d
    """ % cid)
    r = q.to_pandas_dataframe()
    print("  %-22s persons=%-7s rows=%s"
          % (nm, r["n_p"][0], r["n_rows"][0]))

print("\n" + "=" * 70)
print("5. VISITS PER PERSON (index visit check)")
print("=" * 70)
q = ds.query("""
SELECT
  COUNT(DISTINCT person_id) AS n_persons,
  COUNT(*) AS n_visits,
  COUNT(*) / COUNT(DISTINCT person_id)
    AS visits_per_person
FROM visit_occurrence
""")
print(q.to_pandas_dataframe().to_string(
    index=False))

print("\nDone.")

