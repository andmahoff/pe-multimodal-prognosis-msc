import pandas as pd
import redivis

BASE = "."
DATA = BASE + "/fusion_workspace/data"
COV = DATA + "/inspect_loinc_coverage.csv"
OUT = DATA + "/inspect_loinc_named.csv"

cov = pd.read_csv(COV)
cov = cov[cov["loinc"].astype(str).str.contains(
    r"^\d+-\d+$", regex=True)]
print("standard LOINC codes:", len(cov))

top = cov.sort_values("d7", ascending=False)
top = top.head(200)
codes = top["loinc"].tolist()
lst = ",".join("'%s'" % c for c in codes)

ds = redivis.user("shahlab").dataset(
    "inspect_ehr:dzc6:v1_2")
ds.get()

sql = """
SELECT concept_code, concept_name,
       domain_id, concept_class_id,
       standard_concept
FROM concept
WHERE vocabulary_id = 'LOINC'
  AND concept_code IN (%s)
""" % lst

r = ds.query(sql).to_pandas_dataframe(
    dtype_backend="numpy")
print("named:", len(r))

m = top.merge(r, left_on="loinc",
              right_on="concept_code",
              how="left")
m = m.drop(columns=["desc", "description",
                    "bare"], errors="ignore")
m = m.sort_values("d7", ascending=False)
m.to_csv(OUT, index=False)

print("named fraction: %.3f"
      % m["concept_name"].notna().mean())
print("")
print("TOP 70 BY 7-DAY COVERAGE")
print(m.head(70)[["loinc", "concept_name",
                  "concept_class_id",
                  "d7", "d30", "all"]]
      .to_string(index=False))
print("")
print("saved", OUT)

