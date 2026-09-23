import redivis

DS = "inspect_ehr:dzc6:v1_2"
ds = redivis.user("shahlab").dataset(DS)


def run(sql):
    q = ds.query(sql)
    return q.to_pandas_dataframe(
        dtype_backend="numpy")


print("=" * 55)
print("NOTE: other string columns")
sql = """
SELECT
  COUNTIF(note_title IS NOT NULL) AS t_nn,
  MAX(LENGTH(note_title)) AS t_max,
  COUNT(DISTINCT note_title) AS t_dis,
  COUNTIF(note_source_value IS NOT NULL) AS s_nn,
  MAX(LENGTH(note_source_value)) AS s_max,
  COUNT(DISTINCT note_source_value) AS s_dis
FROM `note:ys4x`
"""
print(run(sql).to_string())

print("")
print("=" * 55)
print("NOTE: class concept counts")
sql = """
SELECT note_class_concept_id,
       note_title,
       COUNT(*) AS n
FROM `note:ys4x`
GROUP BY 1, 2
ORDER BY n DESC
LIMIT 25
"""
print(run(sql).to_string())

print("")
print("=" * 55)
print("NOTE: longest source values")
sql = """
SELECT note_id, person_id,
       LENGTH(note_source_value) AS L,
       SUBSTR(note_source_value, 1, 400) AS snip
FROM `note:ys4x`
WHERE note_source_value IS NOT NULL
ORDER BY L DESC
LIMIT 5
"""
try:
    d = run(sql)
    for _, r in d.iterrows():
        print("-" * 50)
        print("len", r["L"])
        print(r["snip"])
except Exception as e:
    print("failed:", e)

for nm in ["Linkage", "MEDS"]:
    print("")
    print("=" * 55)
    print("TABLE:", nm)
    try:
        t = ds.table(nm)
        t.get()
        print("  rows:", t.properties.get("numRows"))
        vs = t.list_variables()
        for v in vs:
            print("   ", v.name)
        df = t.to_pandas_dataframe(
            max_results=3,
            dtype_backend="numpy")
        print(df.to_string())
    except Exception as e:
        print("  failed:", type(e).__name__, e)

