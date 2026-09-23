import redivis

DS = "inspect_ehr:dzc6:v1_2"


def get_ds():
    try:
        d = redivis.organization("stanford").dataset(DS)
        d.get()
        return d
    except Exception:
        d = redivis.user("shahlab").dataset(DS)
        d.get()
        return d


ds = get_ds()


def run(sql):
    q = ds.query(sql)
    return q.to_pandas_dataframe(
        dtype_backend="numpy")


sql1 = """
SELECT
  COUNT(*) AS n_rows,
  COUNTIF(note_text IS NOT NULL) AS n_text,
  COUNT(DISTINCT person_id) AS n_person,
  COUNT(DISTINCT CASE WHEN note_text IS NOT NULL
        THEN person_id END) AS n_person_text
FROM note
"""
print("OVERALL")
print(run(sql1))
print("")

sql2 = """
SELECT
  load_table_id,
  note_title,
  note_class_concept_id,
  note_type_concept_id,
  COUNT(*) AS n,
  COUNTIF(note_text IS NOT NULL) AS n_text
FROM note
GROUP BY 1, 2, 3, 4
HAVING n_text > 0
ORDER BY n_text DESC
LIMIT 40
"""
print("GROUPS WITH TEXT")
d2 = run(sql2)
print(d2.to_string())
print("")

sql3 = """
SELECT note_id, person_id, note_DATE,
       note_title, load_table_id,
       LENGTH(note_text) AS n_chars,
       SUBSTR(note_text, 1, 300) AS snippet
FROM note
WHERE note_text IS NOT NULL
LIMIT 5
"""
print("SAMPLES")
d3 = run(sql3)
for _, r in d3.iterrows():
    print("-" * 55)
    print(r["note_title"], "|",
          r["load_table_id"], "|",
          r["n_chars"], "chars")
    print(r["snippet"])

