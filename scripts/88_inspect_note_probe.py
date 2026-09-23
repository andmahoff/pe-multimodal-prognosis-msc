import redivis

OWN = "shahlab"
DS = "inspect_ehr:dzc6:v1_2"

ds = redivis.user(OWN).dataset(DS)

tabs = ds.list_tables()
print("total tables:", len(tabs))
print("")

hits = []
for t in tabs:
    nm = t.name
    low = nm.lower()
    if ("note" in low or "report" in low
            or "impress" in low or "text" in low):
        hits.append(t)
    print(nm)

print("")
print("=" * 50)
print("note-like tables:", [t.name for t in hits])
print("=" * 50)

for t in hits:
    print("")
    print("TABLE:", t.name)
    try:
        t.get()
        print("  rows:", t.properties.get("numRows"))
    except Exception as e:
        print("  meta failed:", e)
    try:
        vs = t.list_variables()
        print("  vars:", [v.name for v in vs])
    except Exception as e:
        print("  vars failed:", e)
    try:
        df = t.to_pandas_dataframe(
            max_results=3,
            dtype_backend="numpy")
        print(df.head())
    except Exception as e:
        print("  sample failed:", e)
