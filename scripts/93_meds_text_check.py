import pyarrow.parquet as pq
import pyarrow.compute as pc

BASE = "."
P = BASE + "/inspect_files/meds_omop_inspect"
P = P + "/data/data_4.parquet"

cols = ["subject_id", "code", "note_id",
        "text_value", "table"]
t = pq.read_table(P, columns=cols)
n = t.num_rows
print("rows:", n)

tv = t.column("text_value")
ni = t.column("note_id")
print("text_value non-null:", n - tv.null_count)
print("note_id   non-null:", n - ni.null_count)

mask = pc.is_valid(tv)
sub = t.filter(mask)
print("rows with text:", sub.num_rows)

if sub.num_rows == 0:
    print("no text in this shard")
    raise SystemExit

d = sub.to_pandas()
d["L"] = d["text_value"].str.len()

print("")
print("length stats:")
print(d["L"].describe())

print("")
print("top codes among text rows:")
print(d["code"].value_counts().head(15))

print("")
print("top source tables:")
print(d["table"].value_counts().head(10))

print("")
print("longest 3 samples:")
d = d.sort_values("L", ascending=False)
for _, r in d.head(3).iterrows():
    print("-" * 50)
    print("code:", r["code"])
    print("note_id:", r["note_id"])
    print("len:", r["L"])
    print(str(r["text_value"])[:600])

