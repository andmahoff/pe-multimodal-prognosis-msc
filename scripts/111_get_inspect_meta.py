import os
import redivis
import pandas as pd

OUT = "./inspect_files"
os.makedirs(OUT, exist_ok=True)

FILES = ["impressions_20250611.tsv",
         "labels_20250611.tsv",
         "splits_20250611.tsv",
         "study_mapping_20250611.tsv",
         "study_metadata_20250611.tsv",
         "series_metadata_20250611.tsv"]

d = redivis.user("aimi").dataset("inspect:2n96")
d.get()
t = d.table("full:q80g")

for nm in FILES:
    p = os.path.join(OUT, nm)
    try:
        f = t.file(nm)
        f.download(path=OUT, overwrite=True)
        print("ok  ", nm, os.path.getsize(p))
    except Exception as e:
        print("fail", nm, type(e).__name__, e)

print("")
print("#" * 55)
for nm in FILES:
    p = os.path.join(OUT, nm)
    if not os.path.exists(p):
        continue
    print("")
    print("=" * 55)
    print(nm)
    try:
        df = pd.read_csv(p, sep="\t")
    except Exception as e:
        print("  read failed:", e)
        continue
    print("  rows:", len(df))
    print("  cols:", list(df.columns))
    for c in df.columns:
        if df[c].dtype == object:
            L = df[c].astype(str).str.len()
            print("   %-28s maxlen %d"
                  % (c, int(L.max())))
    print(df.head(2).to_string()[:900])

