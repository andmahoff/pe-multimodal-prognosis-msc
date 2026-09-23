import os
import redivis

OUT = "./inspect_files"
os.makedirs(OUT, exist_ok=True)

org = redivis.organization("aimi")
alld = org.list_datasets()
print("datasets:", len(alld))

hits = [d for d in alld
        if d.name.strip().lower() == "inspect"]
if not hits:
    hits = [d for d in alld
            if "inspect" in d.name.lower()]

print("matched:", [d.name for d in hits])

for d in hits:
    print("")
    print("#" * 55)
    print("DATASET:", d.name)
    try:
        d.get()
    except Exception as e:
        print("  get failed:", e)
    for k in ["qualifiedReference", "name",
              "id", "version"]:
        try:
            print("  %s: %s"
                  % (k, d.properties.get(k)))
        except Exception:
            pass

    try:
        tabs = d.list_tables()
    except Exception as e:
        print("  list_tables failed:", e)
        continue

    for t in tabs:
        n = None
        try:
            t.get()
            n = t.properties.get("numRows")
        except Exception:
            pass
        print("")
        print("  TABLE %-32s rows=%s"
              % (t.name, n))
        try:
            vs = [v.name
                  for v in t.list_variables()]
            print("    vars:", vs)
        except Exception:
            pass
        try:
            fs = t.list_files()
            print("    files:", len(fs))
            for f in fs[:20]:
                print("     ", f.name, f.size)
                low = f.name.lower()
                if f.size > 100_000_000:
                    continue
                if ("impress" in low
                        or "cohort" in low
                        or "label" in low
                        or low.endswith(".csv")):
                    try:
                        f.download(path=OUT,
                                   overwrite=True)
                        print("        downloaded")
                    except Exception as e:
                        print("        dl fail:", e)
        except Exception:
            pass
        if n and n < 200000:
            try:
                df = t.to_pandas_dataframe(
                    max_results=3,
                    dtype_backend="numpy")
                print(df.to_string()[:1200])
            except Exception as e:
                print("    sample failed:", e)

print("")
print("saved to", OUT)

