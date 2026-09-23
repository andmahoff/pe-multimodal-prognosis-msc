import os
import redivis

OUT = "./inspect_files"
os.makedirs(OUT, exist_ok=True)

REFS = ["inspect:2n96", "2n96",
        "INSPECT:2n96",
        "inspect_multimodal:2n96"]
OWNERS = ["shahlab", "stanford", "Stanford"]

ds = None
for ow in OWNERS:
    for rf in REFS:
        try:
            d = redivis.user(ow).dataset(rf)
            d.get()
            ds = d
            print("OK user:", ow, "ref:", rf)
            break
        except Exception:
            pass
        try:
            d = redivis.organization(ow).dataset(rf)
            d.get()
            ds = d
            print("OK org:", ow, "ref:", rf)
            break
        except Exception:
            pass
    if ds:
        break

if ds is None:
    raise SystemExit("could not resolve dataset")

tabs = ds.list_tables()
print("tables:", len(tabs))
for t in tabs:
    print("  ", t.name)

WANT = ["impression", "cohort", "label"]

for t in tabs:
    print("")
    print("=" * 55)
    print("TABLE:", t.name)
    try:
        fs = t.list_files()
    except Exception as e:
        print("  not a file table:", type(e).__name__)
        continue
    for f in fs:
        nm = f.name
        print("  file:", nm, f.size)
        low = nm.lower()
        small = f.size < 200_000_000
        hit = any(w in low for w in WANT)
        if hit and small:
            try:
                f.download(path=OUT, overwrite=True)
                print("    downloaded")
            except Exception as e:
                print("    failed:", e)

