import os
import redivis

DS = "inspect_ehr:dzc6:v1_2"
OUT = "./inspect_files"
os.makedirs(OUT, exist_ok=True)

ds = redivis.user("shahlab").dataset(DS)

for nm in ["Linkage", "MEDS"]:
    print("=" * 55)
    print("TABLE:", nm)
    t = ds.table(nm)
    try:
        fs = t.list_files()
    except Exception as e:
        print("  list_files failed:", e)
        continue
    for f in fs:
        try:
            print("  file:", f.name, f.size)
        except Exception:
            print("  file obj:", dir(f))
            continue
        if nm == "MEDS" and "reader" in f.name:
            print("   skipping reader tarball")
            continue
        try:
            f.download(path=OUT, overwrite=True)
            print("   downloaded")
        except Exception as e:
            print("   download failed:", e)

print("")
print("done ->", OUT)

