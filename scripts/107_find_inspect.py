import redivis

ID = "2n96"
OWNERS = ["shahlab", "stanford", "Stanford",
          "StanfordAIMI", "stanfordaimi",
          "aimi", "som-shahlab", "phsdata",
          "StanfordMedicine"]
NAMES = [
    "inspect",
    "INSPECT",
    "inspect_multimodal",
    "inspect_a_multimodal_dataset_for_"
    "pulmonary_embolism_diagnosis_and_prognosis",
    "inspect__a_multimodal_dataset_for_"
    "pulmonary_embolism_diagnosis_and_prognosis",
]

refs = []
for n in NAMES:
    refs.append(n + ":" + ID)
    refs.append(n)
refs.append(ID)
refs.append("2n96-d71hggrbf")

ds = None
for ow in OWNERS:
    for rf in refs:
        for kind in ["user", "org"]:
            try:
                if kind == "user":
                    d = redivis.user(ow).dataset(rf)
                else:
                    d = redivis.organization(
                        ow).dataset(rf)
                d.get()
                print("SUCCESS", kind, ow, "|", rf)
                ds = d
                break
            except Exception:
                pass
        if ds:
            break
    if ds:
        break

if ds is None:
    print("no combination resolved")
    raise SystemExit

print("")
print("qualified:",
      ds.properties.get("qualifiedReference"))
print("name:", ds.properties.get("name"))

print("")
print("TABLES")
for t in ds.list_tables():
    n = None
    try:
        t.get()
        n = t.properties.get("numRows")
    except Exception:
        pass
    print("  %-45s rows=%s" % (t.name, n))
    try:
        vs = [v.name for v in t.list_variables()]
        print("     vars:", vs[:12])
    except Exception:
        pass
    try:
        fs = t.list_files()
        for f in fs[:8]:
            print("     file:", f.name, f.size)
    except Exception:
        pass

