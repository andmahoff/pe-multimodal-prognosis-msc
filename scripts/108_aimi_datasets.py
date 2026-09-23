import redivis

ds = None
for how in ["org", "user"]:
    try:
        h = (redivis.organization("aimi")
             if how == "org"
             else redivis.user("aimi"))
        ds = h.list_datasets()
        print("listed via", how, "-", len(ds))
        break
    except Exception as e:
        print(how, "failed:", e)

if ds:
    for d in ds:
        low = d.name.lower()
        mark = ""
        for w in ["inspect", "impress", "report",
                  "label", "cohort", "text"]:
            if w in low:
                mark = "   (possible match)"
                break
        print("  %s%s" % (d.name, mark))
else:
    print("")
    print("enumeration failed - trying names")
    CAND = ["inspect_impressions",
            "inspect_reports",
            "inspect_radiology_reports",
            "inspect_radiology",
            "inspect_text", "inspect_labels",
            "inspect_cohort", "inspect_tabular",
            "inspect_metadata", "inspect_ehr",
            "inspect_anon_impression",
            "inspect"]
    for nm in CAND:
        for how in ["user", "org"]:
            try:
                h = (redivis.user("aimi")
                     if how == "user"
                     else redivis.organization(
                         "aimi"))
                d = h.dataset(nm)
                d.get()
                print("HIT", how, nm, "->",
                      d.properties.get(
                          "qualifiedReference"))
            except Exception:
                pass

print("")
print("=" * 55)
print("versions / tables of 2n96")
d = redivis.user("aimi").dataset("inspect:2n96")
d.get()
try:
    for v in d.list_versions():
        print("  version:", v.properties.get("tag"))
except Exception as e:
    print("  list_versions failed:", e)
for t in d.list_tables():
    print("  table:", t.name)

