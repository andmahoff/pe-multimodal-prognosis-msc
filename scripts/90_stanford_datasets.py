import redivis

def show(label, lister):
    print("=" * 55)
    print(label)
    try:
        ds = lister()
    except Exception as e:
        print("  failed:", type(e).__name__, e)
        return
    print("  datasets:", len(ds))
    for d in ds:
        nm = d.name
        low = nm.lower()
        mark = ""
        if "inspect" in low or "ctpa" in low:
            mark = "   (possible match)"
        if "note" in low or "report" in low:
            mark = "   (possible match)"
        print("   ", nm, mark)


show("user: shahlab",
     lambda: redivis.user("shahlab").list_datasets())

for slug in ["StanfordPHS", "stanfordphs",
             "Stanford", "stanford_medicine",
             "StanfordMedicine", "som-shahlab"]:
    show("org: " + slug,
         lambda s=slug:
         redivis.organization(s).list_datasets())

