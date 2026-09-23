import os
import pandas as pd

BASE = "."
REPO = BASE + "/mimic-omop/extras/concept"
DATA = BASE + "/fusion_workspace/data"
OUT = DATA + "/mimic_inspect_map.csv"

FILES = ["lab_label_to_concept.csv",
         "chart_label_to_concept.csv",
         "labs_from_chartevents_to_concept.csv"]

for f in FILES:
    p = os.path.join(REPO, f)
    print("")
    print("=" * 58)
    print(f)
    try:
        d = pd.read_csv(p)
    except Exception:
        d = pd.read_csv(p, sep="\t")
    print("rows:", len(d))
    print("cols:", list(d.columns))
    print(d.head(4).to_string())

lab = pd.read_csv(
    os.path.join(REPO, "lab_label_to_concept.csv"))
cols = [c.lower() for c in lab.columns]
lab.columns = cols
print("")
print("=" * 58)
print("lab_label_to_concept columns:", cols)

idc = [c for c in cols if "itemid" in c]
cnc = [c for c in cols
       if "concept" in c and "id" in c]
lbc = [c for c in cols
       if "label" in c or "name" in c]
print("itemid:", idc, "concept:", cnc,
      "label:", lbc)

nm = pd.read_csv(
    DATA + "/inspect_loinc_named.csv")
print("")
print("INSPECT named coverage rows:", len(nm))
print(nm.head(3).to_string())

