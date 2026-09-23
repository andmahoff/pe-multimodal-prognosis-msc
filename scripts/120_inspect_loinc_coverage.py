import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import Counter

BASE = "."
INS = BASE + "/inspect_files"
DATA = BASE + "/fusion_workspace/data"
MEDS = INS + "/meds_omop_inspect"
SHARDS = MEDS + "/data/*.parquet"
CODES = MEDS + "/metadata/codes.parquet"
OUT = DATA + "/inspect_loinc_coverage.csv"

print("codes.parquet schema")
cf = pq.ParquetFile(CODES)
print(cf.schema)
cd = pq.read_table(CODES).to_pandas()
print("rows:", len(cd))
print(cd.head(5).to_string())

desc = None
lc = [c for c in cd.columns
      if c.lower() in ("code",)]
dc = [c for c in cd.columns
      if "desc" in c.lower()
      or "title" in c.lower()
      or "name" in c.lower()]
print("")
print("code col:", lc, " desc col:", dc)
if lc and dc:
    desc = cd[[lc[0], dc[0]]].copy()
    desc.columns = ["code", "desc"]
    desc = desc.drop_duplicates("code")
    print("descriptions available:", len(desc))

coh = pd.read_csv(
    DATA + "/inspect_count_cohort.csv")
mp = pd.read_csv(
    INS + "/study_mapping_20250611.tsv",
    sep="\t")
mp["pdt"] = pd.to_datetime(
    mp["procedure_DATETIME"], errors="coerce")
f = mp.sort_values(["person_id", "pdt"])
f = f.groupby("person_id", as_index=False).first()
coh = coh.merge(f[["person_id", "pdt"]],
                on="person_id", how="left")
coh = coh[coh["pdt"].notna()]
idx = dict(zip(coh["person_id"], coh["pdt"]))
subs = set(coh["person_id"])
ni = len(coh)
print("")
print("INSPECT cohort with index date:", ni)

W = {"d7": 7, "d30": 30, "d365": 365}
seen = {k: set() for k in W}
seen["all"] = set()
nrow = 0

files = sorted(glob.glob(SHARDS))
for i, fp in enumerate(files):
    t = pq.read_table(fp, columns=[
        "subject_id", "time", "code",
        "numeric_value"])
    d = t.to_pandas()
    d = d[d["subject_id"].isin(subs)]
    if len(d) == 0:
        continue
    d = d[d["numeric_value"].notna()]
    d["code"] = d["code"].astype(str)
    d = d[d["code"].str.startswith("LOINC/")]
    if len(d) == 0:
        continue
    d["idx"] = d["subject_id"].map(idx)
    d["time"] = pd.to_datetime(d["time"],
                               errors="coerce")
    d = d[d["time"] < d["idx"]]
    if len(d) == 0:
        continue
    nrow += len(d)
    pairs = list(zip(d["subject_id"],
                     d["code"]))
    seen["all"].update(pairs)
    for k, nd in W.items():
        lo = d["idx"] - pd.Timedelta(days=nd)
        w = d[d["time"] >= lo]
        if len(w):
            seen[k].update(
                zip(w["subject_id"], w["code"]))
    if (i + 1) % 20 == 0:
        print("  shard %d/%d rows %d"
              % (i + 1, len(files), nrow))

print("")
print("LOINC numeric events pre-index:", nrow)

cols = {}
for k in ["d7", "d30", "d365", "all"]:
    cols[k] = Counter(c for _, c in seen[k])

allc = sorted(cols["all"].keys())
r = pd.DataFrame({"code": allc})
for k in ["d7", "d30", "d365", "all"]:
    r[k] = [cols[k].get(c, 0) / ni
            for c in allc]
r["loinc"] = r["code"].str[6:]
if desc is not None:
    r = r.merge(desc, on="code", how="left")
else:
    r["desc"] = ""

r = r.sort_values("d30", ascending=False)
r.to_csv(OUT, index=False)

print("")
print("distinct LOINC codes:", len(r))
for t in [0.3, 0.5, 0.7, 0.9]:
    print("  d7 >=%.1f: %3d   d30 >=%.1f: %3d"
          "   all >=%.1f: %3d"
          % (t, int((r["d7"] >= t).sum()),
             t, int((r["d30"] >= t).sum()),
             t, int((r["all"] >= t).sum())))

print("")
print("TOP 80 BY 30-DAY COVERAGE")
sh = r.head(80)[["loinc", "desc", "d7",
                 "d30", "d365", "all"]]
print(sh.round(3).to_string(index=False))
print("")
print("saved", OUT)

