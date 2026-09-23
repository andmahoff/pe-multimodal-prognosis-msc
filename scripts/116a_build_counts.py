import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import Counter

BASE = "."
INS = BASE + "/inspect_files"
DATA = BASE + "/fusion_workspace/data"
D = "_20250611.tsv"
SHARDS = INS + "/meds_omop_inspect/data/*.parquet"

GAP_D = 30
MIN_SUBJ = 50

spl = pd.read_csv(INS + "/splits" + D, sep="\t")
mp = pd.read_csv(INS + "/study_mapping" + D,
                 sep="\t")
lb = pd.read_csv(INS + "/labels" + D, sep="\t")
hl = pd.read_csv(
    DATA + "/inspect_labels_final.csv")
hl["pdate"] = pd.to_datetime(hl["pe_date"],
                             errors="coerce")

g = spl.merge(mp[["impression_id",
                  "procedure_DATETIME"]],
              on="impression_id")
g = g.merge(lb[["impression_id",
                "1_month_mortality"]],
            on="impression_id")
g["pdt"] = pd.to_datetime(
    g["procedure_DATETIME"], errors="coerce")
g = g[g["person_id"].isin(set(hl["person_id"]))]
g = g.merge(hl[["person_id", "pdate"]],
            on="person_id", how="left")
g["gap"] = (g["pdt"] - g["pdate"]).abs()
g = g[g["gap"] <= pd.Timedelta(days=GAP_D)]
g = g.sort_values(["person_id", "gap"])
coh = g.groupby("person_id",
                as_index=False).first()
print("cohort:", len(coh))

s = coh["1_month_mortality"].astype(
    str).str.lower()
coh = coh[s.isin(["true", "false"])].copy()
coh["y"] = (coh["1_month_mortality"].astype(str)
            .str.lower() == "true").astype(int)
print("after censoring drop:", len(coh),
      "events:", int(coh["y"].sum()))

idx = dict(zip(coh["person_id"], coh["pdt"]))
subs = set(coh["person_id"])

files = sorted(glob.glob(SHARDS))
print("shards:", len(files))
df_cnt = Counter()
per_sub = {}
n_ev = 0

for i, f in enumerate(files):
    t = pq.read_table(f, columns=["subject_id",
                                  "time", "code"])
    d = t.to_pandas()
    d = d[d["subject_id"].isin(subs)]
    if len(d) == 0:
        continue
    d["idx"] = d["subject_id"].map(idx)
    d["time"] = pd.to_datetime(d["time"],
                               errors="coerce")
    d = d[d["time"] < d["idx"]]
    if len(d) == 0:
        continue
    n_ev += len(d)
    d["code"] = d["code"].astype(str)
    cc = d["code"].str.split("/", n=1,
                             expand=True)
    voc = cc[0]
    body = cc[1].fillna("") if cc.shape[1] > 1 \
        else pd.Series([""] * len(d),
                       index=d.index)
    d["par"] = voc + "/" + body.str[:3]

    a = d.groupby(["subject_id",
                   "code"]).size()
    b = d.groupby(["subject_id",
                   "par"]).size()
    b.index = b.index.set_names(["subject_id",
                                 "code"])
    both = pd.concat([a, b])
    seen = set()
    for (sid, code), n in both.items():
        per_sub.setdefault(sid,
                           Counter())[code] += n
        seen.add(code)
    df_cnt.update(seen)
    if (i + 1) % 10 == 0:
        print("  shard %d/%d  events %d"
              "  codes %d"
              % (i + 1, len(files), n_ev,
                 len(df_cnt)))

print("pre-index events kept:", n_ev)
print("distinct codes:", len(df_cnt))
keep = sorted([c for c, n in df_cnt.items()
               if n >= MIN_SUBJ])
print("kept (>=%d shards-subjects): %d"
      % (MIN_SUBJ, len(keep)))
kidx = {c: i for i, c in enumerate(keep)}

pids = coh["person_id"].values
X = np.zeros((len(pids), len(keep)),
             dtype=np.float32)
for r, p in enumerate(pids):
    cnts = per_sub.get(p)
    if not cnts:
        continue
    for c, n in cnts.items():
        j = kidx.get(c)
        if j is not None:
            X[r, j] = n

nz = (X > 0).sum(axis=1)
print("codes per subject: median %.0f"
      "  mean %.0f  max %.0f  zero-rows %d"
      % (np.median(nz), nz.mean(), nz.max(),
         int((nz == 0).sum())))

np.save(DATA + "/inspect_count_X.npy", X)
pd.Series(keep).to_csv(
    DATA + "/inspect_count_cols.csv",
    index=False, header=False)
coh[["person_id", "y", "split"]].to_csv(
    DATA + "/inspect_count_cohort.csv",
    index=False)
print("saved X", X.shape)

