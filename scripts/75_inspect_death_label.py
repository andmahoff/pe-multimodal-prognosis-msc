import os
import numpy as np
import pandas as pd
import redivis

IW = (
    "./"
    "inspect_workspace/data/processed/"
)
FD = (
    "./"
    "fusion_workspace/data/"
)
OUT = os.path.join(
    FD, "inspect_death_label.csv"
)

print("Pulling death table...")
ds = redivis.user("shahlab").dataset(
    "inspect_ehr:dzc6:v1_2"
)
dth = ds.table("death:mg89").to_pandas_dataframe()
print("  death rows:", dth.shape)
print("  cols:", dth.columns.tolist())

print("\ncause_concept_id values:")
print(dth["cause_concept_id"]
      .value_counts().head().to_string())

cands = [
    c for c in dth.columns
    if c.lower() in ("death_datetime",
                     "death_date")
]
best = None
bestn = -1
for c in cands:
    n = dth[c].notna().sum()
    print("  candidate %-16s non-null %d"
          % (c, n))
    if n > bestn:
        best, bestn = c, n
print("  using:", best)

print("\nLoading cohort demographics...")
coh = pd.read_csv(
    os.path.join(IW, "cohort_demographics.csv")
)
print("  rows:", len(coh))
print("  persons:", coh["person_id"].nunique())

coh["vs"] = pd.to_datetime(
    coh["visit_start_DATETIME"],
    errors="coerce"
)
try:
    coh["vs"] = coh["vs"].dt.tz_localize(None)
except TypeError:
    pass
coh["vs"] = coh["vs"].dt.normalize()
print("  visit_start nulls:",
      int(coh["vs"].isna().sum()))
print("  visit_start range:",
      coh["vs"].min(), "->", coh["vs"].max())

d = dth[["person_id", best]].copy()
d["dd"] = pd.to_datetime(
    d[best].astype(str), errors="coerce"
)
try:
    d["dd"] = d["dd"].dt.tz_localize(None)
except TypeError:
    pass
d = d[["person_id", "dd"]].dropna()
d = d.groupby("person_id")["dd"].min()
d = d.reset_index()
print("  death dates parsed:", len(d))
print("  death range:",
      d["dd"].min(), "->", d["dd"].max())

m = coh.merge(d, on="person_id", how="left")
m["days_to_death"] = (
    m["dd"] - m["vs"]
).dt.total_seconds() / 86400.0

print("\npersons with a death record:",
      int(m["dd"].notna().sum()))
print("days_to_death describe:")
print(m["days_to_death"].describe())
print("negative (death before visit):",
      int((m["days_to_death"] < 0).sum()))

for w in [30, 90, 180, 365]:
    lab = (
        (m["days_to_death"] >= 0)
        & (m["days_to_death"] <= w)
    ).astype(int)
    print("  mortality within %3dd: "
          "%5d (%.4f)"
          % (w, int(lab.sum()), lab.mean()))

m["death_30d"] = (
    (m["days_to_death"] >= 0)
    & (m["days_to_death"] <= 30)
).astype(int)

print("\nComparing to existing mace_30d...")
old = pd.read_csv(
    os.path.join(IW, "mace_labels.csv")
)
cmp = m[[
    "person_id", "death_30d",
    "days_to_death"
]].merge(old, on="person_id", how="inner")
print("  matched persons:", len(cmp))
print("  old mace_30d rate:  %.4f"
      % cmp["mace_30d"].mean())
print("  new death_30d rate: %.4f"
      % cmp["death_30d"].mean())
print("\n  crosstab (rows=old, cols=new):")
print(pd.crosstab(
    cmp["mace_30d"], cmp["death_30d"]
).to_string())

sub = cmp[cmp["death_30d"] == 1]
print("\n  of %d 30-day deaths, %d (%.3f) "
      "were old-label positive"
      % (len(sub), int(sub["mace_30d"].sum()),
         sub["mace_30d"].mean()))

out = m[[
    "person_id", "visit_occurrence_id",
    "days_to_death", "death_30d"
]].drop_duplicates("person_id")
out.to_csv(OUT, index=False)
print("\nSaved", OUT, out.shape)

