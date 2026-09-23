import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = "."
INS = BASE + "/inspect_files"
DATA = BASE + "/fusion_workspace/data"
SHARDS = INS + "/meds_omop_inspect/data/*.parquet"
OUT = DATA + "/inspect_expanded_feats.csv"

PRE_D = 7
GAP_D = 30

TIER_A = {
    "temp": "8310-5", "hr": "8867-4",
    "sbp": "8480-6", "dbp": "8462-4",
    "rr": "9279-1",
    "creatinine": "2160-0", "sodium": "2951-2",
    "potassium": "2823-3", "bun": "3094-0",
    "glucose": "2345-7", "calcium": "17861-6",
    "bicarb": "2028-9", "chloride": "2075-0",
    "hct": "4544-3", "plt": "777-3",
    "wbc": "20584-9", "hgb": "718-7",
    "aniongap": "33037-3", "rbc": "789-8",
    "mchc": "786-4", "mch": "785-6",
    "mcv": "787-2", "rdw": "788-0",
}
TIER_B = {
    "inr": "6301-6", "pt": "5902-2",
    "aptt": "14979-9", "magnesium": "19123-9",
    "albumin": "1751-7", "bili": "1975-2",
    "alp": "6768-6", "alt": "1742-6",
    "ast": "1920-8", "neut_pct": "770-8",
    "lymph_pct": "736-9", "mono_pct": "5905-5",
    "eos_pct": "713-8", "baso_pct": "706-2",
}
TIER_C = {
    "ntprobnp": "33762-6", "phosphate": "2777-1",
}
FLAGGED = {
    "troponin": "10839-9",
}

ALL = {}
for dd in (TIER_A, TIER_B, TIER_C, FLAGGED):
    ALL.update(dd)
code2name = {"LOINC/" + v: k
             for k, v in ALL.items()}
print("target analytes:", len(ALL))

hl = pd.read_csv(
    DATA + "/inspect_labels_final.csv")
hl["pdate"] = pd.to_datetime(hl["pe_date"],
                             errors="coerce")
mp = pd.read_csv(
    INS + "/study_mapping_20250611.tsv",
    sep="\t")
mp["pdt"] = pd.to_datetime(
    mp["procedure_DATETIME"], errors="coerce")
mp = mp[mp["pdt"].notna()]
mp = mp.merge(hl[["person_id", "pdate"]],
              on="person_id", how="inner")
mp["gap"] = (mp["pdt"] - mp["pdate"]).abs()
mp = mp[mp["gap"] <= pd.Timedelta(days=GAP_D)]
mp = mp.sort_values(["person_id", "gap"])
coh = mp.groupby("person_id",
                 as_index=False).first()
coh["idx"] = coh["pdt"]
idx = dict(zip(coh["person_id"], coh["idx"]))
subs = set(coh["person_id"])
print("cohort:", len(coh), "(index = CTPA date)")

acc7 = {}
accall = {}
nf = 0
files = sorted(glob.glob(SHARDS))
nev = 0

for i, f in enumerate(files):
    t = pq.read_table(f, columns=[
        "subject_id", "time", "code",
        "numeric_value"])
    d = t.to_pandas()
    d = d[d["subject_id"].isin(subs)]
    if len(d) == 0:
        continue
    d["code"] = d["code"].astype(str)
    d = d[d["code"].isin(code2name)]
    d = d[d["numeric_value"].notna()]
    if len(d) == 0:
        continue
    d["idx"] = d["subject_id"].map(idx)
    d["time"] = pd.to_datetime(d["time"],
                               errors="coerce")
    d = d[d["time"] < d["idx"]]
    if len(d) == 0:
        continue
    nev += len(d)
    d["name"] = d["code"].map(code2name)

    ft = ((d["name"] == "temp")
          & (d["numeric_value"] >= 50))
    nf += int(ft.sum())
    d.loc[ft, "numeric_value"] = (
        d.loc[ft, "numeric_value"]
        - 32.0) * 5.0 / 9.0

    g = d.groupby(["subject_id", "name"])[
        "numeric_value"].agg(["sum", "count"])
    for (s, n), row in g.iterrows():
        k = (s, n)
        a = accall.get(k, [0.0, 0])
        a[0] += row["sum"]
        a[1] += row["count"]
        accall[k] = a

    lo = d["idx"] - pd.Timedelta(days=PRE_D)
    w = d[d["time"] >= lo]
    if len(w):
        g2 = w.groupby(["subject_id", "name"])[
            "numeric_value"].agg(["sum", "count"])
        for (s, n), row in g2.iterrows():
            k = (s, n)
            a = acc7.get(k, [0.0, 0])
            a[0] += row["sum"]
            a[1] += row["count"]
            acc7[k] = a

    if (i + 1) % 20 == 0:
        print("  shard %d/%d events %d"
              % (i + 1, len(files), nev))

print("matched events:", nev)
print("fahrenheit temps converted:", nf)


def build(acc, tag):
    rows = {}
    for (s, n), (sm, ct) in acc.items():
        rows.setdefault(s, {})[n] = sm / ct
    df = pd.DataFrame.from_dict(rows,
                                orient="index")
    df.index.name = "person_id"
    df = df.reset_index()
    df.columns = ["person_id"] + [
        tag + "_" + c for c in df.columns[1:]]
    return df


f7 = build(acc7, "m7")
fa = build(accall, "mall")
out = coh[["person_id", "idx"]].copy()
out = out.merge(f7, on="person_id", how="left")
out = out.merge(fa, on="person_id", how="left")
out.to_csv(OUT, index=False)
print("saved", OUT, out.shape)

print("")
print("COVERAGE (non-null fraction)")
print("%-12s %7s %7s" % ("analyte", "7d", "all"))
for k in ALL:
    c = "m7_" + k
    ca = "mall_" + k
    v7 = out[c].notna().mean() if c in out else 0.0
    va = out[ca].notna().mean() if ca in out else 0.0
    print("  %-12s %.3f   %.3f" % (k, v7, va))

print("")
print("MEDIAN VALUES (7d)")
for k in ALL:
    c = "m7_" + k
    if c in out and out[c].notna().sum() > 20:
        print("  %-12s %.3f"
              % (k, float(out[c].median())))
