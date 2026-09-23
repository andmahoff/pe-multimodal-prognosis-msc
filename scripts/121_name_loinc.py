import pandas as pd
import pyarrow.parquet as pq

BASE = "."
INS = BASE + "/inspect_files"
DATA = BASE + "/fusion_workspace/data"
CODES = (INS + "/meds_omop_inspect"
         "/metadata/codes.parquet")
COV = DATA + "/inspect_loinc_coverage.csv"
OUT = DATA + "/inspect_loinc_named.csv"

cd = pq.read_table(CODES).to_pandas()
cd["code"] = cd["code"].astype(str)
print("codes.parquet rows:", len(cd))

pref = cd["code"].str.split("/", n=1).str[0]
print("")
print("vocabulary prefixes (top 20):")
print(pref.value_counts().head(20).to_string())

lo = cd[pref == "LOINC"]
print("")
print("LOINC entries:", len(lo))
print(lo.head(8)[["code", "description"]]
      .to_string(index=False))

cov = pd.read_csv(COV)
print("")
print("coverage rows:", len(cov))
print("sample codes:", cov["code"].head(3).tolist())

s1 = set(cov["code"])
s2 = set(cd["code"])
print("")
print("exact overlap:", len(s1 & s2))

if len(s1 & s2) < len(s1) / 2:
    print("-> trying match on bare LOINC id")
    cd["bare"] = cd["code"].str.split(
        "/", n=1).str[-1]
    m = cd.drop_duplicates("bare")[
        ["bare", "description"]]
    cov = cov.merge(m, left_on="loinc",
                    right_on="bare", how="left")
else:
    m = cd.drop_duplicates("code")[
        ["code", "description"]]
    cov = cov.drop(columns=["desc"],
                   errors="ignore")
    cov = cov.merge(m, on="code", how="left")

got = cov["description"].notna().mean()
print("")
print("named fraction: %.3f" % got)

cov = cov.sort_values("d7", ascending=False)
cov.to_csv(OUT, index=False)

print("")
print("TOP 70 BY 7-DAY COVERAGE")
print(cov.head(70)[["loinc", "description",
                    "d7", "d30", "all"]]
      .round(3).to_string(index=False))
print("")
print("saved", OUT)

