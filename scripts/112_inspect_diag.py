import re
import os
import numpy as np
import pandas as pd

BASE = "."
INS = BASE + "/inspect_files"
DATA = BASE + "/fusion_workspace/data"
D = "_20250611.tsv"

imp = pd.read_csv(INS + "/impressions" + D,
                  sep="\t")
lab = pd.read_csv(INS + "/labels" + D, sep="\t")
spl = pd.read_csv(INS + "/splits" + D, sep="\t")
mp = pd.read_csv(INS + "/study_mapping" + D,
                 sep="\t")

print("impressions %d  labels %d  splits %d"
      " mapping %d"
      % (len(imp), len(lab), len(spl), len(mp)))

d = imp.merge(lab, on="impression_id")
d = d.merge(spl, on="impression_id")
d = d.merge(mp[["impression_id", "person_id",
                "procedure_DATETIME", "note_id",
                "image_id"]],
            on="impression_id",
            suffixes=("", "_m"))
print("merged:", len(d))
print("unique persons:", d["person_id"].nunique())

d["txt"] = d["impressions"].fillna("").astype(str)
d["L"] = d["txt"].str.len()
print("")
print("=" * 55)
print("INSPECT impression length")
print(d["L"].describe().round(1).to_string())
print("  empty:", int((d["L"] == 0).sum()))

# MIMIC comparison
def get_imp(t):
    t = str(t)
    m = re.search(r"IMPRESSION[S]?\s*:", t,
                  flags=re.I)
    if not m:
        return ""
    s = t[m.end():]
    c = re.search(r"\n\s*(?:ADDENDUM"
                  r"|NOTIFICATION|WET READ"
                  r"|PRELIMINARY)", s, flags=re.I)
    if c:
        s = s[:c.start()]
    s = re.sub(r"_{2,}", " ", s)
    return re.sub(r"\s+", " ", s).strip()


p = DATA + "/ctpa_notes_index.csv"
if os.path.exists(p):
    mm = pd.read_csv(p)
    mm["charttime"] = pd.to_datetime(
        mm["charttime"], errors="coerce")
    mm = mm.sort_values(["idx_hadm",
                         "charttime"])
    mm = mm.groupby("idx_hadm",
                    as_index=False).first()
    mm["imp"] = mm["text"].apply(get_imp)
    mm["L"] = mm["imp"].str.len()
    print("")
    print("MIMIC impression length (n=%d)"
          % len(mm))
    print(mm["L"].describe().round(1).to_string())

print("")
print("=" * 55)
print("CTPAs per person")
vc = d.groupby("person_id").size()
print(vc.describe().round(2).to_string())
print("persons with >1:", int((vc > 1).sum()))

d["pdt"] = pd.to_datetime(
    d["procedure_DATETIME"], errors="coerce")
print("procedure_DATETIME parsed:",
      int(d["pdt"].notna().sum()), "of", len(d))
print("range:", d["pdt"].min(), "->",
      d["pdt"].max())

print("")
print("=" * 55)
print("NATIVE LABELS")
LB = ["1_month_mortality", "6_month_mortality",
      "12_month_mortality",
      "1_month_readmission",
      "6_month_readmission",
      "12_month_readmission", "12_month_PH"]
for c in LB:
    s = d[c].astype(str).str.strip().str.lower()
    n_t = int((s == "true").sum())
    n_f = int((s == "false").sum())
    n_c = len(s) - n_t - n_f
    den = n_t + n_f
    r = n_t / den if den else np.nan
    print("  %-22s pos %5d neg %6d cens %5d"
          "  rate %.4f"
          % (c, n_t, n_f, n_c, r))

print("")
print("PE labels")
for c in ["pe_positive", "pe_positive_nlp",
          "pe_acute", "pe_subsegmentalonly"]:
    if c not in d.columns:
        continue
    s = pd.to_numeric(d[c], errors="coerce")
    if s.notna().sum() == 0:
        s = (d[c].astype(str).str.lower()
             == "true").astype(int)
    print("  %-20s mean %.4f"
          % (c, float(s.mean())))

print("")
print("tte_mortality (censored vs not)")
tt = pd.to_numeric(d["tte_mortality"],
                   errors="coerce")
cz = d["is_censored_mortality"].astype(
    str).str.lower() == "true"
print("  censored n=%d median %.1f"
      % (int(cz.sum()),
         float(tt[cz].median())))
print("  events   n=%d median %.1f"
      % (int((~cz).sum()),
         float(tt[~cz].median())))
print("  (if minutes, /1440 = days)")
print("  events median days: %.1f"
      % (float(tt[~cz].median()) / 1440.0))

print("")
print("=" * 55)
print("OVERLAP WITH HARMONISED COHORT")
hp = DATA + "/inspect_labels_final.csv"
if os.path.exists(hp):
    hl = pd.read_csv(hp)
    print("harmonised rows:", len(hl))
    print("cols:", list(hl.columns))
    pid = set(hl["person_id"])
    inb = d["person_id"].isin(pid)
    print("studies in PE-positive cohort:",
          int(inb.sum()), "of", len(d))
    print("persons matched:",
          d.loc[inb, "person_id"].nunique(),
          "of", len(pid))
    pp = pd.to_numeric(d["pe_positive"],
                       errors="coerce").fillna(0)
    print("")
    print("agreement INSPECT pe_positive vs"
          " PE-code cohort:")
    print(pd.crosstab(inb, pp > 0))
else:
    print("missing", hp)

print("")
print("=" * 55)
print("SAMPLE IMPRESSIONS")
sm = d[d["L"] > 100].head(3)
for _, r in sm.iterrows():
    print("-" * 50)
    print("len", r["L"], "pe_pos",
          r["pe_positive"])
    print(r["txt"][:700])

