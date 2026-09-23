import os
import re
import pandas as pd
from google.cloud import bigquery

PROJ = os.environ.get("GCP_PROJECT_ID")
BASE = "."
DATA = BASE + "/fusion_workspace/data"
LAB = DATA + "/mimic_labels_harmonised.csv"
OUT = DATA + "/ctpa_report_coverage.csv"

# set this to your MIMIC-IV-Ext-PE csv, or None
EXTPE = None

client = bigquery.Client(project=PROJ)

lab = pd.read_csv(LAB)
print("cohort admissions:", len(lab))
subs = sorted(lab["subject_id"].unique().tolist())
hadms = sorted(lab["hadm_id"].unique().tolist())
print("subjects:", len(subs))

sub_str = ",".join(str(s) for s in subs)
hadm_str = ",".join(str(h) for h in hadms)

adm_q = """
SELECT hadm_id, admittime, dischtime
FROM `physionet-data.mimiciv_3_1_hosp.admissions`
WHERE hadm_id IN (%s)
""" % hadm_str

adm = client.query(adm_q).to_dataframe()
print("admissions pulled:", len(adm))

pat = r"(?i)(CTPA|CTA chest|chest CTA|CTA of the chest"
pat += r"|CTA thorax|torso CTA|CTA torso"
pat += r"|CTA pulmonary angiogram)"

note_q = """
SELECT note_id, subject_id, hadm_id, charttime,
       LENGTH(text) AS n_chars
FROM `physionet-data.mimiciv_note.radiology`
WHERE subject_id IN (%s)
  AND REGEXP_CONTAINS(text, r'%s')
""" % (sub_str, pat)

nt = client.query(note_q).to_dataframe()
print("candidate CTPA notes:", len(nt))
print("unique subjects with one:", nt["subject_id"].nunique())

# join every note to every index admission of that subject
key = lab[["subject_id", "hadm_id"]].copy()
key = key.merge(adm, on="hadm_id", how="left")
key = key.rename(columns={"hadm_id": "idx_hadm"})

m = nt.merge(key, on="subject_id", how="inner")
print("note x index-admission pairs:", len(m))

for c in ["charttime", "admittime", "dischtime"]:
    m[c] = pd.to_datetime(m[c], errors="coerce")

before = m["charttime"] < m["admittime"]
after = m["charttime"] > m["dischtime"]
before = before.fillna(False)
after = after.fillna(False)

m["rel"] = "during"
m.loc[before.values, "rel"] = "before"
m.loc[after.values, "rel"] = "after"
print(m["rel"].value_counts())

dur = m[m["rel"] == "during"]
n_dur = dur.groupby("idx_hadm").size()
n_dur = n_dur.rename("n_ctpa_dur")

cov = lab[["subject_id", "hadm_id"]].copy()
cov = cov.merge(n_dur, left_on="hadm_id",
                right_index=True, how="left")
cov["n_ctpa_dur"] = cov["n_ctpa_dur"].fillna(0)

has = cov["n_ctpa_dur"] > 0
print("")
print("=" * 50)
print("admissions with >=1 index CTPA report:",
      int(has.sum()), "of", len(cov))
print("=" * 50)

# event counts on that subset, per outcome
outs = ["cv_first", "composite_30d",
        "death_30d", "death_30d_inhosp"]
lj = lab.set_index("hadm_id")
for o in outs:
    if o not in lab.columns:
        continue
    y = lj.loc[cov["hadm_id"].values, o].values
    sub = y[has.values]
    print(o, "-> n=%d ev=%d (%.4f)"
          % (len(sub), int(sub.sum()), sub.mean()))

cov.to_csv(OUT, index=False)
print("saved", OUT)

if EXTPE:
    ep = pd.read_csv(EXTPE)
    print("")
    print("Ext-PE columns:", list(ep.columns))
    print(ep.head())

