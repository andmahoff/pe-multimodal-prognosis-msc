
import pandas as pd
import numpy as np

BASE = "."
DATA = BASE + "/fusion_workspace/data"
NOTE = BASE + "/mimic_note/radiology.csv.gz"
LAB = DATA + "/mimic_labels_harmonised.csv"
ADM = DATA + "/index_admission_times.csv"
OUT = DATA + "/ctpa_notes_index.csv"

PRE_H = 48

PAT = (r"CTPA"
       r"|CTA CHEST|CHEST CTA"
       r"|CTA OF THE CHEST"
       r"|CTA THORAX|TORSO CTA|CTA TORSO"
       r"|CT ANGIOGRAM|CT ANGIOGRAPHY"
       r"|PULMONARY ANGIOGRAM"
       r"|PULMONARY ANGIOGRAPHY"
       r"|PULMONARY EMBOL"
       r"|PE PROTOCOL|P\.E\. PROTOCOL"
       r"|RULE OUT PE|R/O PE")

lab = pd.read_csv(LAB)
adm = pd.read_csv(ADM)
subs = set(lab["subject_id"].unique())
print("cohort admissions:", len(lab))
print("cohort subjects:", len(subs))

cols = ["note_id", "subject_id", "hadm_id",
        "charttime", "text"]
keep = []
seen = 0
it = pd.read_csv(NOTE, usecols=cols,
                 chunksize=50000)
for ch in it:
    seen += len(ch)
    ch = ch[ch["subject_id"].isin(subs)]
    if len(ch) == 0:
        continue
    head = ch["text"].fillna("").str[:600]
    hit = head.str.upper().str.contains(
        PAT, regex=True)
    ch = ch[hit]
    if len(ch):
        keep.append(ch)
    if seen % 500000 == 0:
        got = sum(len(k) for k in keep)
        print("scanned", seen, "kept", got)

nt = pd.concat(keep, ignore_index=True)
print("candidate CTPA notes:", len(nt))
print("subjects:", nt["subject_id"].nunique())

key = adm.rename(columns={"hadm_id": "idx_hadm"})
key = key[["subject_id", "idx_hadm",
           "admittime", "dischtime"]]
m = nt.merge(key, on="subject_id", how="inner")

for c in ["charttime", "admittime", "dischtime"]:
    m[c] = pd.to_datetime(m[c], errors="coerce")

dt = (m["admittime"] - m["charttime"])
m["h_before"] = dt.dt.total_seconds() / 3600.0

print("")
print("WINDOW SENSITIVITY")
print("hours_pre  notes  admissions")
for h in [0, 6, 12, 24, 48, 72, 168]:
    lo = m["charttime"] >= (
        m["admittime"] - pd.Timedelta(hours=h))
    hi = m["charttime"] <= m["dischtime"]
    ok = (lo & hi).fillna(False)
    sub = m[ok.values]
    print("%9d  %5d  %10d"
          % (h, len(sub), sub["idx_hadm"].nunique()))

lo = m["charttime"] >= (
    m["admittime"] - pd.Timedelta(hours=PRE_H))
hi = m["charttime"] <= m["dischtime"]
ok = (lo & hi).fillna(False)
sel = m[ok.values].copy()
sel.to_csv(OUT, index=False)

nadm = sel["idx_hadm"].nunique()
print("")
print("=" * 50)
print("PRE_H = %d hours" % PRE_H)
print("admissions with index CTPA:",
      nadm, "of", len(lab))
print("=" * 50)

lj = lab.set_index("hadm_id")
ids = sel["idx_hadm"].unique()
for o in ["cv_first", "composite_30d",
          "death_30d", "death_30d_inhosp"]:
    if o not in lab.columns:
        continue
    y = lj.loc[ids, o].values
    print(o, "-> n=%d ev=%d (%.4f)"
          % (len(y), int(y.sum()), y.mean()))

print("saved", OUT)
