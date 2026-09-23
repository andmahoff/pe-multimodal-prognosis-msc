import os
import re
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

BASE = "."
DATA = BASE + "/fusion_workspace/data"
OUT = DATA + "/llm_extract"
os.makedirs(OUT, exist_ok=True)
MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_TEST = 20
POST_H = 24
MAXNEW = 300

FIELDS = ["pe_present", "pe_location",
          "rv_strain", "pleural_effusion",
          "malignancy", "consolidation",
          "atelectasis", "pulmonary_oedema",
          "cardiomegaly", "lymphadenopathy",
          "ascites", "endotracheal_tube"]

SYS = (
    "You are a radiologist extracting "
    "structured findings from a CT "
    "pulmonary angiogram report. "
    "Answer ONLY with a JSON object. "
    "No explanation, no markdown.")

TMPL = """Extract these findings from the report.

For each field answer exactly one of:
"yes"  - the finding is present
"no"   - the finding is explicitly absent or negated
"na"   - the finding is not mentioned

Except pe_location, which must be one of:
"saddle", "central", "lobar", "segmental",
"subsegmental", "none", "na"

Fields: pe_present, pe_location, rv_strain,
pleural_effusion, malignancy, consolidation,
atelectasis, pulmonary_oedema, cardiomegaly,
lymphadenopathy, ascites, endotracheal_tube

REPORT:
{report}

JSON:"""


def parse(s):
    s = s.strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(
                m.group(0).replace("'", '"'))
        except Exception:
            return None


nt = pd.read_csv(DATA + "/ctpa_notes_index.csv")
for c in ["charttime", "admittime"]:
    nt[c] = pd.to_datetime(nt[c],
                           errors="coerce")
nt = nt.sort_values(["idx_hadm", "charttime"])
nt = nt.groupby("idx_hadm",
                as_index=False).first()
nt["h"] = ((nt["charttime"] - nt["admittime"])
           .dt.total_seconds() / 3600.0)
nt = nt[nt["h"] <= POST_H].reset_index(
    drop=True)
print("presentation notes:", len(nt))

fx = pd.read_csv(
    DATA + "/ctpa_comorb_features.csv")
sub = nt.sample(N_TEST,
                random_state=42).copy()
print("sampled:", len(sub))

print("loading model...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16,
    device_map="auto")
mdl.eval()
print("loaded in %.0fs  device %s"
      % (time.time() - t0,
         next(mdl.parameters()).device))

rows = []
t0 = time.time()
for i, (_, r) in enumerate(sub.iterrows()):
    txt = str(r["text"])[:6000]
    msgs = [{"role": "system",
             "content": SYS},
            {"role": "user",
             "content": TMPL.format(
                 report=txt)}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True,
        return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            ids, max_new_tokens=MAXNEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id)
    gen = tok.decode(
        out[0][ids.shape[1]:],
        skip_special_tokens=True)
    js = parse(gen)
    ok = js is not None
    rec = {"hadm_id": r["idx_hadm"],
           "parsed": ok, "raw": gen[:400]}
    if ok:
        for f in FIELDS:
            rec[f] = str(js.get(f, "")).lower()
    rows.append(rec)
    print("  %2d/%d parsed=%s"
          % (i + 1, len(sub), ok))

el = time.time() - t0
print("")
print("%.1fs total, %.1fs per note"
      % (el, el / max(len(sub), 1)))
print("estimated for 1707 notes: %.0f min"
      % (el / max(len(sub), 1) * 1707 / 60.0))

t = pd.DataFrame(rows)
t.to_csv(OUT + "/smoke_test.csv", index=False)
print("")
print("parse rate: %.2f"
      % float(t["parsed"].mean()))
if t["parsed"].sum() == 0:
    print("NO PARSES -- sample output:")
    print(t["raw"].iloc[0])
    raise SystemExit(1)

MAP = {"pe_present": "pe_pos",
       "rv_strain": "rv_strain",
       "pleural_effusion": "effusion",
       "malignancy": "malignancy",
       "consolidation": "consolid",
       "atelectasis": "atelect",
       "pulmonary_oedema": "edema",
       "cardiomegaly": "cardiomeg",
       "lymphadenopathy": "adenopathy",
       "ascites": "cm_ascites",
       "endotracheal_tube": "dv_ett"}

g = t[t["parsed"]].merge(
    fx, on="hadm_id", how="left")
print("")
print("AGREEMENT WITH REGEX (n=%d)" % len(g))
print("  %-18s %-12s %6s %6s %6s"
      % ("llm field", "regex", "agree",
         "llm+", "rgx+"))
for lf, rf in MAP.items():
    if rf not in g.columns or lf not in \
            g.columns:
        continue
    lv = (g[lf].astype(str)
          == "yes").astype(int)
    rv = g[rf].fillna(0).astype(int)
    print("  %-18s %-12s %6.2f %6d %6d"
          % (lf, rf,
             float((lv == rv).mean()),
             int(lv.sum()), int(rv.sum())))

print("")
print("saved", OUT + "/smoke_test.csv")
