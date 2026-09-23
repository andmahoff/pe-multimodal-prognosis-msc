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
RAW = OUT + "/mimic_llm_raw.csv"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
POST_H = 24
MAXNEW = 300
SAVE_EVERY = 100

FIELDS = ["pe_present", "pe_location",
          "rv_strain", "pleural_effusion",
          "malignancy", "consolidation",
          "atelectasis", "pulmonary_oedema",
          "cardiomegaly", "lymphadenopathy",
          "ascites", "endotracheal_tube"]

SYS = ("You are a radiologist extracting "
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
    m = re.search(r"\{.*\}", s.strip(),
                  flags=re.S)
    if not m:
        return None
    for cand in [m.group(0),
                 m.group(0).replace("'", '"')]:
        try:
            return json.loads(cand)
        except Exception:
            continue
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

done = set()
prev = []
if os.path.exists(RAW):
    p = pd.read_csv(RAW)
    done = set(p["hadm_id"].tolist())
    prev = p.to_dict("records")
    print("resuming, already done:", len(done))

todo = nt[~nt["idx_hadm"].isin(done)]
print("to process:", len(todo))
if len(todo) == 0:
    print("nothing to do")

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16,
    device_map="auto")
mdl.eval()
gcfg = mdl.generation_config
gcfg.temperature = None
gcfg.top_p = None
gcfg.top_k = None
print("model ready on",
      next(mdl.parameters()).device)

rows = list(prev)
t0 = time.time()
for i, (_, r) in enumerate(todo.iterrows()):
    txt = str(r["text"])[:6000]
    msgs = [{"role": "system", "content": SYS},
            {"role": "user",
             "content": TMPL.format(
                 report=txt)}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True)
    enc = {k: v.to(mdl.device)
           for k, v in enc.items()}
    with torch.no_grad():
        out = mdl.generate(
            **enc, max_new_tokens=MAXNEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id)
    gen = tok.decode(
        out[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=True)
    js = parse(gen)
    rec = {"hadm_id": int(r["idx_hadm"]),
           "parsed": js is not None}
    for f in FIELDS:
        rec["llm_" + f] = (
            str(js.get(f, "")).lower().strip()
            if js is not None else "")
    if js is None:
        rec["raw"] = gen[:300]
    rows.append(rec)

    if (i + 1) % SAVE_EVERY == 0 or \
            (i + 1) == len(todo):
        pd.DataFrame(rows).to_csv(
            RAW, index=False)
        el = time.time() - t0
        rate = el / (i + 1)
        rem = (len(todo) - i - 1) * rate / 60.0
        print("  %d/%d  %.1fs/note  ~%.0f min"
              " remaining"
              % (i + 1, len(todo), rate, rem),
              flush=True)

t = pd.DataFrame(rows)
t.to_csv(RAW, index=False)
print("")
print("total notes:", len(t))
print("parse rate: %.4f"
      % float(t["parsed"].mean()))

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

fx = pd.read_csv(
    DATA + "/ctpa_comorb_features.csv")
g = t[t["parsed"]].merge(
    fx, on="hadm_id", how="inner")
print("merged with regex features:", len(g))

ar = []
print("")
print("AGREEMENT WITH REGEX (n=%d)" % len(g))
print("  %-18s %-12s %6s %6s %6s %6s %6s"
      % ("llm field", "regex", "agree",
         "kappa", "llm+", "rgx+", "both"))
for lf, rf in MAP.items():
    lc = "llm_" + lf
    if lc not in g.columns or rf not in \
            g.columns:
        print("  %-18s MISSING" % lf)
        continue
    lv = (g[lc].astype(str)
          == "yes").astype(int).values
    rv = g[rf].fillna(0).astype(int).values
    ag = float((lv == rv).mean())
    po = ag
    pe = (lv.mean() * rv.mean()
          + (1 - lv.mean()) * (1 - rv.mean()))
    kap = ((po - pe) / (1 - pe)
           if pe < 1 else np.nan)
    both = int(((lv == 1) & (rv == 1)).sum())
    print("  %-18s %-12s %6.3f %6.3f %6d"
          " %6d %6d"
          % (lf, rf, ag, kap, int(lv.sum()),
             int(rv.sum()), both))
    ar.append({"llm": lf, "regex": rf,
               "agree": ag, "kappa": kap,
               "llm_pos": int(lv.sum()),
               "rgx_pos": int(rv.sum()),
               "both": both})

pd.DataFrame(ar).to_csv(
    OUT + "/agreement_mimic.csv",
    index=False)

print("")
print("PE LOCATION DISTRIBUTION")
print(t["llm_pe_location"].value_counts(
).to_string())
print("")
print("saved", RAW)
