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
RAW = OUT + "/mimic_burden_raw.csv"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
POST_H = 24
MAXNEW = 350
SAVE_EVERY = 100

YN = ["space_occupying", "any_effusion",
      "airspace_disease", "volume_loss",
      "congestion", "chronic_lung",
      "nodal_enlargement", "organ_chronic",
      "cachexia_frailty", "support_device",
      "pe_present"]
NUM = ["pe_extent", "acute_severity",
       "chronic_burden"]

SYS = ("You are a radiologist assessing "
       "overall disease burden from a CT "
       "pulmonary angiogram report, for the "
       "purpose of predicting short-term "
       "mortality risk. Answer ONLY with a "
       "JSON object. No explanation, no "
       "markdown.")

TMPL = """Assess this CT pulmonary angiogram report.

Answer "yes" or "no" for each of these.
Answer "yes" if ANY qualifying feature is
described anywhere in the report, including
incidental findings. Answer "no" if absent,
negated, or not mentioned.

space_occupying: any mass, nodule, tumour,
  metastasis or space-occupying lesion in the
  chest or upper abdomen, whatever the cause
any_effusion: any pleural effusion, pericardial
  effusion, or abdominal free fluid
airspace_disease: consolidation, pneumonia,
  infiltrate, ground-glass opacity
volume_loss: atelectasis or lobar collapse
congestion: pulmonary oedema, vascular
  congestion, or cardiac enlargement
chronic_lung: emphysema, fibrosis,
  bronchiectasis, interstitial lung disease
nodal_enlargement: any enlarged, prominent or
  pathological lymph nodes
organ_chronic: cirrhosis, atrophic or scarred
  kidney, aortic calcification or aneurysm,
  vertebral fracture, prior surgery
cachexia_frailty: muscle wasting, sarcopenia,
  cachexia, or very low body habitus
support_device: any endotracheal tube,
  tracheostomy, central line, PICC, pacemaker,
  chest tube, feeding tube, IVC filter or
  sternotomy wires
pe_present: pulmonary embolism is present

Also give three integers:
pe_extent: 0 none, 1 subsegmental only,
  2 segmental, 3 lobar, 4 central or saddle
acute_severity: overall acute illness severity
  visible on this scan, 0 none, 1 mild,
  2 moderate, 3 severe
chronic_burden: overall chronic disease burden
  visible on this scan, 0 none, 1 mild,
  2 moderate, 3 severe

REPORT:
{report}

JSON:"""


def parse(s):
    m = re.search(r"\{.*\}", s.strip(),
                  flags=re.S)
    if not m:
        return None
    for c in [m.group(0),
              m.group(0).replace("'", '"')]:
        try:
            return json.loads(c)
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
    print("resuming, done:", len(done))
todo = nt[~nt["idx_hadm"].isin(done)]
print("to process:", len(todo))

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16,
    device_map="auto")
mdl.eval()
g = mdl.generation_config
g.temperature = None
g.top_p = None
g.top_k = None
print("ready on",
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
        o = mdl.generate(
            **enc, max_new_tokens=MAXNEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id)
    gen = tok.decode(
        o[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=True)
    js = parse(gen)
    rec = {"hadm_id": int(r["idx_hadm"]),
           "parsed": js is not None}
    for f in YN:
        rec["bu_" + f] = (
            str(js.get(f, "")).lower().strip()
            if js is not None else "")
    for f in NUM:
        v = np.nan
        if js is not None:
            try:
                v = float(js.get(f, np.nan))
            except Exception:
                v = np.nan
        rec["bu_" + f] = v
    rows.append(rec)

    if (i + 1) % SAVE_EVERY == 0 or \
            (i + 1) == len(todo):
        pd.DataFrame(rows).to_csv(
            RAW, index=False)
        el = time.time() - t0
        rate = el / (i + 1)
        print("  %d/%d  %.1fs/note  ~%.0f min"
              " left"
              % (i + 1, len(todo), rate,
                 (len(todo) - i - 1)
                 * rate / 60.0), flush=True)

t = pd.DataFrame(rows)
t.to_csv(RAW, index=False)
print("")
print("notes:", len(t), " parse rate %.4f"
      % float(t["parsed"].mean()))

print("")
print("PREVALENCE / DISTRIBUTION")
for f in YN:
    c = "bu_" + f
    v = (t[c].astype(str).str.strip()
         == "yes").astype(int)
    print("  %-20s %.3f" % (f, v.mean()))
for f in NUM:
    c = "bu_" + f
    print("  %-20s median %.1f  na %d"
          % (f, float(t[c].median()),
             int(t[c].isna().sum())))
    print("      dist:",
          t[c].value_counts().sort_index(
          ).to_dict())
print("")
print("saved", RAW)
