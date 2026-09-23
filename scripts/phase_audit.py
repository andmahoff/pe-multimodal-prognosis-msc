"""Classify every data artefact and script as Phase 1
or Phase 2, by label column rather than by date.

Phase 1 = superseded composite (mace_30d_label).
Phase 2 = harmonised three-label outcome set.

Writes phase_manifest.csv and prints a summary.
"""
import os
import re
import csv
import glob

ROOT = "./fusion_workspace"
DATA = os.path.join(ROOT, "data")
SCR = os.path.join(ROOT, "scripts")

P1 = ["mace_30d_label", "mace_30d", "y_mace"]
P2 = ["composite_30d", "death_30d", "cv_first",
      "death_30d_inhosp"]


def header(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.readline().strip()
    except OSError:
        return ""


def classify(cols):
    a = any(c in cols for c in P1)
    b = any(c in cols for c in P2)
    if a and b:
        return "MIXED"
    if a:
        return "PHASE1"
    if b:
        return "PHASE2"
    return "unlabelled"


rows = []
files = sorted(glob.glob(os.path.join(DATA, "*.csv")))
print("data files:", len(files))

dmap = {}
for p in files:
    h = header(p)
    cols = [c.strip().strip('"') for c in h.split(",")]
    ph = classify(cols)
    nm = os.path.basename(p)
    dmap[nm] = ph
    rows.append({"kind": "data", "name": nm,
                 "phase": ph,
                 "detail": h[:120]})

scripts = sorted(glob.glob(os.path.join(SCR, "*.py")))
print("scripts:", len(scripts))

pat = re.compile(r"[\w\-.]+\.csv")
for p in scripts:
    try:
        with open(p, "r", errors="replace") as f:
            src = f.read()
    except OSError:
        continue
    refs = sorted(set(pat.findall(src)))
    hits = [dmap.get(r) for r in refs if r in dmap]
    if "PHASE1" in hits and "PHASE2" in hits:
        ph = "MIXED"
    elif "PHASE1" in hits:
        ph = "PHASE1"
    elif "PHASE2" in hits:
        ph = "PHASE2"
    else:
        ph = "unlabelled"
    rows.append({"kind": "script",
                 "name": os.path.basename(p),
                 "phase": ph,
                 "detail": ";".join(refs)[:200]})

out = os.path.join(ROOT, "phase_manifest.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, ["kind", "name", "phase",
                           "detail"])
    w.writeheader()
    w.writerows(rows)

print("\nwrote:", out)
print()
for k in ["data", "script"]:
    print("---", k, "---")
    for ph in ["PHASE1", "MIXED", "PHASE2",
               "unlabelled"]:
        n = [r for r in rows
             if r["kind"] == k and r["phase"] == ph]
        print("  %-11s %3d" % (ph, len(n)))

print("\n=== PHASE 1 DATA FILES ===")
for r in rows:
    if r["kind"] == "data" and r["phase"] == "PHASE1":
        print(" ", r["name"])

print("\n=== MIXED OR PHASE 1 SCRIPTS ===")
for r in rows:
    if r["kind"] == "script" and \
            r["phase"] in ("PHASE1", "MIXED"):
        print("  %-34s %s" % (r["name"], r["phase"]))