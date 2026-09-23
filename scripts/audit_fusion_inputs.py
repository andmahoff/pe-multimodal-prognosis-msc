import os
import pandas as pd

BASE = "."
P2 = os.path.join(BASE, "phase2_mimic")
IW = os.path.join(BASE, "inspect_workspace")

SEARCH_DIRS = [
    P2,
    os.path.join(P2, "results"),
    IW,
    os.path.join(IW, "results"),
    os.path.join(IW, "data", "processed"),
]

KNOWN = [
    os.path.join(P2, "logit_final_pecg.csv"),
    os.path.join(P2, "mimic_pe_mace_cohort.csv"),
    os.path.join(P2, "mimic_pe_ehr_baseline_v5.csv"),
]

KEYS = ["subject_id", "hadm_id", "study_id",
        "ecg_path", "stay_id", "person_id"]


def header_of(path):
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception as e:
        return ["<unreadable: %s>" % type(e).__name__]


def describe(path):
    print("=" * 62)
    print(path)
    if not os.path.exists(path):
        print("  MISSING")
        return
    size = os.path.getsize(path)
    print("  size:", size, "bytes")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("  unreadable:", e)
        return
    print("  shape:", df.shape)
    cols = list(df.columns)
    print("  n_cols:", len(cols))
    print("  first cols:", cols[:20])
    for k in KEYS:
        if k in cols:
            print("   key", k, "nunique:", df[k].nunique())
    for c in cols:
        lc = c.lower()
        if "mace" in lc or lc in ("label", "y", "target"):
            try:
                print("   LABEL", c,
                      "mean:", round(df[c].mean(), 4),
                      "n_pos:", int(df[c].sum()))
            except Exception:
                pass
        if lc.startswith("p_") or "pred" in lc or "prob" in lc:
            try:
                print("   PRED", c,
                      "min:", round(df[c].min(), 4),
                      "max:", round(df[c].max(), 4),
                      "mean:", round(df[c].mean(), 4),
                      "n_null:", int(df[c].isna().sum()))
            except Exception:
                pass


print("### KNOWN CANDIDATES")
for p in KNOWN:
    describe(p)

print()
print("### BROAD SCAN FOR PREDICTION-LIKE FILES")
seen = set()
for d in SEARCH_DIRS:
    if not os.path.isdir(d):
        print("(no dir)", d)
        continue
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"):
            continue
        full = os.path.join(d, f)
        if full in seen:
            continue
        seen.add(full)
        cols = header_of(full)
        low = [c.lower() for c in cols]
        has_pred = any(
            c.startswith("p_") or "pred" in c or "prob" in c
            for c in low
        )
        has_key = any(k in low for k in KEYS)
        if has_pred and has_key:
            print("-" * 58)
            print(full)
            print("  cols:", cols[:20])

