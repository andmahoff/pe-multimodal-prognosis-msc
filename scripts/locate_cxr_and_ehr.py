import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

BASE = "."
RES = os.path.join(BASE, "inspect_workspace", "results")

SKIP_DIRS = {
    "fusion_env", "ecg_env", "cxr_env",
    "ecg_bench_env", "raw_mimic_ecgs",
    "node_modules", ".git",
}

EHR_FILES = [
    "mimic_pehr_predictions.csv",
    "mimic_ehr_probabilities.csv",
    "mimic_dann_vrex_finegrained_predictions.csv",
]

TARGETS = [
    ("DANN+VREx", 0.7379),
    ("VREx", 0.7036),
    ("IRM", 0.7009),
    ("XGBoost", 0.6525),
    ("ERM", 0.6239),
]


def find_label(cols):
    for c in cols:
        lc = c.lower()
        if "mace" in lc or "target" in lc:
            return c
    return None


def find_pred(cols):
    for c in cols:
        if c.lower().startswith("p_"):
            return c
    return None


print("### PART 1: IDENTIFY EHR PREDICTION FILES")
for f in EHR_FILES:
    path = os.path.join(RES, f)
    print("=" * 62)
    print(path)
    if not os.path.exists(path):
        print("  MISSING")
        continue
    mt = os.path.getmtime(path)
    print("  mtime:", pd.Timestamp(mt, unit="s"))
    df = pd.read_csv(path)
    print("  shape:", df.shape)
    print("  cols:", list(df.columns))
    lab = find_label(df.columns)
    pred = find_pred(df.columns)
    if lab is None or pred is None:
        print("  cannot identify label/pred cols")
        continue
    print("  n_subjects:", df["subject_id"].nunique())
    print("  n_pos:", int(df[lab].sum()),
          "prev:", round(df[lab].mean(), 4))
    auc = roc_auc_score(df[lab], df[pred])
    ap = average_precision_score(df[lab], df[pred])
    print("  AUC-ROC:", round(auc, 4))
    print("  AUPRC:", round(ap, 4))
    best = min(TARGETS, key=lambda t: abs(t[1] - auc))
    print("  CLOSEST KNOWN MODEL:", best[0],
          "(reported", best[1], ",",
          "diff", round(abs(best[1] - auc), 4), ")")

print()
print("### PART 2: RECURSIVE HUNT FOR CXR OUTPUTS")
hits_csv = []
hits_py = []
hits_ckpt = []

for root, dirs, files in os.walk(BASE):
    dirs[:] = [d for d in dirs
               if d not in SKIP_DIRS
               and not d.startswith(".")]
    depth = root[len(BASE):].count(os.sep)
    if depth > 4:
        dirs[:] = []
        continue
    for f in files:
        lf = f.lower()
        full = os.path.join(root, f)
        is_cxr = ("cxr" in lf or "chex" in lf
                  or "densenet" in lf
                  or "kfold" in lf or "fold" in lf)
        if not is_cxr:
            continue
        if f.endswith(".csv"):
            hits_csv.append(full)
        elif f.endswith(".py"):
            hits_py.append(full)
        elif f.endswith((".pt", ".pth")):
            hits_ckpt.append(full)

print("-- CXR-related CSVs --")
for p in sorted(hits_csv):
    try:
        cols = list(pd.read_csv(p, nrows=0).columns)
    except Exception:
        cols = ["<unreadable>"]
    print(p)
    print("   cols:", cols[:15])

print()
print("-- CXR-related scripts --")
for p in sorted(hits_py):
    print(p)

print()
print("-- CXR-related checkpoints --")
for p in sorted(hits_ckpt):
    sz = os.path.getsize(p) // (1024 * 1024)
    print(p, "(%d MB)" % sz)

