import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FD = (
    "./"
    "fusion_workspace/data/"
)
OUT_A = FD + "cxr_metadata_ci.csv"
OUT_B = FD + "cxr_strata_ci.csv"
TIME = FD + "cxr_image_timing.csv"
LABF = FD + "mimic_labels_harmonised.csv"

KEY = ["subject_id", "hadm_id"]
SEED = 42
NBOOT = 2000
PCOL = "p_cxr"

STRATA = ["All portable", "Mixed",
          "None portable"]

# prediction file, candidate label names, pretty
JOBS = [
    (
        "p_cxr_harm_composite_30d_fixed.csv",
        ["composite_30d", "mace_30d",
         "composite", "y_composite_30d"],
        "Composite",
    ),
    (
        "p_cxr_harm_death_30d_fixed.csv",
        ["death_30d", "death_30d_dod",
         "y_death_30d", "death"],
        "30-day death",
    ),
    (
        "p_cxr_harm_cv_first_fixed.csv",
        ["cv_first", "cv_first_30d",
         "y_cv_first", "cv_30d"],
        "CV readmission",
    ),
]


def boot_ci(y, p, g, n=NBOOT):
    """Subject-level bootstrap CI for one AUROC."""
    rng = np.random.RandomState(SEED)
    uq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uq}
    vals = []
    for _ in range(n):
        s = rng.choice(uq, len(uq), replace=True)
        t = np.concatenate([idx[u] for u in s])
        if len(np.unique(y[t])) < 2:
            continue
        vals.append(roc_auc_score(y[t], p[t]))
    if len(vals) < 100:
        return np.nan, np.nan
    return tuple(np.percentile(vals, [2.5, 97.5]))


def stratum(f):
    if f >= 0.999:
        return "All portable"
    if f <= 0.001:
        return "None portable"
    return "Mixed"


def find_label(lab, cands):
    for c in cands:
        if c in lab.columns:
            return c
    return None


def build(fn, lc, lab, meta):
    """Merge predictions, labels and metadata."""
    pr = pd.read_csv(FD + fn)
    df = pr.merge(lab[KEY + [lc]], on=KEY,
                  how="inner")
    df = df.merge(meta, on=KEY, how="inner")
    df = df[df[lc].notna()].copy()
    df["stratum"] = df["frac_portable"].apply(
        stratum)
    return df


# ---- acquisition metadata ------------------------
tm = pd.read_csv(TIME)
tm = tm[tm["rel"] == "during"].copy()
tm["port"] = tm["port"].astype(float)
print("during-index images:", len(tm))

meta = (
    tm.groupby(KEY)
    .agg(frac_portable=("port", "mean"),
         n_dur=("port", "size"))
    .reset_index()
)
print("admissions with metadata:", len(meta))
print()

# ---- labels --------------------------------------
lab = pd.read_csv(LABF)
print("label file:", lab.shape)
print("columns:", list(lab.columns))
print()

# ==================================================
# PART A - image model against acquisition context
# ==================================================
print("=" * 52)
print("PART A: image vs acquisition context")
print("=" * 52)

rows = []
for fn, cands, pretty in JOBS:
    lc = find_label(lab, cands)
    if lc is None:
        print("NO label column for", pretty)
        print("  tried:", cands)
        continue

    df = build(fn, lc, lab, meta)
    y = df[lc].astype(float).values
    g = df["subject_id"].values
    print(pretty, "| label:", lc,
          "| n =", len(df),
          "| events =", int(y.sum()))

    srcs = {
        "CXR image model": df[PCOL].values,
        "Film portability":
            df["frac_portable"].values,
        "Total film count":
            df["n_dur"].astype(float).values,
    }
    for name, p in srcs.items():
        a = roc_auc_score(y, p)
        lo, hi = boot_ci(y, p, g)
        rows.append({
            "outcome": pretty,
            "source": name,
            "n": len(df),
            "events": int(y.sum()),
            "auc": round(a, 4),
            "lo": round(lo, 4),
            "hi": round(hi, 4),
        })
        print(f"   {name:18s} {a:.4f} "
              f"[{lo:.4f}, {hi:.4f}]")
    print()

# ==================================================
# PART B - discrimination within portability strata
# ==================================================
print("=" * 52)
print("PART B: within portability strata")
print("=" * 52)

srows = []
for fn, cands, pretty in JOBS:
    lc = find_label(lab, cands)
    if lc is None:
        continue

    df = build(fn, lc, lab, meta)
    print(pretty, "| total n =", len(df))

    for st in STRATA:
        sub = df[df["stratum"] == st]
        if len(sub) == 0:
            print("  ", st, "- empty")
            continue
        y = sub[lc].astype(float).values
        p = sub[PCOL].values
        g = sub["subject_id"].values
        if len(np.unique(y)) < 2:
            print(f"   {st:15s} n={len(sub):4d}"
                  "  constant outcome, skipped")
            continue
        a = roc_auc_score(y, p)
        lo, hi = boot_ci(y, p, g)
        srows.append({
            "outcome": pretty,
            "stratum": st,
            "n": len(sub),
            "events": int(y.sum()),
            "auc": round(a, 4),
            "lo": round(lo, 4),
            "hi": round(hi, 4),
        })
        print(f"   {st:15s} n={len(sub):4d} "
              f"ev={int(y.sum()):3d} {a:.4f} "
              f"[{lo:.4f}, {hi:.4f}]")
    print()

# ---- write ---------------------------------------
if rows:
    pd.DataFrame(rows).to_csv(OUT_A, index=False)
    print("saved:", OUT_A)
else:
    print("PART A empty - check label names")

if srows:
    pd.DataFrame(srows).to_csv(OUT_B, index=False)
    print("saved:", OUT_B)
else:
    print("PART B empty - check strata split")
