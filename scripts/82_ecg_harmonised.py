import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
KEY = ["subject_id", "hadm_id"]
SEED = 42
FEAT = "bench_feats_logit.csv"
SUM = os.path.join(
    FD, "harmonised_ecg_summary.csv"
)


def norm_path(p):
    s = str(p).rstrip("/")
    parts = s.split("/")
    return "/".join(parts[-4:])


print("=" * 66)
print("1. LOAD ECG FEATURES")
print("=" * 66)
X = pd.read_csv(os.path.join(P2, FEAT))
print("  raw:", X.shape)
print("  sample paths:")
for v in X["ecg_path"].head(3):
    print("   ", v)

X["_k"] = X["ecg_path"].map(norm_path)
print("  sample keys:")
for v in X["_k"].head(3):
    print("   ", v)
print("  unique keys:", X["_k"].nunique())
X = X.drop_duplicates("_k")
print("  after dedup:", X.shape)

print("\n" + "=" * 66)
print("2. ATTACH KEYS + LABELS")
print("=" * 66)
coh = pd.read_csv(
    os.path.join(
        P2, "mimic_pe_mace_cohort.csv"
    )
)[["subject_id", "hadm_id", "ecg_path"]]
print("  cohort sample paths:")
for v in coh["ecg_path"].head(3):
    print("   ", v)
coh["_k"] = coh["ecg_path"].map(norm_path)
coh = coh.drop_duplicates("_k")
print("  cohort unique keys:", len(coh))

d = X.merge(
    coh[["_k", "subject_id", "hadm_id"]],
    on="_k", how="inner"
)
print("  matched:", len(d))
print("  admissions:",
      d.drop_duplicates(KEY).shape[0])

lab = pd.read_csv(
    os.path.join(
        FD, "mimic_labels_harmonised.csv"
    )
)
d = d.merge(lab, on=KEY, how="inner")
d = d.reset_index(drop=True)
print("  after label merge:", len(d))

DROP = set(KEY + [
    "ecg_path", "_k", "ecg_study_id",
    "mace_30d_label", "mace_30d",
    "p_ecg", "fold", "dod_days",
    "inhosp_days", "cv_days",
    "death_days", "death_30d",
    "death_30d_inhosp", "cv_30d",
    "death_first", "cv_first",
    "composite_30d", "composite_inhosp",
])
FEATS = [
    c for c in d.columns
    if c not in DROP
    and pd.api.types.is_numeric_dtype(d[c])
]
print("  n features:", len(FEATS))

TASKS = [
    ("cv_first",
     (d["death_first"] == 0).values,
     "CV only"),
    ("composite_30d",
     np.ones(len(d), bool), "composite"),
    ("death_30d",
     np.ones(len(d), bool), "death (dod)"),
    ("death_30d_inhosp",
     np.ones(len(d), bool),
     "death (in-hosp)"),
]

print("\n" + "=" * 66)
print("3. TRAIN PER LABEL")
print("=" * 66)
rows = []
for lab_col, mask, nm in TASKS:
    sub = d[mask].reset_index(drop=True)
    y = sub[lab_col].values.astype(int)
    g = sub["subject_id"].values
    Xf = np.nan_to_num(
        sub[FEATS].values.astype(float),
        nan=0.0
    )

    skf = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SEED
    )
    oof = np.zeros(len(sub))
    for tr, te in skf.split(Xf, y, groups=g):
        sc = StandardScaler()
        a = sc.fit_transform(Xf[tr])
        b = sc.transform(Xf[te])
        clf = LogisticRegression(
            max_iter=5000, C=1.0,
            class_weight="balanced"
        )
        clf.fit(a, y[tr])
        oof[te] = clf.predict_proba(b)[:, 1]

    sub["p_ecg"] = oof
    row_auc = roc_auc_score(y, oof)

    adm = sub.groupby(KEY).agg(
        y=(lab_col, "max"),
        p=("p_ecg", "mean"),
    ).reset_index()
    a_auc = roc_auc_score(adm["y"], adm["p"])
    a_ap = average_precision_score(
        adm["y"], adm["p"]
    )

    print("\n  %-16s rows=%d ev=%d | "
          "adm=%d ev=%d"
          % (nm, len(sub), int(y.sum()),
             len(adm), int(adm["y"].sum())))
    print("    row-level AUC  %.4f" % row_auc)
    print("    admission AUC  %.4f  AP %.4f"
          % (a_auc, a_ap))

    out = adm.rename(
        columns={"p": "p_ecg"}
    )[KEY + ["p_ecg"]]
    pth = os.path.join(
        FD, "p_ecg_harm_%s.csv" % lab_col
    )
    out.to_csv(pth, index=False)
    print("    saved", pth)

    rows.append({
        "label": lab_col, "name": nm,
        "n_adm": len(adm),
        "events": int(adm["y"].sum()),
        "auc_row": row_auc,
        "auc_adm": a_auc, "ap_adm": a_ap,
    })

r = pd.DataFrame(rows)
print("\n" + "=" * 66)
print("SUMMARY")
print("=" * 66)
print(r.to_string(index=False))
r.to_csv(SUM, index=False)
print("Saved", SUM)

