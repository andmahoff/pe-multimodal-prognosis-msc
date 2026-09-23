import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
CXR = os.path.join(
    FD, "cxr_oof_predictions.csv"
)
ECG = os.path.join(
    P2, "logit_final_pecg.csv"
)
V7 = os.path.join(
    P2, "mimic_pe_ehr_baseline_v7.csv"
)
OUT = os.path.join(
    FD, "fusion_cohort_keys.csv"
)

KEY = ["subject_id", "hadm_id"]


def auc(y, p):
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return np.nan


print("=" * 60)
print("PART 1: CXR per-fold verification")
print("=" * 60)
c = pd.read_csv(CXR)
print("images:", len(c))
print("subjects:", c["subject_id"].nunique())
print("admissions:",
      c.drop_duplicates(KEY).shape[0])

rows = []
for f in sorted(c["fold"].unique()):
    d = c[c["fold"] == f]
    ia = auc(d["mace_30d_label"], d["p_cxr"])
    g = d.groupby("subject_id").agg(
        y=("mace_30d_label", "max"),
        p=("p_cxr", "mean"),
    )
    sa = auc(g["y"], g["p"])
    h = d.groupby(KEY).agg(
        y=("mace_30d_label", "max"),
        p=("p_cxr", "mean"),
    )
    aa = auc(h["y"], h["p"])
    rows.append({
        "fold": f, "n_img": len(d),
        "n_subj": len(g), "n_adm": len(h),
        "img_auc": ia, "subj_auc": sa,
        "adm_auc": aa,
    })
r = pd.DataFrame(rows)
print()
print(r.to_string(index=False))
print()
print("mean image AUC      %.4f +/- %.4f"
      % (r["img_auc"].mean(),
         r["img_auc"].std()))
print("mean subject AUC    %.4f +/- %.4f"
      % (r["subj_auc"].mean(),
         r["subj_auc"].std()))
print("mean admission AUC  %.4f +/- %.4f"
      % (r["adm_auc"].mean(),
         r["adm_auc"].std()))

gp = c.groupby(KEY).agg(
    y=("mace_30d_label", "max"),
    p=("p_cxr", "mean"),
).reset_index()
print("\npooled admission AUC %.4f"
      % auc(gp["y"], gp["p"]))
print("  (a pooled AUC above the fold mean "
      "reflects pooling across folds)")

print("\n" + "=" * 60)
print("PART 2: aggregate modalities to admission")
print("=" * 60)

cxr_adm = c.groupby(KEY).agg(
    p_cxr=("p_cxr", "mean"),
    n_cxr=("p_cxr", "size"),
).reset_index()
print("CXR admissions:", len(cxr_adm))

e = pd.read_csv(ECG)
print("ECG rows:", len(e))
ecg_adm = e.groupby(KEY).agg(
    p_ecg=("p_ecg", "mean"),
    p_ecg_max=("p_ecg", "max"),
    n_ecg=("p_ecg", "size"),
).reset_index()
print("ECG admissions:", len(ecg_adm))

v7 = pd.read_csv(V7)
ehr_adm = v7[KEY + ["mace_30d_label"]].copy()
ehr_adm = ehr_adm.drop_duplicates(KEY)
print("EHR admissions:", len(ehr_adm))

print("\n" + "=" * 60)
print("PART 3: coverage")
print("=" * 60)

m = ehr_adm.copy()
m["has_ehr"] = 1
m = m.merge(ecg_adm, on=KEY, how="outer")
m = m.merge(cxr_adm, on=KEY, how="outer")
m["has_ehr"] = m["has_ehr"].fillna(0)
m["has_ecg"] = m["p_ecg"].notna().astype(int)
m["has_cxr"] = m["p_cxr"].notna().astype(int)
m["has_ehr"] = m["has_ehr"].astype(int)
m["n_mod"] = (
    m["has_ehr"] + m["has_ecg"] + m["has_cxr"]
)

lab = pd.concat([
    ehr_adm[KEY + ["mace_30d_label"]],
    e.drop_duplicates(KEY)[
        KEY + ["mace_30d_label"]
    ],
    c.drop_duplicates(KEY)[
        KEY + ["mace_30d_label"]
    ],
]).drop_duplicates(KEY)
m = m.drop(
    columns=["mace_30d_label"], errors="ignore"
)
m = m.merge(lab, on=KEY, how="left")

print("total admissions seen:", len(m))
print()
print(
    m.groupby("n_mod").agg(
        n=("hadm_id", "size"),
        mace=("mace_30d_label", "sum"),
        rate=("mace_30d_label", "mean"),
    ).to_string()
)

print("\nper-modality coverage:")
for k in ["has_ehr", "has_ecg", "has_cxr"]:
    print("  %-8s %d" % (k, m[k].sum()))

print("\npairwise overlaps:")
print("  EHR & ECG %d"
      % ((m["has_ehr"] == 1)
         & (m["has_ecg"] == 1)).sum())
print("  EHR & CXR %d"
      % ((m["has_ehr"] == 1)
         & (m["has_cxr"] == 1)).sum())
print("  ECG & CXR %d"
      % ((m["has_ecg"] == 1)
         & (m["has_cxr"] == 1)).sum())

tri = m[m["n_mod"] == 3]
print("\n" + "=" * 60)
print("THREE-WAY COHORT")
print("=" * 60)
print("admissions: %d" % len(tri))
print("patients:   %d"
      % tri["subject_id"].nunique())
print("MACE events: %d (%.4f)"
      % (tri["mace_30d_label"].sum(),
         tri["mace_30d_label"].mean()))

if len(tri) > 0:
    print("\nunimodal AUC ON THE 3-WAY COHORT:")
    y = tri["mace_30d_label"]
    for col in ["p_ecg", "p_cxr"]:
        print("  %-8s %.4f (AP %.4f)"
              % (col, auc(y, tri[col]),
                 average_precision_score(
                     y, tri[col])))
    print("  (p_ehr not included in this file)")

m.to_csv(OUT, index=False)
print("\nSaved", OUT)

