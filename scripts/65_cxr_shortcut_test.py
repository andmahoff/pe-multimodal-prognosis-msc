import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

P2 = "./phase2_mimic/"
FD = (
    "./"
    "fusion_workspace/data/"
)
KEY = ["subject_id", "hadm_id"]


def auc(y, p):
    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return roc_auc_score(y, p)
    except ValueError:
        return np.nan


print("Loading predictions + metadata...")
c = pd.read_csv(
    os.path.join(FD, "cxr_oof_predictions.csv")
)
meta = pd.concat([
    pd.read_csv(P2 + "train.csv"),
    pd.read_csv(P2 + "val.csv"),
    pd.read_csv(P2 + "test.csv"),
], ignore_index=True)
meta = meta[[
    "dicom_id", "ViewPosition",
    "PerformedProcedureStepDescription",
]].drop_duplicates("dicom_id")

m = c.merge(meta, on="dicom_id", how="left")
print("merged:", m.shape)

desc = m[
    "PerformedProcedureStepDescription"
].fillna("").str.upper()
m["portable"] = desc.str.contains(
    "PORT"
).astype(int)
m["is_ap"] = (
    m["ViewPosition"].fillna("") == "AP"
).astype(int)

print("\nimage-level portable rate: %.3f"
      % m["portable"].mean())
print("image-level AP rate:       %.3f"
      % m["is_ap"].mean())

print("\nViewPosition counts:")
print(m["ViewPosition"].value_counts().head(8))
print("\nProcedure desc counts:")
print(desc.value_counts().head(8))

a = m.groupby(KEY).agg(
    y=("mace_30d_label", "max"),
    p_cxr=("p_cxr", "mean"),
    frac_port=("portable", "mean"),
    frac_ap=("is_ap", "mean"),
    n_img=("p_cxr", "size"),
).reset_index()
print("\nadmissions:", len(a))
print("MACE rate: %.4f" % a["y"].mean())

print("\n" + "=" * 55)
print("ADMISSION-LEVEL AUCs")
print("=" * 55)
print("p_cxr           %.4f" % auc(a["y"], a["p_cxr"]))
print("frac_portable   %.4f" % auc(a["y"], a["frac_port"]))
print("frac_AP         %.4f" % auc(a["y"], a["frac_ap"]))
print("n_images        %.4f" % auc(a["y"], a["n_img"]))

print("\ncorrelations with p_cxr:")
for col in ["frac_port", "frac_ap", "n_img"]:
    print("  %-12s pearson %+.3f  spearman %+.3f"
          % (col,
             a["p_cxr"].corr(a[col]),
             a["p_cxr"].corr(
                 a[col], method="spearman")))

print("\n" + "=" * 55)
print("p_cxr WITHIN PORTABLE STRATA")
print("=" * 55)
for name, sub in [
    ("all-portable (frac=1)",
     a[a["frac_port"] == 1]),
    ("none-portable (frac=0)",
     a[a["frac_port"] == 0]),
    ("mixed", a[(a["frac_port"] > 0)
                & (a["frac_port"] < 1)]),
]:
    print("%-24s n=%-5d mace=%-4d AUC %.4f"
          % (name, len(sub), sub["y"].sum(),
             auc(sub["y"], sub["p_cxr"])))

print("\n" + "=" * 55)
print("p_cxr WITHIN n_images STRATA")
print("=" * 55)
a["nbin"] = pd.qcut(
    a["n_img"], 4, duplicates="drop"
)
for b, sub in a.groupby("nbin"):
    print("%-22s n=%-5d mace=%-4d AUC %.4f"
          % (str(b), len(sub), sub["y"].sum(),
             auc(sub["y"], sub["p_cxr"])))

out = os.path.join(
    FD, "cxr_shortcut_test.csv"
)
a.to_csv(out, index=False)
print("\nSaved", out)

