"""Re-aggregate the CXR predictions to admission level,
matching the timing label on both dicom_id and hadm_id.

Script 87 matched on dicom_id alone, so an image taken
during one admission was also treated as "during" for any
other admission it was linked to. No retraining is needed:
the saved per-image predictions are reused.
"""
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

ROOT = "."
DATA = ROOT + "/fusion_workspace/data"
KEY = ["subject_id", "hadm_id"]

tim = pd.read_csv(DATA + "/cxr_image_timing.csv")
tim = tim[["dicom_id", "hadm_id", "rel"]]
tim = tim.drop_duplicates(["dicom_id", "hadm_id"])
print("timing pairs:", len(tim))

for lab in ["composite_30d", "death_30d", "cv_first"]:
    f = DATA + "/cxr_harm_%s_preds.csv" % lab
    if not os.path.exists(f):
        print("SKIP", lab)
        continue
    P = pd.read_csv(f)
    D = P.merge(tim, on=["dicom_id", "hadm_id"],
                how="left")
    D = D[D["rel"] == "during"]
    h = D.groupby(KEY).agg(
        y=("_y", "max"), p=("p_cxr", "mean")
    ).reset_index()
    old = pd.read_csv(DATA + "/p_cxr_harm_%s.csv" % lab)
    print("\n---", lab, "---")
    print("  old n = %d   new n = %d"
          % (len(old), len(h)))
    print("  events = %d" % int(h["y"].sum()))
    print("  AUROC %.4f   AP %.4f"
          % (roc_auc_score(h["y"], h["p"]),
             average_precision_score(h["y"], h["p"])))
    out = DATA + "/p_cxr_harm_%s_fixed.csv" % lab
    h.rename(columns={"p": "p_cxr"})[
        KEY + ["p_cxr"]].to_csv(out, index=False)
    print("  wrote", os.path.basename(out))