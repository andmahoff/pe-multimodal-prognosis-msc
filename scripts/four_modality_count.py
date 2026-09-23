import glob
import os
import pandas as pd

D = "./fusion_workspace/data"
lab = pd.read_csv(D + "/mimic_labels_harmonised.csv")


def ids(pat):
    f = sorted(glob.glob(D + "/" + pat))
    print(" ", pat, "->",
          [os.path.basename(x) for x in f])
    if not f:
        return None
    return set(pd.read_csv(f[0])["hadm_id"])


ehr = ids("p_ehr_harm_composite_30d.csv")
ecg = ids("p_ecg_harm_composite_30d.csv")
ctpa = ids("p_ctpa_pres_*_composite_30d.csv")
cxr = ids("p_cxr_harm_composite_30d_fixed.csv")

for n, s in [("EHR", ehr), ("ECG", ecg),
             ("CTPA", ctpa), ("CXR", cxr)]:
    print(n, len(s) if s else "MISSING")

four = ehr & ecg & ctpa & cxr
print()
print("four-modality admissions:", len(four))
sub = lab[lab["hadm_id"].isin(four)]
for o in ["composite_30d", "death_30d",
          "death_30d_inhosp"]:
    if o in sub.columns:
        print("  ", o, "events", int(sub[o].sum()))

