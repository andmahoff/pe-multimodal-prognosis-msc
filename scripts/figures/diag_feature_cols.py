"""Column discovery for the two figures that must
recompute from scratch: Figure 20 (EHR importance,
source vs target) and Figure 30 (covariance shift).
"""
import pandas as pd
import fig_data as fd

ins = pd.read_csv(fd.DATA + "/inspect_expanded_feats.csv",
                  nrows=5)
print("INSPECT cols (%d):" % len(ins.columns))
print(list(ins.columns))

mim = pd.read_csv(fd.DATA + "/mimic_expanded_feats.csv",
                  nrows=5)
print("\nMIMIC cols (%d):" % len(mim.columns))
print(list(mim.columns))

istems = set(c[3:] for c in ins.columns
             if c.startswith("m7_"))
mstems = set(c[3:] for c in mim.columns
             if c.startswith("mi_"))
print("\nshared stems (%d):" % len(istems & mstems))
print(sorted(istems & mstems))
print("\nINSPECT only:", sorted(istems - mstems))
print("MIMIC only:", sorted(mstems - istems))

lab = pd.read_csv(fd.DATA + "/inspect_labels_final.csv",
                  nrows=5)
print("\nINSPECT label cols:", list(lab.columns))

