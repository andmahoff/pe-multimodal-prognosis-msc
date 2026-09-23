"""Compares the CTPA increment recomputed from the
saved p_wmean3_ctpa_*.csv files with the value
reported by script 104 in ro4_ctpa_results.csv.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score
import fig_data as fd

lab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")
rep = pd.read_csv(fd.DATA + "/ro4_ctpa_results.csv")
print("--- ro4_ctpa_results.csv ---")
print(rep.to_string())

for o in ["composite_30d", "death_30d"]:
    d = pd.read_csv(fd.DATA + "/p_wmean3_ctpa_%s.csv" % o)
    d = d.merge(lab[fd.KEY + [o]], on=fd.KEY)
    y = d[o].values
    print("\n===", o, " n=%d ev=%d" % (len(d), int(y.sum())))
    for c in ["p_wmean2", "p_wmean3", "p_wmean3_cal"]:
        if c in d.columns:
            print("  %-14s AUC %.4f"
                  % (c, roc_auc_score(y, d[c])))
    print("  raw diff  %.4f"
          % (roc_auc_score(y, d["p_wmean3"])
             - roc_auc_score(y, d["p_wmean2"])))

