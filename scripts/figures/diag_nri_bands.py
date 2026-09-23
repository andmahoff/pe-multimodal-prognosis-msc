"""Checks why the recomputed NRI is much larger than
script 86's. Prints each model's band distribution
before and after calibration: if the two calibrated
scores place very different fractions in each band,
the difference comes from the score distributions
rather than from reclassification.
"""
import numpy as np
import pandas as pd
import fig_data as fd
import fig_calib as fc

BANDS = [0.0, 0.05, 0.15, 1.01]


def dist(p):
    b = np.digitize(p, BANDS[1:-1])
    n = float(len(b))
    return [float(np.sum(b == k)) / n for k in range(3)]


for o in ["composite_30d", "death_30d_inhosp"]:
    d = fd.load(o).dropna(subset=["p_wmean2"])
    y = d["y"].values.astype(float)
    g = d["subject_id"].values
    print("\n===", o, " prevalence %.4f" % y.mean())
    for nm, col in [("p_ehr", "p_ehr"),
                    ("p_wmean2", "p_wmean2")]:
        raw = d[col].values
        cal = fc.platt_oof(y, raw, g)
        print("  %-9s raw   min %.3f med %.3f max %.3f"
              % (nm, raw.min(), np.median(raw), raw.max()))
        print("  %-9s cal   min %.3f med %.3f max %.3f"
              % (nm, cal.min(), np.median(cal), cal.max()))
        print("  %-9s bands <5%% / 5-15%% / >=15%%  "
              "%.3f  %.3f  %.3f"
              % ((nm,) + tuple(dist(cal))))
    if "p_wmean2_cal" in d.columns:
        print("  saved p_wmean2_cal bands  %.3f %.3f %.3f"
              % tuple(dist(d["p_wmean2_cal"].values)))

