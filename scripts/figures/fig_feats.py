"""Rebuilds the 28-feature matrices (features measured
in both INSPECT and MIMIC-IV), because scripts 129 and
138 did not save the matrices they used.

INSPECT vitals and labs come from inspect_expanded_feats
(m7_ = 7-day pre-index window); MIMIC labs from
mimic_expanded_feats, MIMIC vitals from v7, and age plus
the four condition flags from v6.
"""
import glob
import numpy as np
import pandas as pd
import fig_data as fd

VITALS = ["temp", "hr", "sbp", "dbp", "rr"]
LABS = ["creatinine", "sodium", "potassium", "bun",
        "glucose", "calcium", "bicarb", "chloride",
        "aniongap", "hct", "plt", "wbc", "hgb", "rbc",
        "mchc", "mch", "mcv", "rdw"]
ANALYTES = VITALS + LABS
FLAGS = ["cancer", "heart_failure", "copd", "afib"]

NICEF = {"temp": "Temperature", "hr": "Heart rate",
         "sbp": "Systolic BP", "dbp": "Diastolic BP",
         "rr": "Respiratory rate",
         "creatinine": "Creatinine", "sodium": "Sodium",
         "potassium": "Potassium",
         "bun": "Urea nitrogen", "glucose": "Glucose",
         "calcium": "Calcium", "bicarb": "Bicarbonate",
         "chloride": "Chloride",
         "aniongap": "Anion gap", "hct": "Haematocrit",
         "plt": "Platelets", "wbc": "Leukocytes",
         "hgb": "Haemoglobin", "rbc": "Erythrocytes",
         "mchc": "MCHC", "mch": "MCH", "mcv": "MCV",
         "rdw": "RDW", "age": "Age",
         "cancer": "Cancer",
         "heart_failure": "Heart failure",
         "copd": "COPD", "afib": "Atrial fibrillation"}

INSPECT_ROOT = "./inspect_workspace"


def _find_inspect_extra():
    pats = [INSPECT_ROOT + "/**/final_feature_matrix_v2_labeled.csv",
            INSPECT_ROOT + "/**/final_feature_matrix_v2.csv"]
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if hits:
            print("INSPECT extras from:", hits[0])
            return hits[0]
    print("WARNING: INSPECT age/flag file not found")
    return None


def load_pair(outcome):
    """Return Xs, ys, Xt, yt, cols on a common column
    set. Source = INSPECT, target = MIMIC.

    No competing-risk exclusion is applied here. For
    cv_first, callers must first drop patients with
    death_first == 1, as every other cv_first
    analysis does (export_shap_long.py does this)."""
    # ---------------- source ------------------------
    ins = pd.read_csv(fd.DATA + "/inspect_expanded_feats.csv")
    s = ins[["person_id"]].copy()
    for a in ANALYTES:
        c = "m7_" + a
        s[a] = ins[c] if c in ins.columns else np.nan

    extra = _find_inspect_extra()
    have_extra = False
    if extra is not None:
        e = pd.read_csv(extra)
        keep = ["person_id"] + [c for c in ["age"] + FLAGS
                                if c in e.columns]
        if len(keep) > 1:
            s = s.merge(e[keep].drop_duplicates("person_id"),
                        on="person_id", how="left")
            have_extra = True
            print("INSPECT extras merged:", keep[1:])

    slab = pd.read_csv(fd.DATA + "/inspect_labels_final.csv")
    if outcome not in slab.columns:
        raise ValueError("no %s in INSPECT labels" % outcome)
    s = s.merge(slab[["person_id", outcome]],
                on="person_id", how="inner")

    # ---------------- target ------------------------
    mim = pd.read_csv(fd.DATA + "/mimic_expanded_feats.csv")
    t = mim[fd.KEY].copy()
    for a in LABS:
        c = "mi_" + a
        t[a] = mim[c] if c in mim.columns else np.nan

    v7 = pd.read_csv(fd.P2 + "/mimic_pe_ehr_baseline_v7.csv")
    vcols = fd.KEY + [c for c in
                      ["mean_" + a for a in VITALS]
                      if c in v7.columns]
    v7s = v7[vcols].drop_duplicates(subset=fd.KEY)
    t = t.merge(v7s, on=fd.KEY, how="left")
    for a in VITALS:
        t[a] = t.get("mean_" + a, np.nan)
    t = t.drop(columns=[c for c in t.columns
                        if c.startswith("mean_")])

    v6 = pd.read_csv(fd.P2 + "/mimic_pe_ehr_baseline_v6.csv")
    e6 = fd.KEY + [c for c in ["age_at_admit"] + FLAGS
                   if c in v6.columns]
    v6s = v6[e6].drop_duplicates(subset=fd.KEY)
    t = t.merge(v6s, on=fd.KEY, how="left")
    if "age_at_admit" in t.columns:
        t = t.rename(columns={"age_at_admit": "age"})

    tlab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")
    t = t.merge(tlab[fd.KEY + [outcome]], on=fd.KEY,
                how="inner")

    # ---------------- common columns ----------------
    cols = list(ANALYTES)
    if have_extra:
        for c in ["age"] + FLAGS:
            if c in s.columns and c in t.columns:
                cols.append(c)
    cols = [c for c in cols
            if c in s.columns and c in t.columns
            and s[c].notna().any() and t[c].notna().any()]

    print("%s | features %d | INSPECT n=%d ev=%d | "
          "MIMIC n=%d ev=%d"
          % (outcome, len(cols), len(s),
             int(s[outcome].sum()), len(t),
             int(t[outcome].sum())))
    return (s[cols], s[outcome].values.astype(float),
            t[cols], t[outcome].values.astype(float),
            cols)

