"""Shared data loaders for the results figures.
Recomputes sPESI-6 so no figure depends on the older
five-criterion column saved in rh2_twoway_table.csv.
"""
import pandas as pd

DATA = ("./"
        "fusion_workspace/data")
P2 = "./phase2_mimic"

OUTCOMES = ["composite_30d", "death_30d",
            "death_30d_inhosp", "cv_first"]

NICE = {
    "composite_30d": "Composite (30-day)",
    "death_30d": "30-day death",
    "death_30d_inhosp": "In-hospital death",
    "cv_first": "CV readmission",
}

# The EHR model was fitted for three targets; in-hospital
# death reuses the death_30d model (EHRMAP convention).
EHRMAP = {
    "composite_30d": "composite_30d",
    "death_30d": "death_30d",
    "death_30d_inhosp": "death_30d",
    "cv_first": "cv_first",
}

KEY = ["subject_id", "hadm_id"]


def spesi6():
    """Six-criterion sPESI from the v6 baseline matrix."""
    cols = ["subject_id", "hadm_id", "age_at_admit",
            "mean_hr", "mean_sbp", "mean_spo2",
            "cancer", "heart_failure", "copd"]
    d = pd.read_csv(P2 + "/mimic_pe_ehr_baseline_v6.csv",
                    usecols=cols)
    d = d.drop_duplicates(subset=KEY)
    s = ((d["age_at_admit"] > 80).astype(float)
         + (d["mean_hr"] >= 110).astype(float)
         + (d["mean_sbp"] < 100).astype(float)
         + (d["mean_spo2"] < 90).astype(float)
         + (d["cancer"] > 0).astype(float)
         + ((d["heart_failure"] > 0)
            | (d["copd"] > 0)).astype(float))
    out = d[KEY].copy()
    out["spesi"] = s.values
    return out


def rank01(x):
    r = pd.Series(list(x)).rank(method="average").values
    return (r - 1.0) / (float(len(r)) - 1.0)


def load(outcome, ctpa=False):
    """Admission-level frame: y, p_ehr, p_ecg, spesi,
    p_wmean2 (and p_ctpa / p_wmean3 if requested)."""
    lab = pd.read_csv(DATA + "/mimic_labels_harmonised.csv")
    lab = lab[KEY + [outcome]].rename(
        columns={outcome: "y"})
    ehr = pd.read_csv(DATA + "/p_ehr_harm_%s.csv"
                      % EHRMAP[outcome])
    ecg = pd.read_csv(DATA + "/p_ecg_harm_%s.csv"
                      % outcome)
    df = lab.merge(ehr, on=KEY).merge(ecg, on=KEY)
    df = df.merge(spesi6(), on=KEY)
    try:
        w = pd.read_csv(DATA + "/p_wmean2_%s.csv"
                        % outcome)
        df = df.merge(w, on=KEY, how="left")
    except Exception as e:
        print("  no p_wmean2 for", outcome, e)
    if ctpa:
        c = pd.read_csv(
            DATA + "/p_ctpa_pres_base_cm_dv_%s.csv"
            % outcome)
        df = df.merge(c, on=KEY)
        try:
            w3 = pd.read_csv(
                DATA + "/p_wmean3_ctpa_%s.csv" % outcome)
            w3 = w3[KEY + ["p_wmean3", "p_wmean3_cal"]]
            df = df.merge(w3, on=KEY, how="left")
        except Exception as e:
            print("  no p_wmean3 for", outcome, e)
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    print("%-20s n=%d  ev=%d" %
          (outcome, len(df), int(df["y"].sum())))
    return df

