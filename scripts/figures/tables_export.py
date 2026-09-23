"""Writes the summary tables (table03, table07, table08,
table09, table11 and table13 CSV files) from saved
results. Only the route counts in table11 are
recomputed.
"""
import numpy as np
import pandas as pd
import fig_data as fd

OUT = "./fusion_workspace/tables"

# ---- Table 3: outcome definitions and prevalences ---
ml = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")
il = pd.read_csv(fd.DATA + "/inspect_labels_final.csv")
ecg = pd.read_csv(fd.DATA
                  + "/p_ecg_harm_composite_30d.csv")
m = ml[ml["hadm_id"].isin(set(ecg["hadm_id"]))]
nm = len(m)
ncv = nm - int((m["death_first"] == 1).sum())

t3 = []
for nice, col, den in [
        ("Composite (30-day)", "composite_30d", nm),
        ("30-day death (dod)", "death_30d", nm),
        ("In-hospital death", "death_30d_inhosp", nm),
        ("CV readmission (first)", "cv_first", ncv)]:
    ev = int(m[col].sum())
    t3.append({"Outcome": nice, "MIMIC events": ev,
               "MIMIC n": den,
               "MIMIC %": round(100.0 * ev / den, 2)})
for r, col in zip(t3, ["composite_30d", "death_30d",
                       None, "cv_first"]):
    if col and col in il.columns:
        ev = int(il[col].sum())
        r["INSPECT events"] = ev
        r["INSPECT n"] = len(il)
        r["INSPECT %"] = round(100.0 * ev / len(il), 2)
    else:
        r["INSPECT events"] = ""
        r["INSPECT n"] = ""
        r["INSPECT %"] = ""
pd.DataFrame(t3).to_csv(OUT + "/table03_outcomes.csv",
                        index=False)
print(pd.DataFrame(t3).to_string(index=False))

# ---- Table 7: unimodal performance ------------------
r4 = pd.read_csv(fd.DATA + "/ro4_harmonised_results.csv")
t7 = r4.pivot_table(index="model", columns="outcome",
                    values=["auc", "ap"])
t7.round(4).to_csv(OUT + "/table07_unimodal.csv")
print("\n", t7.round(4).to_string())

# ---- Table 8: two-modality fusion with CIs ----------
inc = pd.read_csv("fig11_increments.csv")
inc["ΔAUC [95% CI]"] = inc.apply(
    lambda r: "%+.4f [%+.4f, %+.4f]"
    % (r["diff"], r["lo"], r["hi"]), axis=1)
inc["Significant"] = np.where(
    (inc["lo"] > 0) | (inc["hi"] < 0), "yes", "no")
inc[["outcome", "comp", "ev", "ΔAUC [95% CI]",
     "Significant"]].to_csv(
    OUT + "/table08_fusion_increments.csv", index=False)

# ---- Table 9: three-modality fusion -----------------
cp = pd.read_csv(fd.DATA + "/ctpa_fusion_pres.csv")
cp.round(4).to_csv(OUT + "/table09_three_modality.csv",
                   index=False)

# ---- Table 11: outcome decomposition by route -------
w2 = pd.read_csv(fd.DATA
                 + "/p_wmean2_composite_30d.csv")
d = ml.merge(w2, on=fd.KEY)
death = (d["death_first"] == 1).values
cv = (d["cv_first"] == 1).values
dd = pd.to_numeric(d["dod_days"], errors="coerce").values
cd = pd.to_numeric(d["cv_days"], errors="coerce").values
day = np.where(death, dd, np.where(cv, cd, np.nan))
t11 = []
for nice, mask in [("via all-cause death", death),
                   ("via CV readmission", cv)]:
    v = day[mask]
    v = v[~np.isnan(v)]
    t11.append({
        "Route": nice, "Events": int(mask.sum()),
        "% of composite":
            round(100.0 * mask.sum()
                  / float((death | cv).sum()), 1),
        "Median day": float(np.median(v)),
        "IQR": "%.0f-%.0f" % (np.percentile(v, 25),
                              np.percentile(v, 75)),
        "0-7 d": int((v <= 7).sum()),
        "8-14 d": int(((v > 7) & (v <= 14)).sum()),
        "15-30 d": int((v > 14).sum())})
pd.DataFrame(t11).to_csv(
    OUT + "/table11_route_decomposition.csv",
    index=False)
print("\n", pd.DataFrame(t11).to_string(index=False))

# ---- Table 13: supervision ladder -------------------
sf = pd.read_csv(fd.DATA + "/supervised_fusion.csv")
sf["Gain [95% CI]"] = sf.apply(
    lambda r: "%+.4f [%+.4f, %+.4f]"
    % (r["gain"], r["lo"], r["hi"]), axis=1)
sf["Weights (EHR/ECG/CTPA)"] = sf.apply(
    lambda r: "%.2f / %.2f / %.2f"
    % (r["w_ehr"], r["w_ecg"], r["w_ctpa"]), axis=1)
sf[["outcome", "ehr", "ehr_auc", "wmean3", "ap",
    "Gain [95% CI]",
    "Weights (EHR/ECG/CTPA)"]].round(4).to_csv(
    OUT + "/table13_supervision_ladder.csv",
    index=False)

print("\nwritten to", OUT)

