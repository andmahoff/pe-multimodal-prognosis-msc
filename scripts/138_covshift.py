import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "."
DATA = BASE + "/fusion_workspace/data"
P2 = BASE + "/phase2_mimic"
IW = BASE + "/inspect_workspace/data/processed"
OUT = DATA + "/covshift"
os.makedirs(OUT, exist_ok=True)

TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]

ins = pd.read_csv(
    DATA + "/inspect_expanded_feats.csv")
dem = pd.read_csv(
    IW + "/final_feature_matrix_v2_labeled.csv",
    usecols=["person_id", "age"] + FLAGS)
ins = ins.merge(dem.drop_duplicates("person_id"),
                on="person_id", how="left")

mm = pd.read_csv(
    DATA + "/mimic_expanded_feats.csv")
v7 = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v7.csv"
).drop_duplicates("hadm_id")
vit = {"mean_temp": "temp", "mean_hr": "hr",
       "mean_sbp": "sbp", "mean_dbp": "dbp",
       "mean_rr": "rr"}
keep = ["hadm_id"] + [c for c in vit
                      if c in v7.columns] \
    + [c for c in FLAGS if c in v7.columns]
mm = mm.merge(v7[keep].rename(columns=vit),
              on="hadm_id", how="left")
age = pd.read_csv(
    P2 + "/mimic_pe_ehr_baseline_v6.csv",
    usecols=["hadm_id", "age_at_admit"])
mm = mm.merge(
    age.dropna().drop_duplicates("hadm_id")
    .rename(columns={"age_at_admit": "age"}),
    on="hadm_id", how="left")

NAMES = TA + FLAGS + ["age"]
IC = ["m7_" + a for a in TA] + FLAGS + ["age"]
MC = ["mi_" + a if "mi_" + a in mm.columns
      else a for a in TA] + FLAGS + ["age"]
A = ins[IC].astype(float)
B = mm[MC].astype(float)
A.columns = NAMES
B.columns = NAMES
print("INSPECT", A.shape, " MIMIC", B.shape)

print("")
print("MARGINAL COMPARISON")
print("  %-13s %9s %9s %8s %8s"
      % ("feature", "ins_med", "mim_med",
         "ins_iqr", "mim_iqr"))
mar = []
for c in NAMES:
    ai, bi = A[c].dropna(), B[c].dropna()
    if len(ai) < 50 or len(bi) < 50:
        continue
    qa = float(ai.quantile(.75)
               - ai.quantile(.25))
    qb = float(bi.quantile(.75)
               - bi.quantile(.25))
    print("  %-13s %9.2f %9.2f %8.2f %8.2f"
          % (c, ai.median(), bi.median(),
             qa, qb))
    mar.append({"feature": c,
                "ins_med": float(ai.median()),
                "mim_med": float(bi.median()),
                "ins_iqr": qa, "mim_iqr": qb})
pd.DataFrame(mar).to_csv(
    OUT + "/marginals.csv", index=False)

CA = A.corr(method="spearman")
CB = B.corr(method="spearman")
CA.to_csv(OUT + "/corr_inspect.csv")
CB.to_csv(OUT + "/corr_mimic.csv")

D = (CA - CB).abs()
iu = np.triu_indices(len(NAMES), k=1)
diffs = D.values[iu]
fro = float(np.sqrt(np.nansum(
    (CA.values - CB.values) ** 2)))
print("")
print("CORRELATION STRUCTURE")
print("  Frobenius norm of difference: %.3f"
      % fro)
print("  mean |delta rho|: %.3f"
      % float(np.nanmean(diffs)))
print("  max  |delta rho|: %.3f"
      % float(np.nanmax(diffs)))

va = CA.values[iu]
vb = CB.values[iu]
ok = ~(np.isnan(va) | np.isnan(vb))
print("  correlation of the correlations:"
      " %.3f" % float(
          spearmanr(va[ok], vb[ok])[0]))

pairs = []
for i in range(len(NAMES)):
    for j in range(i + 1, len(NAMES)):
        a_, b_ = CA.iloc[i, j], CB.iloc[i, j]
        if np.isnan(a_) or np.isnan(b_):
            continue
        pairs.append({
            "f1": NAMES[i], "f2": NAMES[j],
            "rho_inspect": a_, "rho_mimic": b_,
            "abs_delta": abs(a_ - b_)})
pt = pd.DataFrame(pairs).sort_values(
    "abs_delta", ascending=False)
pt.to_csv(OUT + "/pair_deltas.csv",
          index=False)

print("")
print("20 LARGEST PAIRWISE DIVERGENCES")
print("  %-13s %-13s %8s %8s %8s"
      % ("f1", "f2", "INSPECT", "MIMIC",
         "delta"))
for _, r in pt.head(20).iterrows():
    print("  %-13s %-13s %+8.3f %+8.3f %8.3f"
          % (r["f1"], r["f2"],
             r["rho_inspect"], r["rho_mimic"],
             r["abs_delta"]))

print("")
print("MOST STABLE PAIRS (|rho| > 0.3 both)")
st = pt[(pt["rho_inspect"].abs() > 0.3)
        & (pt["rho_mimic"].abs() > 0.3)]
st = st.sort_values("abs_delta")
for _, r in st.head(10).iterrows():
    print("  %-13s %-13s %+8.3f %+8.3f %8.3f"
          % (r["f1"], r["f2"],
             r["rho_inspect"], r["rho_mimic"],
             r["abs_delta"]))
print("")
print("saved to", OUT)

