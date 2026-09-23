"""Figure 22 - which named SCP-ECG statements drive the
ECG modality (Chapter 4.7). Writes Figures 22a and 22b, and
Table 12.
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import fig_style as fs
import fig_data as fd

warnings.filterwarnings("ignore")
fs.init()

MLB = os.path.expanduser(
    "~/ecg_ptbxl_benchmarking/output/exp0/data/mlb.pkl")
with open(MLB, "rb") as f:
    mlb = pickle.load(f)
names = [str(c) for c in mlb.classes_]


def norm_path(s):
    p = str(s).rstrip("/").split("/")
    return "/".join(p[-4:])


X = pd.read_csv(fd.P2 + "/bench_feats_logit.csv")
X["key"] = X["ecg_path"].map(norm_path)
X = X.drop_duplicates(subset="key")
feat = [c for c in X.columns if c.startswith("feat_")]

coh = pd.read_csv(fd.P2 + "/mimic_pe_mace_cohort.csv",
                  usecols=["subject_id", "hadm_id",
                           "ecg_path"])
coh["key"] = coh["ecg_path"].map(norm_path)
coh = coh.drop_duplicates(subset="key")

d = coh.merge(X[["key"] + feat], on="key")
adm = d.groupby(["subject_id", "hadm_id"],
                as_index=False)[feat].mean()
print("admissions:", len(adm))

lab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")
SPEC = [("death_30d", "30-Day Death", "fig22a_ecg_death"),
        ("composite_30d", "Composite (30-Day)",
         "fig22b_ecg_composite")]

for o, ttl, stem in SPEC:
    m = adm.merge(lab[fd.KEY + [o]], on=fd.KEY).dropna()
    y = m[o].values.astype(float)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(
                         C=0.003, penalty="l2",
                         max_iter=5000))])
    pipe.fit(m[feat].values, y)
    co = pipe.named_steps["lr"].coef_[0]
    t = pd.DataFrame({"statement": names, "coef": co})
    t = t.reindex(t["coef"].abs()
                  .sort_values(ascending=False).index)
    t = t.head(15)
    t.to_csv("table12_ecg_%s.csv" % o, index=False)
    print("\n%s  n=%d ev=%d" % (o, len(m), int(y.sum())))
    print(t.to_string(index=False))

    tt = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cols = [fs.C["fused"] if v > 0 else fs.C["ehr"]
            for v in tt["coef"]]
    hat = ["" if v > 0 else "///" for v in tt["coef"]]
    ax.barh(np.arange(len(tt)), tt["coef"], color=cols,
            hatch=hat, edgecolor="white", linewidth=0.6,
            zorder=3)
    ax.set_yticks(np.arange(len(tt)))
    ax.set_yticklabels(tt["statement"], fontsize=7.4)
    ax.axvline(0, color="#333333", lw=1.0, zorder=2)
    fs.grid(ax, axis="x")
    lim = float(tt["coef"].abs().max()) * 1.18
    ax.set_xlim(-lim, lim)
    ax.set_xticks(np.round(np.linspace(-lim, lim, 5), 2))
    ax.tick_params(axis="x", labelrotation=90,
                   labelsize=7)
    ax.set_xlabel("Standardised Coefficient", labelpad=6)
    ax.barh([], [], color=fs.C["fused"],
            label="raises predicted risk")
    ax.barh([], [], color=fs.C["ehr"], hatch="///",
            label="lowers predicted risk")
    ax.legend(ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), fontsize=7.2,
              handlelength=1.3, columnspacing=1.6)
    ax.set_title("SCP-ECG Statements: %s" % ttl,
                 fontsize=10.5, pad=28)
    fs.save(fig, stem, "02_Results")

