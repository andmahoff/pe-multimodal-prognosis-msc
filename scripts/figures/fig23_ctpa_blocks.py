"""Figure 23 - attribution by feature block within the
CTPA modality (Chapter 4.7). Writes Figures 23a, 23b
and 23c.

Uses LogisticRegressionCV to match the CTPA modality's
own regularisation, with the penalty re-selected inside
each training fold. Features come from an explicit list
of allowed columns (WHITE), never from exclusion.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import fig_style as fs
import fig_data as fd

fs.init()
SEEDS = [42, 7, 13]

PE_DESC = ["pe_pos", "pe_neg", "saddle", "central",
           "lobar", "segmental", "subseg", "bilateral",
           "rv_strain", "septal_bow", "reflux",
           "mpa_enlarge", "infarct"]
INCID = ["effusion", "malignancy", "consolid",
         "atelect", "edema", "cardiomeg", "adenopathy"]
META = ["txt_len", "n_sent"]

raw = pd.read_csv(fd.DATA + "/ctpa_comorb_features.csv")
cm = [c for c in raw.columns if c.startswith("cm_")]
dv = [c for c in raw.columns if c.startswith("dv_")]
WHITE = [c for c in PE_DESC + INCID + META + cm + dv
         if c in raw.columns]
RED = [c for c in WHITE if c not in PE_DESC]
print("whitelist %d | reduced %d" % (len(WHITE),
                                     len(RED)))

BLOCK = {}
for c in WHITE:
    if c in PE_DESC:
        BLOCK[c] = "PE descriptors"
    elif c in INCID:
        BLOCK[c] = "Incidental findings"
    elif c in META:
        BLOCK[c] = "Report metadata"
    elif c.startswith("cm_"):
        BLOCK[c] = "Comorbidity"
    else:
        BLOCK[c] = "Support devices"


def make_cv():
    return LogisticRegressionCV(Cs=10, cv=5,
                                scoring="roc_auc",
                                penalty="l2",
                                max_iter=5000,
                                n_jobs=-1)


def oof(d, y, g, cols, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                              random_state=seed)
    Xc = d[cols]
    for tr, te in cv.split(Xc, y, g):
        i2 = SimpleImputer(
            strategy="median").fit(Xc.iloc[tr])
        s2 = StandardScaler().fit(
            i2.transform(Xc.iloc[tr]))
        m = make_cv()
        m.fit(s2.transform(i2.transform(Xc.iloc[tr])),
              y[tr])
        p[te] = m.predict_proba(
            s2.transform(i2.transform(
                Xc.iloc[te])))[:, 1]
    return roc_auc_score(y, p)


pres = pd.read_csv(fd.DATA
                   + "/p_ctpa_pres_base_cm_dv_death_30d.csv")
keep = set(pres["hadm_id"])
lab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")

OUTS = ["death_30d", "composite_30d",
        "death_30d_inhosp"]
SHORT = {"death_30d": "30-day\ndeath",
         "composite_30d": "Composite\n(30-day)",
         "death_30d_inhosp": "In-hospital\ndeath"}
shares, ablate, arows = {}, {}, []

for o in OUTS:
    d = raw[raw["hadm_id"].isin(keep)][fd.KEY + WHITE]
    d = d.merge(lab[fd.KEY + [o]], on=fd.KEY).dropna(
        subset=[o])
    y = d[o].values.astype(float)
    g = d["subject_id"].values
    X = d[WHITE]

    imp = SimpleImputer(strategy="median").fit(X)
    sc = StandardScaler().fit(imp.transform(X))
    z = sc.transform(imp.transform(X))
    lr = make_cv().fit(z, y)
    shap = np.abs(lr.coef_[0]) * np.mean(np.abs(z), 0)

    t = pd.DataFrame({"feature": WHITE, "shap": shap,
                      "beta": lr.coef_[0]})
    t["block"] = t["feature"].map(BLOCK)
    b = t.groupby("block").agg(
        total=("shap", "sum"), per_feat=("shap", "mean"),
        beta=("beta", "mean"),
        n=("feature", "size")).reset_index()
    b["share"] = b["total"] / b["total"].sum()
    shares[o] = b.sort_values("share", ascending=False)
    t.sort_values("shap", ascending=False).to_csv(
        "fig23_ctpa_features_%s.csv" % o, index=False)

    ds = []
    for s in SEEDS:
        af = oof(d, y, g, WHITE, s)
        ar = oof(d, y, g, RED, s)
        ds.append(ar - af)
        arows.append((o, s, af, ar, ar - af))
        print("  %-18s seed %-3d full %.4f  minus PE "
              "%.4f  delta %+.4f" % (o, s, af, ar,
                                     ar - af))
    ablate[o] = ds
    print("\n%s n=%d ev=%d" % (o, len(d), int(y.sum())))
    print(shares[o][["block", "n", "share", "per_feat",
                     "beta"]].to_string(index=False))

pd.DataFrame(arows, columns=["outcome", "seed", "full",
                             "reduced", "delta"]).to_csv(
    "fig23_ablation.csv", index=False)

blocks = ["Incidental findings", "PE descriptors",
          "Comorbidity", "Support devices",
          "Report metadata"]
COL = {"Incidental findings": fs.C["fused"],
       "PE descriptors": fs.C["ehr"],
       "Comorbidity": fs.C["ctpa"],
       "Support devices": fs.C["cxr"],
       "Report metadata": "#BBBBBB"}
xs = np.arange(len(OUTS))


def val(o, bl, col):
    m = shares[o]["block"] == bl
    return float(shares[o].loc[m, col].iloc[0]) \
        if m.any() else 0.0


# ---------------- 23a: block share ------------------
fig, ax = plt.subplots(figsize=(6.0, 4.0))
bottom = np.zeros(len(OUTS))
for j, bl in enumerate(blocks):
    vals = np.array([val(o, bl, "share") for o in OUTS])
    ax.bar(xs, vals, bottom=bottom, width=0.58,
           color=COL[bl], edgecolor="white",
           linewidth=0.8, hatch=fs.HATCH[j], zorder=3,
           label=bl)
    for x, v, bt in zip(xs, vals, bottom):
        if v > 0.06:
            ax.text(x, bt + v / 2.0, "%.0f%%" % (100 * v),
                    ha="center", va="center",
                    fontsize=6.4, color="white",
                    fontweight="bold")
    bottom = bottom + vals
ax.set_xticks(xs)
ax.set_xticklabels([SHORT.get(o, o) for o in OUTS],
                   fontsize=7.4)
ax.set_ylabel("Share of Total Mean |SHAP|")
ax.set_ylim(0, 1)
fs.grid(ax, axis="y")
ax.legend(ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), fontsize=7.0,
          handlelength=1.3, columnspacing=1.4)
ax.set_title("Attribution Share by Feature Block",
             fontsize=10.5, pad=48)
fs.save(fig, "fig23a_block_share", "02_Results")

# ---------------- 23b: per feature ------------------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
w = 0.80 / len(blocks)
for j, bl in enumerate(blocks):
    vals = [val(o, bl, "per_feat") for o in OUTS]
    ax.bar(xs - 0.40 + w * (j + 0.5), vals,
           width=w * 0.90, color=COL[bl],
           edgecolor="white", linewidth=0.6,
           hatch=fs.HATCH[j], zorder=3, label=bl)
ax.set_xticks(xs)
ax.set_xticklabels([SHORT.get(o, o) for o in OUTS],
                   fontsize=7.4)
ax.set_ylabel("Mean |SHAP| Per Feature")
ax.set_xlim(-0.55, len(OUTS) - 0.45)
fs.grid(ax, axis="y")
ax.legend(ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), fontsize=7.0,
          handlelength=1.3, columnspacing=1.4)
ax.set_title("Attribution Per Feature", fontsize=10.5,
             pad=48)
fs.save(fig, "fig23b_per_feature", "02_Results")

# ---------------- 23c: ablation ---------------------
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ypos = np.arange(len(OUTS))[::-1]
for y0, o in zip(ypos, OUTS):
    ds = ablate[o]
    ax.plot([min(ds), max(ds)], [y0, y0],
            color="#CFCFCF", lw=1.8, zorder=2)
    ax.scatter(ds, [y0] * len(ds), s=26,
               facecolors="none", edgecolors=fs.C["ehr"],
               linewidths=1.2, zorder=4,
               label="individual CV seed"
               if y0 == ypos[0] else None)
    ax.scatter([np.mean(ds)], [y0], s=64, marker="D",
               color=fs.C["fused"], zorder=5,
               edgecolors="white", linewidths=0.7,
               label="mean of three seeds"
               if y0 == ypos[0] else None)
    ax.text(np.mean(ds), y0 + 0.26, "%+.4f"
            % np.mean(ds), fontsize=6.6, ha="center")
ax.axvline(0, color="#333333", lw=1.0, zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels([SHORT.get(o, o) for o in OUTS],
                   fontsize=7.4)
ax.set_xlim(-0.05, 0.05)
ax.set_ylim(-0.6, len(OUTS) - 0.25)
ax.set_xlabel("Change in AUC on Removing the "
              "PE-Descriptor Block")
fs.grid(ax)
ax.legend(ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), fontsize=7.2,
          handlelength=1.2, columnspacing=1.8)
ax.set_title("Ablation Across Three CV Seeds",
             fontsize=10.5, pad=28)
fs.save(fig, "fig23c_ablation", "02_Results")
