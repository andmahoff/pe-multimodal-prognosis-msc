"""Figure 20 - feature attribution of the transferred EHR
model at both hospitals (Chapter 4.7).

For a linear model on standardised inputs the exact
mean |SHAP| of feature j is |beta_j| * mean|z_ij|.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import fig_style as fs
import fig_feats as ff

fs.init()
OUTS = ["death_30d", "composite_30d"]
TTL = {"death_30d": "30-Day Death",
       "composite_30d": "Composite (30-Day)"}
store = {}

for o in OUTS:
    Xs, ys, Xt, yt, cols = ff.load_pair(o)
    imp = SimpleImputer(strategy="median").fit(Xs)
    sc = StandardScaler().fit(imp.transform(Xs))
    zs = sc.transform(imp.transform(Xs))
    zt = sc.transform(imp.transform(Xt))

    lr = LogisticRegression(max_iter=5000).fit(zs, ys)
    beta = lr.coef_[0]
    print("  zero-shot MIMIC AUC %.4f"
          % roc_auc_score(yt, lr.decision_function(zt)))

    src = np.abs(beta) * np.mean(np.abs(zs), axis=0)
    tgt = np.abs(beta) * np.mean(np.abs(zt), axis=0)
    rho = spearmanr(src, tgt).correlation

    df = pd.DataFrame({"feature": cols, "beta": beta,
                       "src": src, "tgt": tgt})
    df["r_src"] = df["src"].rank(ascending=False)
    df["r_tgt"] = df["tgt"].rank(ascending=False)
    df["shift"] = df["r_src"] - df["r_tgt"]
    df = df.sort_values("tgt", ascending=False)
    df.to_csv("fig20_importance_%s.csv" % o, index=False)
    store[o] = (df, rho)
    print("  rank rho %.4f" % rho)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 5.6))
for ax, o, tag in zip(axes, OUTS, ["A", "B"]):
    df, rho = store[o]
    d = df.head(18).iloc[::-1]
    ypos = np.arange(len(d))
    for y0, r in zip(ypos, d.itertuples()):
        ax.plot([r.src, r.tgt], [y0, y0], color="#CFCFCF",
                lw=1.8, zorder=2)
    ax.scatter(d["src"], ypos, s=30, color=fs.C["ehr"],
               marker="o", zorder=4,
               edgecolors="white", linewidths=0.6,
               label="INSPECT (source)")
    ax.scatter(d["tgt"], ypos, s=30, color=fs.C["fused"],
               marker="s", zorder=4,
               edgecolors="white", linewidths=0.6,
               label="MIMIC-IV (target)")
    ax.set_yticks(ypos)
    ax.set_yticklabels([ff.NICEF.get(f, f)
                        for f in d["feature"]],
                       fontsize=7)
    ax.set_xlabel(r"Mean $|\phi_j|$  (SD units)")
    ax.set_title(r"%s      $\rho_{s}$ = %.3f"
                 % (TTL.get(o, o), rho), fontsize=9.0,
                 pad=8)
    fs.grid(ax, axis="x")
    fs.panel_tag(ax, tag, x=-0.42, y=1.10)

hand, labs = axes[0].get_legend_handles_labels()
fig.legend(hand, labs, ncol=2, loc="lower center",
           bbox_to_anchor=(0.5, 0.995), fontsize=7.4,
           handlelength=1.2, columnspacing=2.0)
fs.suptitle(fig, "Feature Attribution Transfers "
                 "Between Institutions", y=1.055)
fig.tight_layout()
fs.save(fig, "fig20_ehr_importance", "02_Results")

