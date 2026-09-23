"""Figure 21 - observed event rate by quartile of the
strongest continuous predictors (Chapter 4.7).
Descriptive: no model is involved.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()

lab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")
d = lab[fd.KEY + ["death_30d", "composite_30d"]].copy()

ex = pd.read_csv(fd.DATA + "/mimic_expanded_feats.csv",
                 usecols=fd.KEY + ["mi_bun"])
d = d.merge(ex, on=fd.KEY, how="left")

v6 = pd.read_csv(fd.P2 + "/mimic_pe_ehr_baseline_v6.csv",
                 usecols=fd.KEY + ["age_at_admit"])
v6 = v6.drop_duplicates(subset=fd.KEY)
d = d.merge(v6, on=fd.KEY, how="left")

v7 = pd.read_csv(fd.P2 + "/mimic_pe_ehr_baseline_v7.csv",
                 usecols=fd.KEY + ["mean_rr", "mean_hr"])
d = d.merge(v7, on=fd.KEY, how="left")

PICKS = [("mi_bun", "Urea Nitrogen", "mmol/L",
          fs.C["ehr"]),
         ("age_at_admit", "Age", "years", fs.C["fused"]),
         ("mean_rr", "Respiratory Rate", "/min",
          fs.C["ctpa"]),
         ("mean_hr", "Heart Rate", "/min", fs.C["cxr"])]
PICKS = [p for p in PICKS if p[0] in d.columns]
print("using:", [p[0] for p in PICKS])

fig, axes = plt.subplots(1, len(PICKS),
                         figsize=(1.85 * len(PICKS) + 0.6,
                                  3.5), sharey=True)
if len(PICKS) == 1:
    axes = [axes]

for ax, (col, nice, unit, bar) in zip(axes, PICKS):
    s = d[[col, "death_30d"]].dropna()
    q = pd.qcut(s[col], 4, labels=False,
                duplicates="drop")
    rates, ns, mids = [], [], []
    for k in sorted(pd.unique(q)):
        m = q == k
        rates.append(float(s.loc[m, "death_30d"].mean()))
        ns.append(int(m.sum()))
        mids.append(float(s.loc[m, col].median()))
    xs = np.arange(len(rates))
    ax.bar(xs, rates, width=0.68, color=bar,
           edgecolor="white", linewidth=0.6)
    for x, r in zip(xs, rates):
        ax.text(x, r + 0.005, "%.3f" % r, ha="center",
                va="bottom", fontsize=6.2)
    base = float(s["death_30d"].mean())
    ax.axhline(base, ls=":", lw=1.1, color="#333333")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%.0f" % m for m in mids],
                       fontsize=6.8)
    ax.set_title("%s\nQ4 / Q1 = %.1f$\\times$"
                 % (nice, rates[-1]
                    / max(rates[0], 1e-9)),
                 fontsize=8.6, pad=8)
    ax.set_xlabel("Quartile Median (%s)" % unit,
                  fontsize=7.4)
    print("%-18s rates %s  n %s"
          % (nice, ["%.4f" % r for r in rates], ns))

axes[0].set_ylabel("Observed 30-Day Mortality")
fig.suptitle("Risk Gradient Across Quartiles of the "
             "Strongest Predictors", fontsize=10.5,
             y=1.04)
fig.tight_layout()
fs.save(fig, "fig21_quartiles", "02_Results")

