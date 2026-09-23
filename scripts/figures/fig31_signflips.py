"""Figure 31 - three feature pairs whose correlation
reverses sign between institutions (Chapter 4.9)."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import fig_style as fs
import fig_feats as ff

fs.init()
Xs, ys, Xt, yt, cols = ff.load_pair("death_30d")
PAIRS = [("rr", "dbp"), ("rr", "sbp"), ("rr", "calcium")]
PAIRS = [p for p in PAIRS
         if p[0] in cols and p[1] in cols]

fig, axes = plt.subplots(2, len(PAIRS),
                         figsize=(2.5 * len(PAIRS) + 0.5,
                                  4.8), sharex="col")

for j, (a, b) in enumerate(PAIRS):
    for i, (X, nm, col) in enumerate(
            [(Xs, "INSPECT", fs.C["ehr"]),
             (Xt, "MIMIC-IV", fs.C["fused"])]):
        ax = axes[i, j] if len(PAIRS) > 1 else axes[i]
        d = X[[a, b]].dropna()
        lo_a, hi_a = d[a].quantile([0.01, 0.99])
        lo_b, hi_b = d[b].quantile([0.01, 0.99])
        d = d[(d[a] >= lo_a) & (d[a] <= hi_a)
              & (d[b] >= lo_b) & (d[b] <= hi_b)]
        rho = spearmanr(d[a], d[b]).correlation
        ax.scatter(d[a], d[b], s=3, alpha=0.18,
                   color=col, edgecolors="none")
        z = np.polyfit(d[a], d[b], 1)
        gx = np.linspace(d[a].min(), d[a].max(), 20)
        ax.plot(gx, np.polyval(z, gx), color="#222222",
                lw=1.4)
        ax.set_title("%s   $\\rho$ = %+.3f" % (nm, rho),
                     fontsize=8.0, pad=6)
        ax.grid(True, color="#EEEEEE")
        ax.set_ylabel(ff.NICEF.get(b, b), fontsize=7.4)
        if i == 1:
            ax.set_xlabel(ff.NICEF.get(a, a),
                          fontsize=7.4)
        print("%-10s %-10s %-9s rho %+.4f n=%d"
              % (a, b, nm, rho, len(d)))

fig.suptitle("Associations That Reverse Between "
             "Institutions", fontsize=10.5, y=1.03)
fig.tight_layout()
fs.save(fig, "fig31_signflips", "02_Results")

