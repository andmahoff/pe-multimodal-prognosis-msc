"""Figure 30 - which relationships between features
survive the move between hospitals (Chapter 4.9)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_feats as ff

fs.init()
Xs, ys, Xt, yt, cols = ff.load_pair("death_30d")
cont = [c for c in cols if c not in ff.FLAGS]

cs = Xs[cont].corr(method="spearman")
ct = Xt[cont].corr(method="spearman")
order = ff.VITALS + [c for c in ff.LABS if c in cont]
order = [c for c in order if c in cont]
diff = (cs - ct).loc[order, order]
cs = cs.loc[order, order]
ct = ct.loc[order, order]

fro = float(np.sqrt(np.nansum(diff.values ** 2)))
iu = np.triu_indices(len(order), k=1)
vals = diff.values[iu]
print("Frobenius %.3f | mean |delta| %.4f | max %.4f"
      % (fro, np.nanmean(np.abs(vals)),
         np.nanmax(np.abs(vals))))

pairs = []
for i, j in zip(*iu):
    pairs.append((order[i], order[j], cs.values[i, j],
                  ct.values[i, j], diff.values[i, j]))
pr = pd.DataFrame(pairs, columns=["a", "b", "inspect",
                                  "mimic", "delta"])
pr["absd"] = pr["delta"].abs()
pr = pr.sort_values("absd", ascending=False)
pr.to_csv("fig30_pairs.csv", index=False)
print(pr.head(20).to_string(index=False))

fig = plt.figure(figsize=(9.2, 4.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0],
                      wspace=0.75)

ax = fig.add_subplot(gs[0, 0])
ax.text(-0.34, 1.09, "A", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top")
ax.grid(False)
im = ax.imshow(diff.values, cmap="PuOr", vmin=-0.5,
               vmax=0.5)
lbl = [ff.NICEF.get(c, c) for c in order]
ax.set_xticks(range(len(order)))
ax.set_xticklabels(lbl, rotation=90, fontsize=5.4)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(lbl, fontsize=5.4)
nv = len([c for c in ff.VITALS if c in order])
ax.axhline(nv - 0.5, color="#333333", lw=0.9)
ax.axvline(nv - 0.5, color="#333333", lw=0.9)
cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.04)
cb.set_label("$\\rho$ INSPECT $-$ $\\rho$ MIMIC-IV",
             fontsize=6.8, labelpad=4)
cb.ax.tick_params(labelsize=6)
ax.set_title("Correlation Difference", fontsize=9.5,
             pad=12)

ax = fig.add_subplot(gs[0, 1])
ax.text(-0.62, 1.09, "B", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top")
top = pr.head(14).iloc[::-1]
ypos = np.arange(len(top))
for y0, r in zip(ypos, top.itertuples()):
    flip = (r.inspect * r.mimic) < 0
    ax.plot([r.inspect, r.mimic], [y0, y0],
            color=fs.C["neg"] if flip else "#CCCCCC",
            lw=1.6, zorder=1)
    ax.scatter([r.inspect], [y0], s=24,
               color=fs.C["ehr"], zorder=3)
    ax.scatter([r.mimic], [y0], s=24,
               color=fs.C["fused"], zorder=3)
ax.axvline(0, color="#333333", lw=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels(["%s $-$ %s"
                    % (ff.NICEF.get(r.a, r.a),
                       ff.NICEF.get(r.b, r.b))
                    for r in top.itertuples()],
                   fontsize=6.2)
ax.set_xlabel("Spearman Correlation")
ax.grid(axis="x")
ax.set_title("Largest Divergences", fontsize=9.5,
             pad=12)
ax.scatter([], [], s=24, color=fs.C["ehr"],
           label="INSPECT")
ax.scatter([], [], s=24, color=fs.C["fused"],
           label="MIMIC-IV")
ax.plot([], [], color=fs.C["neg"], lw=1.6,
        label="sign reversal")
ax.legend(ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.04), fontsize=6.6,
          handlelength=1.2, columnspacing=1.2)

fig.suptitle("Covariance Structure Between Institutions",
             fontsize=10.5, y=1.04)
fs.save(fig, "fig30_covariance", "02_Results")

