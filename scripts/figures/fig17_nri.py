"""Figure 17 - categorical net reclassification
improvement, bands <5% / 5-15% / >=15% (Ch 4.5)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd
import fig_calib as fc

fs.init()
BANDS = [0.0, 0.05, 0.15, 1.01]
NBOOT = 2000
rng = np.random.RandomState(42)


def band(p):
    return np.digitize(p, BANDS[1:-1])


def nri_parts(y, ba, bb):
    ev = y == 1
    ne = y == 0
    up_e = float(np.sum(bb[ev] > ba[ev]))
    dn_e = float(np.sum(bb[ev] < ba[ev]))
    up_n = float(np.sum(bb[ne] > ba[ne]))
    dn_n = float(np.sum(bb[ne] < ba[ne]))
    n_e = max(float(ev.sum()), 1.0)
    n_n = max(float(ne.sum()), 1.0)
    e = (up_e - dn_e) / n_e
    n = (dn_n - up_n) / n_n
    return e, n, e + n, (up_e, dn_e, up_n, dn_n)


def nri_ci(y, ba, bb, groups):
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    tot = []
    for _ in range(NBOOT):
        pick = rng.choice(uniq, size=len(uniq),
                          replace=True)
        r = np.concatenate([idx[g] for g in pick])
        if y[r].min() == y[r].max():
            continue
        tot.append(nri_parts(y[r], ba[r], bb[r])[2])
    return np.percentile(tot, [2.5, 97.5])


rows, moves = [], []
lab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")

for o in fd.OUTCOMES:
    d = fd.load(o).dropna(subset=["p_wmean2"])
    y = d["y"].values.astype(float)
    g = d["subject_id"].values
    a = fc.platt_oof(y, d["p_ehr"].values, g)
    b = fc.platt_oof(y, d["p_wmean2"].values, g)
    e, n, t, mv = nri_parts(y, band(a), band(b))
    lo, hi = nri_ci(y, band(a), band(b), g)
    rows.append((fd.NICE[o], "EHR $\\rightarrow$ WMEAN2",
                 e, n, t, lo, hi))
    moves.append((fd.NICE[o], "") + mv)

for o in ["composite_30d", "death_30d"]:
    w = pd.read_csv(fd.DATA + "/p_wmean3_ctpa_%s.csv" % o)
    w = w.merge(lab[fd.KEY + [o]], on=fd.KEY)
    w = w.dropna(subset=["p_wmean2", "p_wmean3", o])
    y = w[o].values.astype(float)
    g = w["subject_id"].values
    a = fc.platt_oof(y, w["p_wmean2"].values, g)
    b = fc.platt_oof(y, w["p_wmean3"].values, g)
    e, n, t, mv = nri_parts(y, band(a), band(b))
    lo, hi = nri_ci(y, band(a), band(b), g)
    rows.append((fd.NICE[o],
                 "WMEAN2 $\\rightarrow$ WMEAN3-CTPA",
                 e, n, t, lo, hi))
    moves.append((fd.NICE[o], "") + mv)

res = pd.DataFrame(rows, columns=["outcome", "step",
                                  "events", "nonevents",
                                  "total", "lo", "hi"])
mv = pd.DataFrame(moves, columns=["outcome", "step",
                                  "up_ev", "dn_ev",
                                  "up_ne", "dn_ne"])
res.to_csv("fig17_nri.csv", index=False)
print(res.to_string())
print(mv.to_string())

fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.6))
lbl = ["%s\n%s" % (r.outcome, r.step)
       for r in res.itertuples()]
ypos = np.arange(len(res))[::-1]

ax = axes[0]
fs.panel_tag(ax, "A")
ax.barh(ypos + 0.22, res["events"], height=0.26,
        color=fs.C["ehr"], label="events")
ax.barh(ypos - 0.10, res["nonevents"], height=0.26,
        color="#9ECAE1", label="non-events")
for y0, r in zip(ypos, res.itertuples()):
    c = fs.C["fused"] if r.lo > 0 else "#777777"
    ax.plot([r.lo, r.hi], [y0 - 0.42, y0 - 0.42],
            color=c, lw=1.4)
    ax.scatter([r.total], [y0 - 0.42], s=26,
               marker="D", color=c, zorder=3,
               label="total (95% CI)"
               if y0 == ypos[0] else None)
ax.axvline(0, color="#333333", lw=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels(lbl, fontsize=6.4)
ax.set_xlabel("Net Reclassification Improvement")
ax.grid(axis="x")
ax.set_title("Total and Components", fontsize=9.5,
             pad=8)
ax.legend(ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.13), fontsize=6.8,
          handlelength=1.2, columnspacing=1.2)

ax = axes[1]
fs.panel_tag(ax, "B")
ax.barh(ypos + 0.22, mv["up_ev"], height=0.26,
        color=fs.C["fused"], label="events up")
ax.barh(ypos + 0.22, -mv["dn_ev"], height=0.26,
        color="#F5C0A6", label="events down")
ax.barh(ypos - 0.22, mv["up_ne"], height=0.26,
        color="#BBBBBB", label="non-events up")
ax.barh(ypos - 0.22, -mv["dn_ne"], height=0.26,
        color=fs.C["ehr"], label="non-events down")
ax.axvline(0, color="#333333", lw=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels([])
ax.set_xlabel("Patients Crossing a Risk Band")
ax.grid(axis="x")
ax.set_title("Where the Movement Happens",
             fontsize=9.5, pad=8)
ax.legend(ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.05), fontsize=6.6,
          handlelength=1.2, columnspacing=1.2)

fig.suptitle("Reclassification Across Risk Bands",
             fontsize=10.5, y=1.10)
fig.tight_layout()
fs.save(fig, "fig17_nri", "02_Results")

