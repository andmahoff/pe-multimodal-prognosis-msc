"""Figure 16 - decision curve analysis on the
three-modality cohort (Chapter 4.5)."""
import numpy as np
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd
import fig_calib as fc

fs.init()
THR = np.arange(0.01, 0.501, 0.005)
OUTS = ["death_30d", "composite_30d"]
fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.0))

for i, o in enumerate(OUTS):
    ax = axes[i]
    d = fd.load(o, ctpa=True)
    d = d.dropna(subset=["p_wmean2", "p_wmean3"])
    y = d["y"].values.astype(float)
    g = d["subject_id"].values
    prev = y.mean()

    series = [
        ("p_wmean3", "WMEAN3-CTPA", fs.C["fused"]),
        ("p_wmean2", "WMEAN2", "#6BAED6"),
        ("p_ehr", "EHR alone", fs.C["ehr"]),
        ("spesi", "sPESI-6", fs.C["spesi"]),
    ]
    for col, lab, c in series:
        p = fc.platt_oof(y, d[col].values, g)
        nb = [fc.net_benefit(y, p, t) for t in THR]
        ax.plot(THR, nb, color=c, lw=1.5, label=lab)

    nb_all = [prev - (1.0 - prev) * (t / (1.0 - t))
              for t in THR]
    ax.plot(THR, nb_all, color="#999999", lw=1.0,
            ls="--", label="treat all")
    ax.axhline(0, color="#333333", lw=0.9)
    ax.axvspan(0.14, 0.21, color="#000000", alpha=0.05,
               zorder=0)

    ax.set_xlim(0.01, 0.50)
    ax.set_ylim(-0.035, max(0.09, prev * 1.05))
    ax.set_xlabel("Threshold Probability")
    if i == 0:
        ax.set_ylabel("Net Benefit")
    ax.set_title("%s\nn = %s, %d events, prevalence %.3f"
                 % (fd.NICE[o], format(len(d), ","),
                    int(y.sum()), prev), fontsize=8.5,
                 pad=6)
    ax.legend(fontsize=6.8, loc="upper right")
    fs.panel_tag(ax, "A" if i == 0 else "B")

fig.suptitle("Clinical Utility Against sPESI-6 and "
             "Treat-All", fontsize=10.5, y=1.02)
fig.tight_layout()
fs.save(fig, "fig16_decision_curve", "02_Results")

