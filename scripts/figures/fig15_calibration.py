"""Figure 15 - reliability of the fused predictions
before and after out-of-fold Platt scaling (Ch 4.5)."""
import numpy as np
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd
import fig_calib as fc

fs.init()
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.9))
axes = axes.ravel()
tags = ["A", "B", "C", "D"]


def deciles(y, p, k=10):
    q = np.quantile(p, np.linspace(0, 1, k + 1))
    q[0] -= 1e-9
    xs, ys = [], []
    for i in range(k):
        m = (p > q[i]) & (p <= q[i + 1])
        if m.sum() < 5:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


for i, o in enumerate(fd.OUTCOMES):
    ax = axes[i]
    d = fd.load(o)
    if "p_wmean2" not in d.columns:
        continue
    d = d.dropna(subset=["p_wmean2"])
    y = d["y"].values.astype(float)
    raw = d["p_wmean2"].values
    cal = (d["p_wmean2_cal"].values
           if "p_wmean2_cal" in d.columns
           else fc.platt_oof(y, raw,
                             d["subject_id"].values))

    ax.plot([0, 1], [0, 1], ls="--", lw=0.9,
            color=fs.C["ref"], zorder=1)
    for p, lab, col, mk in [
            (raw, "raw score", "#999999", "s"),
            (cal, "Platt-calibrated", fs.C["fused"],
             "o")]:
        xs, ys = deciles(y, p)
        ax.plot(xs, ys, "-", marker=mk, color=col,
                markersize=3.6, lw=1.4, zorder=3,
                label="%s  Brier %.4f"
                      % (lab, fc.brier(y, p)))

    hi = min(1.0, max(float(cal.max()),
                      float(raw.max())) * 1.05)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.hist(cal, bins=30, range=(0, hi),
            weights=np.full(len(cal),
                            hi * 0.14 / len(cal)),
            bottom=0.0, color=fs.C["fused"], alpha=0.22,
            zorder=0)
    ax.axhline(y.mean(), ls=":", lw=1.0, color="#333333",
               zorder=2)
    fs.grid(ax)

    ax.set_title("%s\nn = %s, %d events"
                 % (fd.NICE[o], format(len(d), ","),
                    int(y.sum())), fontsize=8.6, pad=8)
    ax.legend(loc="upper left", fontsize=6.6)
    fs.panel_tag(ax, tags[i], x=-0.19, y=1.20)
    if i in (2, 3):
        ax.set_xlabel("Predicted Probability "
                      "(Decile Mean)")
    if i in (0, 2):
        ax.set_ylabel("Observed Frequency")

fs.suptitle(fig, "Calibration Before and After Platt "
                 "Scaling", y=1.005)
fig.tight_layout()
fs.save(fig, "fig15_calibration", "02_Results")

