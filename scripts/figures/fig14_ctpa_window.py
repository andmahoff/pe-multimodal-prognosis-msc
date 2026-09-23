"""Figures 14a and 14b - CTPA modality by feature set, in
each acquisition window (Chapter 4.4).

Written as two separate figures so each can be read at
full width.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
df = pd.read_csv(fd.DATA + "/ctpa_window_results.csv")

SPEC = [("full", "Full Acquisition Window",
         "fig14a_ctpa_full_window"),
        ("pres", "Presentation Window",
         "fig14b_ctpa_presentation_window")]
NICES = {"base": "findings only",
         "base_cm": "+ comorbidity",
         "base_cm_dv": "+ support devices"}
PAL = ["#C6DBEF", "#6BAED6", "#08519C"]

sets = list(pd.unique(df["set"]))
outs = [o for o in fd.OUTCOMES if o in set(df["outcome"])]

for win, ttl, stem in SPEC:
    sub = df[df["window"] == win]
    if not len(sub):
        continue
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xs = np.arange(len(outs))
    w = 0.82 / len(sets)
    for j, s in enumerate(sets):
        vals = []
        for o in outs:
            r = sub[(sub["outcome"] == o)
                    & (sub["set"] == s)]
            vals.append(float(r["auc"].iloc[0])
                        if len(r) else np.nan)
        pos = xs - 0.41 + w * (j + 0.5)
        ax.bar(pos, vals, width=w * 0.92,
               color=PAL[j % len(PAL)],
               edgecolor="white", linewidth=0.7,
               hatch=fs.HATCH[j % len(fs.HATCH)],
               zorder=3, label=NICES.get(s, s))
        for x, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(x, v + 0.004, "%.3f" % v,
                        ha="center", va="bottom",
                        fontsize=6.0, rotation=90)

    fs.chance(ax, 0.5)
    fs.grid(ax, axis="y")
    ax.set_ylim(0.5, 0.86)
    ax.set_xticks(xs)
    ax.set_xticklabels([fd.NICE.get(o, o) for o in outs],
                       fontsize=7.6)
    ax.set_xlim(-0.52, len(outs) - 0.48)
    ax.set_ylabel("Area Under the ROC Curve")
    nmax = int(sub["n"].max())
    ax.legend(ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), fontsize=7.4,
              columnspacing=1.4, handlelength=1.3)
    ax.set_title("%s  (n = %s)"
                 % (ttl, format(nmax, ",")),
                 fontsize=10.5, pad=28)
    fs.save(fig, stem, "02_Results")

