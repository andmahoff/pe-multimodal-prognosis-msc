"""Figure 26 - every combination rule tested, ranked
(Chapter 4.8)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
ml = pd.read_csv(fd.DATA + "/meta_learners.csv")
jm = pd.read_csv(fd.DATA + "/joint_mlp.csv")
e3 = pd.read_csv(fd.DATA + "/early3way.csv")

OUTS = ["death_30d", "composite_30d"]
TTL = {"death_30d": "30-Day Death",
       "composite_30d": "Composite (30-Day)"}
fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4))

for ax, o, tag in zip(axes, OUTS, ["A", "B"]):
    s = ml[ml["outcome"] == o]
    ref = float(s["wmean3"].iloc[0])
    rows = [("WMEAN3-CTPA (weighted late)", ref, "late")]
    for r in s.itertuples():
        if str(r.meta).strip().upper() == "WMEAN3":
            continue
        kind = "equal" if str(r.meta).strip().upper() \
            == "MEAN3" else "meta"
        rows.append((str(r.meta), float(r.auc), kind))
    j = jm[jm["outcome"] == o]
    if len(j):
        rows.append(("joint MLP (intermediate)",
                     float(j["joint"].iloc[0]), "joint"))
    ee = e3[e3["outcome"] == o]
    if len(ee):
        rows.append(("early concatenation "
                     "(MIMIC-trained)",
                     float(ee["early3"].max()), "early"))

    df = pd.DataFrame(rows, columns=["model", "auc",
                                     "kind"])
    df = df.sort_values("auc")
    CMAP = {"late": fs.C["fused"], "meta": "#9ECAE1",
            "equal": "#C6DBEF", "joint": fs.C["ctpa"],
            "early": "#CCCCCC"}
    ypos = np.arange(len(df))
    ax.barh(ypos, df["auc"],
            color=[CMAP[k] for k in df["kind"]],
            edgecolor="white", linewidth=0.6,
            hatch=["///" if k == "early" else ""
                   for k in df["kind"]])
    ax.axvline(ref, color=fs.C["fused"], ls="--", lw=1.1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["model"], fontsize=6.6)
    lo = float(df["auc"].min()) - 0.022
    ax.set_xlim(lo, float(df["auc"].max()) + 0.012)
    ax.set_xlabel("Area Under the ROC Curve")
    ax.set_title("%s  (n = 1,703)" % TTL.get(o, o),
                 fontsize=9.5, pad=14)
    ax.grid(axis="x")
    ax.text(-0.42, 1.10, tag, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
    for y0, v in zip(ypos, df["auc"]):
        ax.text(v + 0.0014, y0, "%.4f" % v, fontsize=5.8,
                va="center")
    df.to_csv("table16_architecture_%s.csv" % o,
              index=False)

fig.suptitle("Architectural Sweep of Combination Rules",
             fontsize=10.5, y=1.03)
fig.tight_layout()
fs.save(fig, "fig26_architecture", "02_Results")

