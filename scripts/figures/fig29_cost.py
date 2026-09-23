"""Figure 29 - decomposing the gap between the deployed
zero-shot EHR modality and the supervised ceiling (Ch 4.9).
The learner is held constant."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
LEARNER = "gb"
c = pd.read_csv(fd.DATA + "/ceiling_results.csv")

OUTS = [o for o in ["death_30d", "death_30d_inhosp",
                    "composite_30d", "cv_first"]
        if o in set(c["outcome"])]
rows = []
for o in OUTS:
    s = c[(c["outcome"] == o)
          & (c["learner"] == LEARNER)]
    z = float(s["zeroshot"].iloc[0])
    t = float(s[s["set"] == "transferable"]["auc"].iloc[0])
    a = float(s[s["set"] == "all"]["auc"].iloc[0])
    rows.append((o, z, t, a, t - z, a - t, a - z))
res = pd.DataFrame(rows, columns=["outcome", "zeroshot",
                                  "transferable", "all",
                                  "transfer_cost",
                                  "feature_cost",
                                  "total"])
res.to_csv("fig29_cost_decomposition.csv", index=False)
print(res.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.6, 4.2))
w = 0.78 / 3.0
xs = np.arange(len(OUTS))

ax.bar(xs - w, res["zeroshot"], width=w * 0.92,
       color=fs.C["ehr"], edgecolor="white",
       label="zero-shot, transferable features")
ax.bar(xs, res["transferable"], width=w * 0.92,
       color="#9ECAE1", edgecolor="white",
       label="MIMIC-trained, same 28 features")
ax.bar(xs + w, res["all"], width=w * 0.92,
       color=fs.C["fused"], edgecolor="white",
       label="MIMIC-trained, all 103 features")

top = float(max(res["all"].max(), res["zeroshot"].max()))
for i, r in enumerate(res.itertuples()):
    ax.text(xs[i] - w / 2.0, top + 0.022,
            "transfer\n%+.4f" % r.transfer_cost,
            ha="center", va="bottom", fontsize=6.2)
    ax.text(xs[i] + w / 2.0, top + 0.022,
            "features\n%+.4f" % r.feature_cost,
            ha="center", va="bottom", fontsize=6.2)

fs.chance(ax, 0.5)
ax.set_ylim(0.5, top + 0.115)
ax.set_xticks(xs)
ax.set_xticklabels([fd.NICE.get(o, o) for o in OUTS],
                   fontsize=7.6)
ax.set_ylabel("Area Under the ROC Curve")
ax.grid(axis="y")
ax.legend(ncol=1, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), fontsize=7.2,
          handlelength=1.3)
ax.set_title("Transfer Cost Against Feature-"
             "Restriction Cost", fontsize=10.5, pad=52)

fs.save(fig, "fig29_cost", "02_Results")

