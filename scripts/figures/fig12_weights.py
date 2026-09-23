"""Figure 12 - how the fusion weight is split between
the two modalities, and how stable that split is
across training folds (Chapter 4.3).

Each bar is one outcome. The bar is the full weight
budget of 1.0, split between the EHR modality (left)
and the ECG modality (right). Vertical ticks mark
where the split landed in each of the five training
folds.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import fig_style as fs
import fig_data as fd

fs.init()
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def best_w(y, ra, rb):
    best, bw = -1.0, 0.5
    for w in GRID:
        a = roc_auc_score(y, w * ra + (1.0 - w) * rb)
        if a > best:
            best, bw = a, w
    return bw


rows = []
for o in fd.OUTCOMES:
    d = fd.load(o)
    y = d["y"].values
    g = d["subject_id"].values
    ra = fd.rank01(d["p_ehr"].values)
    rb = fd.rank01(d["p_ecg"].values)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                              random_state=42)
    for k, (tr, _) in enumerate(cv.split(ra, y, g)):
        rows.append((o, k, best_w(y[tr], ra[tr], rb[tr]),
                     "fold"))
    rows.append((o, -1, best_w(y, ra, rb), "global"))

res = pd.DataFrame(rows, columns=["outcome", "fold",
                                  "w_ehr", "kind"])
res.to_csv("fig12_weights.csv", index=False)
print(res.to_string())

fig, ax = plt.subplots(figsize=(7.6, 3.6))
outs = fd.OUTCOMES
ypos = np.arange(len(outs))[::-1]

for y0, o in zip(ypos, outs):
    f = res[(res["outcome"] == o)
            & (res["kind"] == "fold")]["w_ehr"]
    gl = float(res[(res["outcome"] == o)
                   & (res["kind"] == "global")]
               ["w_ehr"].iloc[0])
    ax.barh(y0, gl, height=0.46, color=fs.C["ehr"],
            edgecolor="white", linewidth=0.8, zorder=3)
    ax.barh(y0, 1.0 - gl, left=gl, height=0.46,
            color=fs.C["ecg"], hatch="///",
            edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(gl / 2.0, y0, "EHR %.2f" % gl, ha="center",
            va="center", fontsize=7.4, color="white",
            fontweight="bold", zorder=5)
    ax.text(gl + (1.0 - gl) / 2.0, y0,
            "ECG %.2f" % (1.0 - gl), ha="center",
            va="center", fontsize=7.4, color="white",
            fontweight="bold", zorder=5)
    for v in f:
        ax.plot([v, v], [y0 - 0.30, y0 + 0.30],
                color="#222222", lw=1.3, zorder=6)
    ax.text(1.015, y0, "folds %.2f$-$%.2f"
            % (f.min(), f.max()), fontsize=6.8,
            va="center", color="#444444")

ax.set_yticks(ypos)
ax.set_yticklabels([fd.NICE[o] for o in outs],
                   fontsize=7.6)
ax.set_xlim(0, 1.20)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("Share of the Fusion Weight "
              "(Both Modalities Sum to 1.0)")
fs.grid(ax, axis="x")
ax.set_ylim(-0.6, len(outs) - 0.4)

leg = [Line2D([], [], color="#222222", lw=1.3,
              label="weight selected in each of the "
                    "five training folds")]
ax.legend(handles=leg, loc="lower center",
          bbox_to_anchor=(0.42, 1.01), fontsize=7.2,
          handlelength=1.4)
ax.set_title("How the Fusion Weight Divides Between "
             "the Two Modalities", fontsize=10.5, pad=26)

fs.save(fig, "fig12_weights", "02_Results")

