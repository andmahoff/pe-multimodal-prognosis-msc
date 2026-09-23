"""Figure 19a - when composite events arrive, by route
(Chapter 4.6). Harmonised labels."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fig_style as fs
import fig_data as fd

fs.init()
lab = pd.read_csv(fd.DATA + "/mimic_labels_harmonised.csv")
w = pd.read_csv(fd.DATA + "/p_wmean2_composite_30d.csv")
d = lab.merge(w, on=fd.KEY)

death = (d["death_first"] == 1).values
cv = (d["cv_first"] == 1).values
dd = pd.to_numeric(d["dod_days"], errors="coerce").values
cd = pd.to_numeric(d["cv_days"], errors="coerce").values
day = np.where(death, dd, np.where(cv, cd, np.nan))
print("death-first %d | cv-first %d"
      % (death.sum(), cv.sum()))

fig, ax = plt.subplots(figsize=(7.2, 3.6))
bins = np.arange(0, 32, 2)
ax.hist([day[death][~np.isnan(day[death])],
         day[cv][~np.isnan(day[cv])]],
        bins=bins, stacked=True,
        color=[fs.C["ehr"], fs.C["ctpa"]],
        hatch=["", "///"],
        edgecolor="white", linewidth=0.7, zorder=3,
        label=["all-cause death first (%d)"
               % int(death.sum()),
               "CV readmission first (%d)"
               % int(cv.sum())])
fs.grid(ax, axis="y")
ax.set_xlabel("Days From Index Admission to "
              "First Event")
ax.set_ylabel("Events")
ax.set_xlim(0, 30)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.legend(ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), fontsize=7.4,
          handlelength=1.3, columnspacing=1.6)
ax.set_title("Timing of First Events by Route",
             fontsize=10.5, pad=28)

fs.save(fig, "fig19a_route_timing", "02_Results")

