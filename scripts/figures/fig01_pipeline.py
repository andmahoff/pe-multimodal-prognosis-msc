"""Figure 1 - study pipeline schematic (Chapter 3.1)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import fig_style as fs

fs.init()
fig, ax = plt.subplots(figsize=(7.4, 5.2))
fs.blank(ax)
ax.set_ylim(0, 1.06)
ax.text(0.5, 1.045, "Study Pipeline and Training "
        "Regime by Modality", ha="center", va="top",
        fontsize=10.5, fontweight="bold")

fs.box(ax, 0.03, 0.855, 0.44, 0.115,
       "STANFORD INSPECT\nCTPA cohort, PE-coded\n"
       "n = 4,524 $\\rightarrow$ 3,300 index scans",
       fc=fs.C["zs"], fs=7.6, weight="bold")
fs.box(ax, 0.53, 0.855, 0.44, 0.115,
       "MIMIC-IV (BIDMC)\nPE index admissions\n"
       "n = 3,512 (3,169 patients)",
       fc=fs.C["tgt"], fs=7.6, weight="bold")

AY, AH = 0.545, 0.225
arms = [
    (0.030, "EHR\n28 features\nL2 logistic\n\n"
            "TRAIN: INSPECT\nZERO-SHOT",
     fs.C["zs"], fs.C["ehr"]),
    (0.275, "ECG\n71 SCP logits\nfrozen PTB-XL\n"
            "xresnet1d101\nTRAIN: MIMIC\nout-of-fold",
     fs.C["tgt"], fs.C["ecg"]),
    (0.520, "CTPA report\n46 rule-based\ntext features\n"
            "L2 logistic\nTRAIN: MIMIC\nout-of-fold",
     fs.C["tgt"], fs.C["ctpa"]),
    (0.765, "CXR\nDenseNet-121\n"
            "CheXpert weights\n\nTRAIN: MIMIC\n"
            "out-of-fold", fs.C["tgt"], fs.C["cxr"]),
]
for x, txt, fc, ec in arms:
    fs.box(ax, x, AY, 0.205, AH, txt, fc=fc, ec=ec,
           fs=7.0)

fs.arrow(ax, 0.25, 0.855, 0.1325, AH + AY + 0.004,
         color="#6699BB", lw=1.1)
for x in [0.1325, 0.3775, 0.6225, 0.8675]:
    fs.arrow(ax, 0.75, 0.855, x, AH + AY + 0.004,
             color="#B08A50", lw=1.0, rad=0.06)

fs.box(ax, 0.13, 0.435, 0.74, 0.065,
       "Rank normalisation of each modality's out-of-fold "
       "predictions", "#F2F4F6", fs=7.6)
for x in [0.1325, 0.3775, 0.6225, 0.8675]:
    fs.arrow(ax, x, AY, min(max(x, 0.17), 0.83), 0.502)

fs.box(ax, 0.13, 0.300, 0.74, 0.090,
       "WEIGHTED LATE FUSION  (WMEAN2 / WMEAN3)\n"
       "simplex weight grid tuned inside training "
       "folds only", fc="#FFFFFF",
       ec=fs.C["fused"], fs=7.8, weight="bold")
fs.arrow(ax, 0.50, 0.435, 0.50, 0.392)

fs.box(ax, 0.13, 0.120, 0.74, 0.135,
       "EVALUATION on MIMIC-IV\n"
       "AUC-ROC | average precision | DeLong | "
       "paired subject-level bootstrap\n"
       "Platt calibration | categorical NRI | "
       "decision curve analysis\n"
       "Benchmark: sPESI-6", fc=fs.C["pale"], fs=7.2)
fs.arrow(ax, 0.50, 0.300, 0.50, 0.257)

leg = [
    Patch(facecolor=fs.C["zs"], edgecolor="#444444",
          label="Zero-shot: developed on INSPECT, never "
                "sees MIMIC-IV labels"),
    Patch(facecolor=fs.C["tgt"], edgecolor="#444444",
          label="MIMIC-trained: cross-validated "
                "out-of-fold within MIMIC-IV"),
]
ax.legend(handles=leg, loc="upper center",
          bbox_to_anchor=(0.5, 0.105), ncol=1,
          fontsize=7.4, handlelength=1.4,
          borderpad=0.2, labelspacing=0.35)

fs.save(fig, "fig01_pipeline", "01_Methodology")

