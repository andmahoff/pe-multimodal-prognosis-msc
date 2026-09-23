"""Figure 2 - two-column cohort flow (Chapter 3.4)."""
import matplotlib.pyplot as plt
import fig_style as fs

fs.init()
fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.8))

BX, BW = 0.030, 0.605
SX, SW = 0.670, 0.320


def chain(ax, title, tcol, rows, stubs):
    fs.blank(ax)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(title, fontsize=9.5, pad=6,
                 color=tcol)
    n = len(rows)
    top, gap = 0.955, 0.058
    h = (top - 0.01 - gap * (n - 1)) / float(n)
    ys = []
    for i, (txt, fc, ec) in enumerate(rows):
        y = top - h - i * (h + gap)
        ys.append(y)
        wt = "bold" if i == n - 1 else "normal"
        fs.box(ax, BX, y, BW, h, txt, fs=6.5,
               weight=wt, fc=fc, ec=ec)
        if i > 0:
            fs.arrow(ax, BX + BW / 2.0, y + h + gap,
                     BX + BW / 2.0, y + h, lw=1.1,
                     color="#555555")
    for i, txt in enumerate(stubs):
        if txt is None:
            continue
        ymid = ys[i] + h + gap / 2.0
        fs.box(ax, SX, ymid - 0.030, SW, 0.060, txt,
               fs=5.9, fc="#F4F0EC", ec="#B08A80",
               pad=0.006)
        fs.arrow(ax, BX + BW / 2.0, ymid, SX, ymid,
                 color="#B08A80")


GREY = ("#EFEFEF", "#8A8A8A")
mimic_rows = [
    ("MIMIC-IV v3.1\nadult hospital admissions",) + GREY,
    ("Pulmonary embolism index admissions\n"
     "ICD-10 I26 or ICD-9 4151, age $\\geq$ 18\n"
     "n = 3,512 admissions (3,169 patients)",
     "#FBE6C9", "#C08A20"),
    ("At least one ECG recorded\n"
     "$-$12 h to $+$48 h of admission\n"
     "PRIMARY COHORT   n = 3,507",
     "#FDEFD0", fs.C["ecg"]),
    ("Index CTPA report\n"
     "$-$48 h to $+$24 h of admission\n"
     "n = 1,703   (1,649 with impression)",
     "#CDEBDF", fs.C["ctpa"]),
    ("CXR during\nthe index admission\n"
     "n = 1,027 (960 patients)",
     "#F7DCEA", fs.C["cxr"]),
]
mimic_stubs = [
    None,
    "Excluded: no PE code;\nage < 18",
    "Excluded: no qualifying\nECG (n = 5)",
    "Excluded: no index\nCTPA report (n = 1,804)",
]

inspect_rows = [
    ("INSPECT EHR (Stanford)\n19,248 persons with CTPA",)
    + GREY,
    ("Pulmonary embolism condition\ncode present\n"
     "n = 4,524 persons", "#E4F0F9", "#6FA8CF"),
    ("Index CTPA within 30 days\nof first PE date\n"
     "n = 3,300 persons", "#D0E6F5", "#3E8FC4"),
    ("Not censored for 1-month mortality\n"
     "TRAINING COHORT   n = 3,136",
     "#BBDBF0", fs.C["ehr"]),
]
inspect_stubs = [
    None,
    "Excluded: no PE code\n(n = 14,724)",
    "Excluded: index scan\n> 30 d from PE date",
]

chain(axes[0], "MIMIC-IV  (Evaluation)", "#8A5A10",
      mimic_rows, mimic_stubs)
chain(axes[1], "Stanford INSPECT  (Development)",
      fs.C["ehr"], inspect_rows, inspect_stubs)

fig.subplots_adjust(wspace=0.16, left=0.01, right=0.99,
                    top=0.92, bottom=0.01)
fs.save(fig, "fig02_cohort_flow", "01_Methodology")

