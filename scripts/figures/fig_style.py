"""Shared style and drawing helpers for all dissertation
figures. Every figure script imports from this file.

Design rules applied throughout, following Chartability
(Elavsky et al., 2022) and Cleveland & McGill (1984):
  - colour is never the only grouping channel; every
    bar series also carries a hatch pattern, so the
    figures survive greyscale printing
  - faint gridlines on both axes for value reading
  - direct labels preferred over legends where space
    allows, to lower cognitive load
  - Okabe-Ito / ColorBrewer colourblind-safe palette
  - titles centre on the whole canvas, not the
    plotting area (see _centre_title)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch

FIGROOT = "./fusion_workspace/figures"

C = {
    "ehr":   "#0072B2",
    "ecg":   "#E69F00",
    "ctpa":  "#009E73",
    "cxr":   "#CC79A7",
    "spesi": "#7F7F7F",
    "fused": "#D55E00",
    "ref":   "#BBBBBB",
    "ink":   "#222222",
    "pale":  "#F5F5F5",
    "zs":    "#D6E8F5",
    "tgt":   "#FBE6C9",
    "neg":   "#D55E00",
    "pos":   "#0072B2",
}

# Redundant encoding for greyscale and for readers who
# cannot separate the hues. Index matches series order.
HATCH = ["", "///", "...", "\\\\\\", "xx", "||", "--"]


def init():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E4E4E4",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "hatch.linewidth": 0.55,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def grid(ax, axis="both"):
    """Faint gridlines on both axes, behind the data."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color="#E4E4E4", lw=0.6,
            zorder=0)


def suptitle(fig, text, y=1.02):
    """Bold figure title. fig.suptitle does not inherit
    axes.titleweight, so weight is set explicitly."""
    fig.suptitle(text, fontsize=10.5,
                 fontweight="bold", y=y)


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def box(ax, x, y, w, h, text, fc="#FFFFFF",
        ec="#444444", fs=7.2, weight="normal",
        va="center", pad=0.010):
    st = "round,pad=%.3f,rounding_size=0.012" % pad
    p = FancyBboxPatch((x, y), w, h, boxstyle=st,
                       linewidth=0.9, facecolor=fc,
                       edgecolor=ec)
    ax.add_patch(p)
    ty = y + h / 2.0 if va == "center" else y + h - 0.02
    ax.text(x + w / 2.0, ty, text, ha="center",
            va=va, fontsize=fs, fontweight=weight,
            linespacing=1.35, color=C["ink"])


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=0.9,
          rad=0.0):
    cs = "arc3,rad=%.2f" % rad
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>",
                        connectionstyle=cs,
                        mutation_scale=8, linewidth=lw,
                        color=color, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def panel_tag(ax, letter, x=-0.14, y=1.14):
    """Panel letter. Defaults sit clear of the topmost
    y tick label; pass x/y to nudge per figure."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")


def chance(ax, y=0.5):
    ax.axhline(y, ls="--", lw=0.8, color=C["ref"],
               zorder=0)


def _centre_title(fig):
    """Promote a lone axes title to a figure title and
    centre it on the tight bounding box, not the figure
    box. savefig crops to the tight bbox, so long y-axis
    labels would otherwise leave the title off-centre.
    Multi-panel figures already use fig.suptitle and are
    left alone."""
    if fig._suptitle is not None:
        return
    titled = [a for a in fig.axes
              if a.get_title().strip()]
    if len(titled) != 1:
        return
    ax = titled[0]
    t = ax.title
    txt = t.get_text()
    size = t.get_fontsize()
    ax.set_title("")
    fig.canvas.draw()
    try:
        r = fig.canvas.get_renderer()
        bb = fig.get_tightbbox(r)
        xc = ((bb.x0 + bb.x1) / 2.0) / fig.get_figwidth()
    except Exception:
        xc = 0.5
    fig.suptitle(txt, x=xc, y=1.0, va="bottom",
                 fontsize=size, fontweight="bold")


def save(fig, stem, chapter):
    _centre_title(fig)
    d = os.path.join(FIGROOT, chapter)
    if not os.path.isdir(d):
        os.makedirs(d)
    fig.savefig(os.path.join(d, stem + ".pdf"))
    fig.savefig(os.path.join(d, stem + ".png"), dpi=300)
    plt.close(fig)
    print("saved: " + chapter + "/" + stem)

