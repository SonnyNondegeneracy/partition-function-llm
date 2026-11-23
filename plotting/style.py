"""Shared Nature-style figure settings."""
import os
from matplotlib import font_manager
import matplotlib as mpl

BLUE, ORANGE = "#0072B2", "#D55E00"      # T1, T2 everywhere
INK, MUTED, GRID = "#243746", "#5D6A73", "#D5DDE3"
MM = 1 / 25.4


def setup():
    for f in ("Arial.ttf", "Arial_Bold.ttf", "Arial_Italic.ttf", "Arial_Bold_Italic.ttf"):
        p = os.path.expanduser(f"~/.fonts/{f}")
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.edgecolor": INK, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    })
