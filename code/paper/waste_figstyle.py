"""Shared figure style for the BanglaWaste-23 paper.

- Font: Latin Modern Roman (the LaTeX font of the paper), shipped in code/paper/fonts/ as TrueType, embedded in the PDFs.
- Width: the paper's text width (390 pt = 5.4 in), so figures are included at 100 % and the font sizes are real sizes.
- Colours: blue / orange / green and a blue sequential scale (as in the first version); the leakage bar chart
  (Fig. 4) adds a hatch pattern so its two series also stay apart in black-and-white print.
- No titles inside figures: titles and explanations go in the LaTeX caption; panels are labelled (a), (b) under each panel.
"""
from pathlib import Path
import matplotlib
from matplotlib import font_manager as fm

FONT_DIR = Path(__file__).resolve().parent / "fonts"
for _f in sorted(FONT_DIR.glob("LMRomanPaper-*.ttf")):
    fm.fontManager.addfont(str(_f))

TEXTWIDTH = 5.4                     # inches = \textwidth of the manuscript (elsarticle, review, 12 pt)
INK, INK2, GRID = "#000000", "#333333", "#d9d9d9"
DARK, MID, LIGHT = "#404040", "#8c8c8c", "#d0d0d0"      # neutral greys for lines and guides

STYLE = {
    "font.family": "LM Roman Paper", "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.titlesize": 9, "axes.titleweight": "normal",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False, "hatch.linewidth": 0.6, "hatch.color": INK,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def apply():
    matplotlib.rcParams.update(STYLE)


def label_panels(fig, axes_letters, pad=0.012, fontsize=9.5):
    """Write (a), (b), ... centred under each axes, below its tick labels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in axes_letters:
        bb = ax.get_tightbbox(r).transformed(inv)
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, bb.y0 - pad, f"({letter})", ha="center", va="top", fontsize=fontsize)
