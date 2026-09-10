"""Shared figure style for the thesis plots (jour fixe 2026-07-29, point 2).

Problem this fixes
------------------
The figures were previously drawn at 11-20 inches wide and then included with
``\\includegraphics[width=\\textwidth]``. The thesis text block is 15.5 cm
(A4 minus 3 cm + 2.5 cm margins) = 6.10 in, so every figure was downscaled by
0.30-0.55x. A 10 pt label therefore reached the page at 3-5 pt, far smaller
than the 12 pt body text, and the effective size differed from figure to
figure depending on the scale factor.

Fix
---
Draw every ``width=\\textwidth`` figure at exactly ``FIGW`` inches so that it is
placed 1:1 with no downscaling, and set every in-plot font to 9.5-10 pt. On the
page all figure text is then within about 0.8x of the body text and uniform
across figures.

Usage::

    import thesisstyle as ts
    ts.apply()
    fig, axes = plt.subplots(2, 2, figsize=(ts.FIGW, 4.6))
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# A4, geometry left=3cm right=2.5cm -> textwidth 15.5 cm = 6.102 in
FIGW = 6.10
# height cap: \textheight is about 24.5 cm; keep single figures well under it
FIGH_MAX = 8.2

BASE = 10.0      # body text is 12 pt; 10 pt reads as "slightly smaller"
SMALL = 9.5      # ticks and legend, deliberately close to BASE

RC = {
    "font.family": "DejaVu Sans",
    "font.size": BASE,
    "axes.titlesize": BASE,
    "axes.titleweight": "bold",
    "axes.labelsize": BASE,
    "xtick.labelsize": SMALL,
    "ytick.labelsize": SMALL,
    "legend.fontsize": SMALL,
    "legend.title_fontsize": SMALL,
    "figure.titlesize": BASE,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    # NOT "tight": a tight bbox trims the canvas to the drawn content, so the
    # saved width drifts from FIGW and LaTeX rescales each figure by a
    # different factor, which is what made the effective font sizes differ
    # between figures in the first place. The scripts call tight_layout(), so
    # everything already fits inside the requested canvas.
    "savefig.bbox": None,
    "savefig.pad_inches": 0.02,
    "figure.dpi": 120,
    "pdf.fonttype": 42,   # embed TrueType, keeps text selectable/searchable
    "ps.fonttype": 42,
}


def apply() -> None:
    """Install the thesis figure style into the global rcParams."""
    plt.rcParams.update(RC)


def panel_label(ax, letter, dx=-0.10, dy=1.10):
    """Bold panel letter, sized to match the rest of the figure text."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=BASE + 1,
            fontweight="bold", va="top", ha="right")
