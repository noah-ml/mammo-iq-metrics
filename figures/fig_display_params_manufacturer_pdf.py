"""Regenerate Figure 4.1 (display parameters by manufacturer) as a PDF.
Fixes: (1) IMS given its own colour (was lumped into grey 'Other'),
       (2) histograms use shared bins + log-count y-axis so the constant
           Siemens Window Width = 1500 spike no longer hides Planmed/IMS.
Driven by the full 20,000-row dicom_analysis/vindr_metadata.csv.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches

CSV = str(pc.DATA_DIR / "vindr_metadata.csv")
OUT = str(pc.FIG_MAIN / "figure2_display_params_by_manufacturer.pdf")

PALETTE = {"SIEMENS": "#2166ac", "Planmed": "#d6604d", "IMS": "#6a51a3", "Other": "#878787"}
ORDER   = ["SIEMENS", "Planmed", "IMS", "Other"]
HIST_ALPHA, SCAT_ALPHA, SCAT_SIZE = 0.55, 0.30, 5

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesisstyle as ts

ts.apply()
plt.rcParams.update({
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4, "legend.frameon": False,
})

df = pd.read_csv(CSV)

def norm_mfr(m):
    m = str(m)
    if m.upper().startswith("SIEMENS"): return "SIEMENS"
    if m.startswith("Planmed"):         return "Planmed"
    if m.upper().startswith("IMS"):     return "IMS"
    return "Other"

df["_m"] = df["Manufacturer"].map(norm_mfr)
present = [g for g in ORDER if (df["_m"] == g).any()]

def shared_bins(col, n=45):
    v = pd.to_numeric(df[col], errors="coerce")
    v = v[v > 0].dropna()
    return np.linspace(v.min(), v.max(), n + 1)

def hist_panel(ax, col, xlabel, title, logy=True):
    bins = shared_bins(col)
    for g in present:
        v = pd.to_numeric(df.loc[df["_m"] == g, col], errors="coerce")
        v = v[v > 0].dropna()
        if len(v):
            ax.hist(v, bins=bins, color=PALETTE[g], alpha=HIST_ALPHA,
                    linewidth=0, label=g)
    if logy:
        ax.set_yscale("log")
        ax.set_ylabel("Count (log scale)")
    else:
        ax.set_ylabel("Count")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=6)

def scatter_panel(ax, xcol, ycol, xlabel, ylabel, title):
    for g in present:
        sub = df.loc[df["_m"] == g, [xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub):
            ax.scatter(sub[xcol], sub[ycol], color=PALETTE[g], alpha=SCAT_ALPHA,
                       s=SCAT_SIZE, linewidths=0, rasterized=True, label=g)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=6)

fig, axes = plt.subplots(2, 2, figsize=(ts.FIGW, 4.6))
fig.subplots_adjust(hspace=0.42, wspace=0.30)
ax1, ax2, ax3, ax4 = axes.flat

# Panel headings match the rest of the thesis: a bold capital letter outside the
# axes (ts.panel_label) plus a plain descriptive title, not an in-title "(a)".
hist_panel(ax1, "WindowCenter", "Window Centre [a.u.]", "Window Centre")
hist_panel(ax2, "WindowWidth",  "Window Width [a.u.]",  "Window Width")
scatter_panel(ax3, "WindowCenter", "WindowWidth",
              "Window Centre [a.u.]", "Window Width [a.u.]", "Centre vs. Width")
hist_panel(ax4, "ImagerPixelSpacing_mm", "Imager Pixel Spacing [mm]",
           "Pixel spacing (first value)")

# The suptitle was set at y=1.02, i.e. above the canvas, and savefig.bbox is
# None, so it was clipped in half in the shipped PDF. It only restated the
# LaTeX caption, so it is dropped rather than moved.
for ax, L in zip((ax1, ax2, ax3, ax4), "ABCD"):
    ts.panel_label(ax, L, dx=-0.16, dy=1.30)
handles = [mpatches.Patch(facecolor=PALETTE[g], alpha=0.75, label=g) for g in present]
fig.legend(handles, [h.get_label() for h in handles],
           loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.004),
           fontsize=ts.SMALL, frameon=False, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT)
plt.close(fig)
print("Saved:", OUT)

# quick sanity print
for g in present:
    ww = pd.to_numeric(df.loc[df["_m"] == g, "WindowWidth"], errors="coerce").dropna()
    print(f"  {g:<8} n={len(ww):5d}  WW median={ww.median():.0f}")
