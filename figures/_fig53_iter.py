#!/usr/bin/env python3
"""Fast layout iteration for Fig 5.3 using the cached study-level rows
(_studylevel_rows.pkl) so we skip the 10-min bootstrap. Renders ONLY the
AUC/AP figure. Once the layout is dialled in, copy the params back into
fig3_4_studylevel.py::fig3()."""
from __future__ import annotations
import sys, pickle
from pathlib import Path
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubcommon as pc

rows, auc0, ap0 = pickle.load(open(SCRIPTS / "_studylevel_rows.pkl", "rb"))
OUT = pc.OUT_DIR / "fig53_iter"
OUT.mkdir(parents=True, exist_ok=True)


def render(figh, ncol, legfs, hlen, cspace, left, right, top, bottom, wspace, legy, tag, final=False):
    with plt.rc_context({"axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": legfs}):
        fig, axes = plt.subplots(1, 2, figsize=(pc.FIGW, figh))
        specs = [("auc", "auc_ci", "ROC-AUC", auc0),
                 ("ap", "ap_ci", "Average precision (AUPRC)", ap0)]
        for ax, (mk, cik, ylab, clean_val) in zip(axes, specs):
            for d in pc.DEG_ORDER:
                xs = [0] + list(range(1, 7))
                ys = [rows[("clean", 0)][mk] if s == 0 else rows[(d, s)][mk] for s in xs]
                lo = [rows[("clean", 0)][cik][0] if s == 0 else rows[(d, s)][cik][0] for s in xs]
                hi = [rows[("clean", 0)][cik][1] if s == 0 else rows[(d, s)][cik][1] for s in xs]
                ax.plot(xs, ys, marker=pc.DEG_MARKER[d], color=pc.DEG_COLOR[d], label=pc.DEG_LABEL[d])
                ax.fill_between(xs, lo, hi, color=pc.DEG_COLOR[d], alpha=0.13, lw=0)
            ax.axhline(clean_val, color="0.4", ls="--", lw=1.1, label="clean baseline")
            ax.axhline(clean_val - 0.05, color="0.6", ls=":", lw=1.0, label="baseline $-$ 0.05")
            pc.severity_axis(ax); ax.set_ylabel(ylab, fontsize=11)
            ax.tick_params(labelsize=10); ax.xaxis.label.set_size(11)
        axes[1].yaxis.set_label_position("right"); axes[1].yaxis.tick_right()
        pc.panel_label(axes[0], "A"); pc.panel_label(axes[1], "B", dx=-0.045)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, fontsize=legfs,
                   bbox_to_anchor=(0.5, legy), handlelength=hlen, columnspacing=cspace)
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)
        fig.savefig(OUT / f"fig53_{tag}.png", dpi=200)
        if final:
            for ext in ("pdf", "png"):
                fig.savefig(pc.FIG_MAIN / f"FIG_auc_auprc_vs_severity.{ext}",
                            dpi=300 if ext == "png" else None)
            print("wrote FINAL figures_main/FIG_auc_auprc_vs_severity")
        plt.close(fig)
        print("wrote", tag)


# FINAL = variant E: 2-row legend (ncol=4) at 10 pt, landscape panels
render(4.5, 4, 10, 1.3, 0.8, 0.11, 0.905, 0.935, 0.27, 0.06, 0.012, "E_final", final=True)
