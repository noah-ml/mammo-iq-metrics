#!/usr/bin/env python3
"""FIG 2 — Global IQM (SSIM, Tenengrad, noise variance, tau) vs severity, all test images."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pubcommon as pc


def boot_mean_ci(x, n_boot=600, seed=pc.RNG_SEED):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); n = len(x)
    ms = [rng.choice(x, n, replace=True).mean() for _ in range(n_boot)]
    return np.mean(x), np.percentile(ms, 2.5), np.percentile(ms, 97.5)


def main():
    R = pc.load_results()
    # Fonts come from thesisstyle (labels 10 pt, ticks/legend 9.5 pt) so this
    # figure matches every other one; the local 11/10 pt override that used to
    # sit here made it the only figure a size step above the rest. Plots are
    # kept at their on-page size by tightening the margins explicitly rather
    # than letting tight_layout inflate them.
    metrics = [("ssim", "SSIM (vs clean)"), ("tenengrad", "Tenengrad"),
               ("noise_var", "Noise variance"), ("tau", "τ")]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(pc.FIGW, 5.7))
    axes = axes.ravel()
    for ax, (m, ylab) in zip(axes, metrics):
        for d in pc.DEG_ORDER:
            xs, ys, los, his = [], [], [], []
            for s in range(0, 7):
                dtype = "clean" if s == 0 else d
                sub = R[(R.degradation_type == dtype) & (R.severity == s)]
                if len(sub) == 0:
                    continue
                mean, lo, hi = boot_mean_ci(sub[m].values)
                xs.append(s); ys.append(mean); los.append(lo); his.append(hi)
                rows.append(dict(metric=m, degradation=d, severity=s, mean=mean,
                                 median=float(sub[m].median()), ci_lo=lo, ci_hi=hi, n=len(sub)))
            ax.plot(xs, ys, marker=pc.DEG_MARKER[d], color=pc.DEG_COLOR[d], label=pc.DEG_LABEL[d])
            ax.fill_between(xs, los, his, color=pc.DEG_COLOR[d], alpha=0.15, lw=0)
        pc.severity_axis(ax); ax.set_ylabel(ylab)
    for idx in (1, 3):                      # right column: y-axis on the outer edge
        axes[idx].yaxis.set_label_position("right")
        axes[idx].yaxis.tick_right()
    for i, (ax, L) in enumerate(zip(axes, "ABCD")):
        # keep the panel letter on the inner-top corner but out of the way of the
        # opposite column when the y-axis sits on the right
        pc.panel_label(ax, L, dx=(-0.045 if i in (1, 3) else -0.10))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.015), handlelength=1.8, columnspacing=1.6)
    # Right-column y-axes moved to the outer edge, so the labels live in the left
    # and right margins and the inter-column gap only needs a little separation.
    fig.subplots_adjust(left=0.132, right=0.872, top=0.945, bottom=0.20,
                        wspace=0.05, hspace=0.40)
    pc.save_fig(fig, "FIG_global_iqm_vs_severity")
    pd.DataFrame(rows).to_csv(pc.TABLES / "global_iqm_vs_severity.csv", index=False)
    print("  wrote tables/global_iqm_vs_severity.csv")


if __name__ == "__main__":
    main()
