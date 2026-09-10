#!/usr/bin/env python3
"""RQ1 multi-architecture robustness: figure + LaTeX table.

Data source: the 3-arch x 3-seed no-ColorJitter degradation-eval consolidated on
the cluster (W&B run rq1_multiarch_nojitter_3arch_3seed_tau_final, 2026-07-20),
shipped in the data record as multiarch_eval_tables/. No inference is repeated here.

- Figure: ROC-AUC vs severity, one panel per degradation type, one line per
  architecture. Shaded band = mean +/- 1 SD over the three seeds (42, 43, 44).
- Table: severity-6 robustness by architecture (clean AUC, AUC@s6, relative
  AUC drop, and the same for AP), seed spread as +/- 1 SD.

Style matches thesis_publication_figures (pubcommon.py): DejaVu Sans, bold panel
titles, light grid. Architecture colours are Wong colour-blind-safe.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(HERE.parent / "experiments"))
from paths import DATA_DIR, OUT_DIR  # noqa: E402

# Aggregated cross-architecture tables, as shipped in the data record.
EVAL = DATA_DIR / "multiarch_eval_tables"
OUT = OUT_DIR / "publication_figures" / "figures_main"
OUT.mkdir(parents=True, exist_ok=True)

# ── style (mirrors thesis pubcommon.py) ──────────────────────────────────────
import thesisstyle as ts

ts.apply()
FIGW = ts.FIGW

DEG_ORDER = ["noise", "blur", "contrast", "jpeg2000", "resolution"]
DEG_LABEL = {
    "noise": "Dose-motivated noise", "blur": "Motion blur", "contrast": "Contrast reduction",
    "jpeg2000": "JPEG2000 compression", "resolution": "Resolution reduction",
}
ARCH_ORDER = ["convnext_tiny", "efficientnet_b4", "resnet18"]
ARCH_LABEL = {
    "convnext_tiny": "ConvNeXt-Tiny", "efficientnet_b4": "EfficientNet-B4",
    "resnet18": "ResNet-18",
}
ARCH_COLOR = {  # Wong palette
    "convnext_tiny": "#D55E00",    # vermillion (primary architecture)
    "efficientnet_b4": "#0072B2",  # blue
    "resnet18": "#009E73",         # green
}
ARCH_MARKER = {"convnext_tiny": "o", "efficientnet_b4": "s", "resnet18": "^"}

# ── data ─────────────────────────────────────────────────────────────────────
arch = pd.read_csv(EVAL / "arch_level_mean_std_by_severity.csv")
clean = pd.read_csv(EVAL / "clean_performance_mean_std.csv").set_index("arch")
s6 = pd.read_csv(EVAL / "severity6_mean_std.csv")


def series(a, d, metric):
    """Return severity (0..6), mean, std arrays for one arch/deg, metric in {auc,ap}.

    Severity 0 is the clean baseline (identical across degradation types)."""
    sub = arch[(arch.arch == a) & (arch.degradation_type == d)].sort_values("severity")
    sev = [0] + sub.severity.tolist()
    mean = [clean.loc[a, f"clean_{metric}_mean"]] + sub[f"{metric}_mean"].tolist()
    std = [clean.loc[a, f"clean_{metric}_std"]] + sub[f"{metric}_std"].tolist()
    return np.array(sev), np.array(mean), np.array(std)


# ── figure: AUC vs severity, panel per degradation, line per architecture ─────
def make_figure():
    # two columns rather than three: at the 6.10 in text width a three-column
    # mosaic leaves ~1.5 in per panel, too narrow for the 10 pt panel titles
    # Sizes come from thesisstyle (labels/titles 10 pt, ticks 9.5 pt). The local
    # 11/10 pt override that used to sit here made this the only results figure
    # a size step above the others.
    mosaic = [["noise", "blur"],
              ["contrast", "jpeg2000"],
              ["resolution", "legend"]]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(FIGW, 6.4),
                                  gridspec_kw={"hspace": 0.58, "wspace": 0.26,
                                               "left": 0.105, "right": 0.985,
                                               "top": 0.950, "bottom": 0.085})
    LEFT_COL = {"noise", "contrast", "resolution"}

    for d in DEG_ORDER:
        ax = axd[d]
        for a in ARCH_ORDER:
            sev, mean, std = series(a, d, "auc")
            ax.plot(sev, mean, marker=ARCH_MARKER[a], color=ARCH_COLOR[a],
                    label=ARCH_LABEL[a], zorder=3, clip_on=False)
            ax.fill_between(sev, mean - std, mean + std, color=ARCH_COLOR[a],
                            alpha=0.15, lw=0, zorder=1)
        ax.axhline(0.5, color="0.35", ls="--", lw=1.0, zorder=0)
        ax.text(6, 0.5, " chance", color="0.35", fontsize=ts.SMALL, va="bottom", ha="right")
        ax.set_title(DEG_LABEL[d], pad=6)
        ax.set_xticks(range(7))
        ax.set_xlabel("Severity (0 = clean)")
        if d in LEFT_COL:
            ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.45, 0.86)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_leg = axd["legend"]
    ax_leg.set_axis_off()
    handles, labels = axd["noise"].get_legend_handles_labels()
    leg = ax_leg.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.62),
                        title="Architecture",
                        title_fontsize=ts.BASE, fontsize=ts.SMALL, frameon=True,
                        framealpha=1.0, edgecolor="#cccccc", borderpad=0.8,
                        handlelength=1.8, labelspacing=0.7)
    leg.get_title().set_fontweight("bold")
    # The explanatory note that used to sit here ("Band: mean +- 1 SD over seeds
    # 42, 43, 44 / Dashed line: AUC = 0.5 (chance)") repeated the LaTeX caption
    # word for word, overran the legend cell once the fonts were normalised to
    # 9.5 pt, and was the last place in the thesis where "SD" appeared
    # unexpanded. The caption states both facts.

    fig.savefig(OUT / "FIG_rq1_multiarch_auc_vs_severity.pdf")
    fig.savefig(OUT / "FIG_rq1_multiarch_auc_vs_severity.png", dpi=300)
    plt.close(fig)
    print("wrote FIG_rq1_multiarch_auc_vs_severity.pdf/.png")


# ── table: severity-6 robustness by architecture ─────────────────────────────
def fmt(mean, std, pct=False, digits=3):
    if pct:
        return f"{100 * mean:.1f} $\\pm$ {100 * std:.1f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def make_table():
    rows = []
    for a in ARCH_ORDER:
        cl_auc = clean.loc[a, "clean_auc_mean"]; cl_auc_sd = clean.loc[a, "clean_auc_std"]
        cl_ap = clean.loc[a, "clean_ap_mean"]; cl_ap_sd = clean.loc[a, "clean_ap_std"]
        for d in DEG_ORDER:
            r = s6[(s6.arch == a) & (s6.degradation_type == d)].iloc[0]
            rows.append(dict(
                arch=a, deg=d,
                clean_auc=cl_auc, clean_auc_sd=cl_auc_sd,
                s6_auc=r.auc_mean, s6_auc_sd=r.auc_std,
                drop_auc=r.auc_rel_drop_mean, drop_auc_sd=r.auc_rel_drop_std,
                clean_ap=cl_ap, clean_ap_sd=cl_ap_sd,
                s6_ap=r.ap_mean, s6_ap_sd=r.ap_std,
                drop_ap=r.ap_rel_drop_mean, drop_ap_sd=r.ap_rel_drop_std,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rq1_multiarch_s6_table.csv", index=False)
    print("wrote rq1_multiarch_s6_table.csv")

    # LaTeX (booktabs). One block per architecture; rows = degradation.
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption[Multi-architecture robustness at maximum severity]{%")
    lines.append(r"Robustness of the three architectures at the strongest tested "
                 r"severity (level~6), no-ColorJitter models. Each entry is the "
                 r"mean $\pm$ one standard deviation over three training seeds "
                 r"(42, 43, 44). $\Delta$ is the relative drop from the clean "
                 r"baseline, $(\text{clean}-\text{severity 6})/\text{clean}$. "
                 r"Contrast reduction is the negative control and leaves ranking "
                 r"performance essentially unchanged for all three architectures; "
                 r"EfficientNet-B4 alone collapses to chance under dose noise.}")
    lines.append(r"\label{tab:rq1-multiarch-s6}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l l r r r}")
    lines.append(r"\toprule")
    lines.append(r"Architecture & Degradation & AUC$_{s6}$ & $\Delta$AUC (\%) "
                 r"& $\Delta$AUPRC (\%) \\")
    lines.append(r"\midrule")
    for a in ARCH_ORDER:
        sub = df[df.arch == a]
        cl_auc = sub.iloc[0].clean_auc; cl_auc_sd = sub.iloc[0].clean_auc_sd
        cl_ap = sub.iloc[0].clean_ap; cl_ap_sd = sub.iloc[0].clean_ap_sd
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{ARCH_LABEL[a]}}} "
                     rf"(clean AUC {cl_auc:.3f}, AUPRC {cl_ap:.3f})}} \\")
        for d in DEG_ORDER:
            r = sub[sub.deg == d].iloc[0]
            lines.append(
                rf"\quad {DEG_LABEL[d]} & & {fmt(r.s6_auc, r.s6_auc_sd)} & "
                rf"{fmt(r.drop_auc, r.drop_auc_sd, pct=True)} & "
                rf"{fmt(r.drop_ap, r.drop_ap_sd, pct=True)} \\")
        if a != ARCH_ORDER[-1]:
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (OUT / "rq1_multiarch_s6_table.tex").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
    print("wrote rq1_multiarch_s6_table.tex")

    # console summary
    print("\n--- severity-6 relative AUC drop (%, mean over 3 seeds) ---")
    piv = df.pivot(index="arch", columns="deg", values="drop_auc").reindex(
        ARCH_ORDER)[DEG_ORDER] * 100
    print(piv.round(1).to_string())


if __name__ == "__main__":
    make_figure()
    make_table()
