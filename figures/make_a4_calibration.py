#!/usr/bin/env python3
"""A4 (Prof. Adam): reliability diagram + class-wise ECE.

Addresses the feedback that RQ3 calibration rested on a single, imbalance-
dominated ECE. Produces, for the primary model (ConvNeXt-Tiny, no
ColorJitter, raw uncalibrated probabilities):

  Panel A - reliability diagram, clean vs worst JPEG2000 (severity 6),
            15-bin risk calibration (positive-class probability against
            empirical event rate), overall ECE annotated.
  Panel B - class-wise (class-conditional) calibration gap: the mean
            absolute deviation of the predicted positive-class probability
            from the true label within each true class,
            ECE_c = mean_{i: y_i=c} |p_i - y_i|. For negatives this is the
            mean assigned risk (ideal 0); for positives the mean risk
            shortfall (ideal 1). Shown for clean and JPEG2000 s6.

Also writes a LaTeX table with overall ECE, the two class-conditional gaps,
and the imbalance-robust balanced class-wise ECE (their unweighted mean).

Data: the primary per-image prediction table from the data record.
(per-image label_45 and prob over 31 conditions; same source as the RQ3
figures). No inference is repeated.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import sys as _sys
_sys.path.insert(0, str(ROOT / "experiments"))
from paths import PRIMARY_RESULTS_CSV, OUT_DIR  # noqa: E402

CSV = PRIMARY_RESULTS_CSV
N_BINS = 15  # matches the thesis ECE definition

import thesisstyle as ts

ts.apply()
plt.rcParams["axes.grid"] = False   # panels set their own grid

COND = [("clean", 0, "Clean", "#000000"),
        ("jpeg2000", 6, "JPEG2000 s6", "#d62728")]
C_NEG, C_POS = "#4477aa", "#cc6677"


def ece(y, p, nb=N_BINS):
    y = np.asarray(y); p = np.asarray(p)
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p >= b[i]) & (p < b[i + 1]) if i < nb - 1 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum():
            e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return e


def reliability_points(y, p, nb=N_BINS):
    y = np.asarray(y); p = np.asarray(p)
    b = np.linspace(0, 1, nb + 1)
    xc, yc, w = [], [], []
    for i in range(nb):
        m = (p >= b[i]) & (p < b[i + 1]) if i < nb - 1 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum():
            xc.append(p[m].mean()); yc.append(y[m].mean()); w.append(int(m.sum()))
    return np.array(xc), np.array(yc), np.array(w)


def classwise_gaps(y, p):
    y = np.asarray(y); p = np.asarray(p)
    neg = p[y == 0]; pos = p[y == 1]
    gap_neg = float(np.mean(np.abs(neg - 0.0)))   # = mean(p | y=0), ideal 0
    gap_pos = float(np.mean(np.abs(pos - 1.0)))   # = 1 - mean(p | y=1), ideal 1
    return gap_neg, gap_pos


def main():
    d = pd.read_csv(CSV)
    rows = []
    # Explicit axes geometry rather than subplots+tight_layout: panel A is a
    # square reliability diagram (the y=x reference line only reads correctly at
    # aspect 1), and under tight_layout a square axes ends up shorter than the
    # rectangular panel B, so the two panel letters land at different heights
    # and A floats in whitespace. Placing both boxes by hand keeps the tops
    # flush and lets the panel letters sit on one baseline.
    FIG_H = 3.16
    AX_BOTTOM, AX_H = 0.166, 0.677
    A_X0 = 0.098
    A_W = AX_H * FIG_H / ts.FIGW          # square: equal width and height in inches
    B_X0 = A_X0 + A_W + 0.118             # gap holds panel B's ylabel and ticks
    B_W = 0.982 - B_X0
    fig = plt.figure(figsize=(ts.FIGW, FIG_H))
    axA = fig.add_axes([A_X0, AX_BOTTOM, A_W, AX_H])
    axB = fig.add_axes([B_X0, AX_BOTTOM, B_W, AX_H])

    # ── Panel A: reliability diagram ─────────────────────────────────────
    axA.plot([0, 1], [0, 1], ls=":", color="0.4", lw=1.2, label="Perfect")
    for dt, s, lab, col in COND:
        sub = d[(d.degradation_type == dt) & (d.severity == s)]
        y = sub.label_45.values; p = sub.prob.values
        e = ece(y, p)
        gneg, gpos = classwise_gaps(y, p)
        rows.append(dict(condition=lab, n=len(sub), n_pos=int(y.sum()),
                         ece=e, gap_neg=gneg, gap_pos=gpos,
                         balanced_ece=(gneg + gpos) / 2))
        xc, yc, w = reliability_points(y, p)
        sizes = 10 + 110 * (w / w.max())
        axA.plot(xc, yc, "-", color=col, lw=1.3, zorder=2)
        axA.scatter(xc, yc, s=sizes, color=col, edgecolor="white", linewidth=0.6,
                    zorder=3, label=f"{lab} (ECE {e:.3f})")
    axA.set_xlim(0, 1); axA.set_ylim(0, 1)
    # short axis label: the full "positive-class" wording is 2.6 in wide against
    # a 2.1 in axes, so it used to run under panel B; the caption carries the
    # precise definition
    axA.set_xlabel("Mean predicted probability")
    axA.set_ylabel("Empirical positive fraction")
    axA.set_title("Reliability diagram")
    # Tight handles/padding so the box stops short of the right spine. At the
    # matplotlib defaults it spanned the full axes width and its near-opaque
    # face hid the top end of the clean curve, whose last bin sits at
    # (0.979, 0.796) with a large marker.
    axA.legend(loc="upper left", framealpha=0.95, fontsize=ts.SMALL,
               handlelength=1.2, handletextpad=0.5, borderpad=0.35,
               borderaxespad=0.3, labelspacing=0.4)
    axA.grid(True, ls=":", lw=0.5, alpha=0.4)

    # ── Panel B: class-wise (class-conditional) calibration gap ──────────
    groups = ["Negatives\n(BI-RADS 1–3)", "Positives\n(BI-RADS 4+5)"]
    x = np.arange(2); bw = 0.36
    for k, (dt, s, lab, col) in enumerate(COND):
        r = rows[k]
        vals = [r["gap_neg"], r["gap_pos"]]
        bars = axB.bar(x + (k - 0.5) * bw, vals, bw, color=col, alpha=0.85,
                       edgecolor="white", label=lab)
        for xi, v in zip(x + (k - 0.5) * bw, vals):
            axB.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=ts.SMALL)
    axB.axhline(0, color="0.3", lw=1.0)
    axB.set_xticks(x); axB.set_xticklabels(groups)
    axB.set_ylim(0, 1.06)
    axB.set_ylabel("Calibration gap  $|\\,\\bar p - y\\,|$")
    axB.set_title("Class-wise calibration")
    axB.legend(loc="upper right", framealpha=0.95, fontsize=ts.SMALL)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

    # panel letters in FIGURE coordinates so they share one baseline
    for ax_x0, L in ((A_X0, "A"), (B_X0, "B")):
        fig.text(ax_x0 - 0.078, 0.978, L, fontsize=ts.BASE + 1,
                 fontweight="bold", va="top", ha="left")
    fig.savefig(HERE / "FIG_reliability_classwise.pdf")
    fig.savefig(HERE / "FIG_reliability_classwise.png", dpi=300)
    plt.close(fig)
    print("wrote FIG_reliability_classwise.pdf/.png")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "a4_classwise_ece.csv", index=False)
    print("wrote a4_classwise_ece.csv")
    print(df.to_string(index=False))

    # ── LaTeX table ──────────────────────────────────────────────────────
    def row(r):
        return (f"        {r['condition']} & {r['ece']:.3f} & {r['gap_neg']:.3f} & "
                f"{r['gap_pos']:.3f} & {r['balanced_ece']:.3f} \\\\")
    tex = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{6pt}",
        r"    \caption[Class-wise calibration decomposition]{%",
        r"             Class-wise decomposition of the expected calibration",
        r"             error for the primary model (raw probabilities). The",
        r"             overall \ac{ECE} ($15$ bins) is dominated by the",
        r"             negative class under the $4.95\,\%$ positive",
        r"             prevalence. The class-conditional gap",
        r"             $\text{ECE}_c = \operatorname{mean}_{i:\,y_i=c}|\hat p_i - y_i|$",
        r"             is the mean predicted risk on true negatives (ideal $0$)",
        r"             and the mean risk shortfall on true positives",
        r"             (ideal $1$); the balanced class-wise \ac{ECE} is their",
        r"             unweighted mean. Worst JPEG2000 inflates the negative",
        r"             class from $0.40$ to $0.79$ while the positive class",
        r"             improves, so the single overall \ac{ECE} overstates the",
        r"             positive-class harm and hides where the miscalibration",
        r"             concentrates.}",
        r"    \label{tab:classwise-ece}",
        r"    \begin{tabular}{lcccc}",
        r"        \toprule",
        r"        \textbf{Condition} & \textbf{ECE} & \textbf{ECE$_{\text{neg}}$}",
        r"        & \textbf{ECE$_{\text{pos}}$} & \textbf{Balanced ECE} \\",
        r"        \midrule",
    ]
    tex += [row(r) for _, r in df.iterrows()]
    tex += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    (HERE / "a4_classwise_ece_table.tex").write_text("\n".join(tex), encoding="utf-8")
    print("wrote a4_classwise_ece_table.tex")


if __name__ == "__main__":
    main()
