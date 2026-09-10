#!/usr/bin/env python3
"""Shared style + metric helpers for the thesis publication figures.

Data source: the production degradation results CSV (ConvNeXt-Tiny no-ColorJitter,
W&B run convnext-tiny-...-nojitter-tau-strict-clean, training job 956694), which
contains per-image per-condition logit, prob and the global IQMs (ssim, tenengrad,
noise_var, tau) plus lesion cnr/delta_mu. The tau column is the nonparametric noise
measure of Anton et al. 2023 (Phys. Med. Biol. 68:045003), computed on the breast
mask. This is the same evaluation as run 957382 with the tau IQM added; clean
AUC/AP are identical (0.8241 / 0.4364). No model inference is repeated in these scripts.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

# Paths resolve through experiments/paths.py, so MAMMO_DATA_DIR and MAMMO_OUT_DIR
# steer the figure scripts and the analysis scripts alike.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from paths import DATA_DIR, OUT_DIR, PRIMARY_RESULTS_CSV  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PUBDIR = OUT_DIR / "publication_figures"
RESULTS_CSV = PRIMARY_RESULTS_CSV
#: Clean validation predictions, an input to the calibration figures.
VAL_PREDICTIONS = DATA_DIR / "val_clean_predictions.csv"

FIG_MAIN = PUBDIR / "figures_main"
FIG_APP = PUBDIR / "figures_appendix"
CAPS = PUBDIR / "captions"
TABLES = PUBDIR / "tables"
for _d in (FIG_MAIN, FIG_APP, CAPS, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

N_TEST = 4000
N_POS = 198
N_BOOT = 1000
RNG_SEED = 2026

# ── consistent style ──────────────────────────────────────────────────────
# Sizes live in thesisstyle.py: every width=\textwidth figure is drawn at
# FIGW = 6.10 in so LaTeX places it 1:1, and all in-plot text sits at
# 9.5-10 pt against the 12 pt body text (jour fixe 2026-07-29).
import thesisstyle as ts

ts.apply()
FIGW = ts.FIGW

DEG_ORDER = ["noise", "blur", "contrast", "jpeg2000", "resolution"]
DEG_LABEL = {
    "noise": "Dose noise", "blur": "Motion blur", "contrast": "Contrast reduction",
    "jpeg2000": "JPEG2000", "resolution": "Resolution reduction",
}
DEG_COLOR = {
    "noise": "#1f77b4", "blur": "#ff7f0e", "contrast": "#2ca02c",
    "jpeg2000": "#d62728", "resolution": "#9467bd",
}
DEG_MARKER = {"noise": "o", "blur": "s", "contrast": "^", "jpeg2000": "D", "resolution": "v"}
IQM_LABEL = {"ssim": "Mean SSIM", "tenengrad": "Mean Tenengrad",
             "noise_var": "Mean noise variance", "tau": "Mean τ"}


def load_results() -> pd.DataFrame:
    R = pd.read_csv(RESULTS_CSV)
    R["image_id"] = R["image_id"].astype(str)
    return R


def long_with_clean(R: pd.DataFrame) -> pd.DataFrame:
    """Attach clean logit/prob/IQMs to every row; add abs/signed delta_logit."""
    clean = R[R.degradation_type == "clean"].set_index("image_id")
    add = clean[["logit", "prob", "ssim", "tenengrad", "noise_var", "tau", "cnr", "delta_mu"]].rename(
        columns=lambda c: f"clean_{c}")
    M = R.merge(add, on="image_id", how="left")
    M["signed_delta_logit"] = M.logit - M.clean_logit
    M["abs_delta_logit"] = M.signed_delta_logit.abs()
    return M


def panel_label(ax, letter, dx=-0.10, dy=1.10):
    ts.panel_label(ax, letter, dx=dx, dy=dy)


def severity_axis(ax):
    ax.set_xticks(range(7))
    ax.set_xlabel("Severity (0 = clean)")


# ── metrics ───────────────────────────────────────────────────────────────
def boot_ci_metric(y, p, fn, n_boot=N_BOOT, seed=RNG_SEED):
    """Bootstrap 95% CI for a ranking metric by resampling images."""
    y = np.asarray(y); p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        vals.append(fn(yb, p[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def expected_calibration_error(y, p, n_bins=15):
    y = np.asarray(y); p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf = p[m].mean(); acc = y[m].mean()
        ece += (m.sum() / len(p)) * abs(acc - conf)
    return float(ece)


def brier(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean((p - y) ** 2))


def threshold_metrics(y, p, thr):
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec and sens and (prec + sens)) else np.nan
    balacc = np.nanmean([sens, spec])
    return dict(sensitivity=sens, specificity=spec, f1=f1, balanced_accuracy=balacc)


def clean_youden_threshold(R: pd.DataFrame | None = None) -> float:
    """Operating threshold fixed on the clean VALIDATION set (Youden's J).

    Determined once on the held-out validation predictions and held constant
    for all test-set evaluation, so the operating point is never tuned on the
    test data (no leakage). The argument ``R`` is accepted for backward
    compatibility but ignored."""
    from sklearn.metrics import roc_curve
    val = pd.read_csv(VAL_PREDICTIONS)
    fpr, tpr, thr = roc_curve(val.label_45, val.prob)
    return float(thr[np.argmax(tpr - fpr)])


def conditions():
    """Yield (degradation_type, severity, label) for all 31 conditions incl clean."""
    yield ("clean", 0)
    for d in DEG_ORDER:
        for s in range(1, 7):
            yield (d, s)


def save_fig(fig, name, appendix=False):
    d = FIG_APP if appendix else FIG_MAIN
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {'figures_appendix' if appendix else 'figures_main'}/{name}.pdf/.png")


def write_caption(name, text):
    (CAPS / f"{name}.txt").write_text(text.strip() + "\n", encoding="utf-8")
