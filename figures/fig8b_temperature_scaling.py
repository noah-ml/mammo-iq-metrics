#!/usr/bin/env python3
"""FIG 8b — Post-hoc calibration fit on VALIDATION, evaluated on held-out TEST.

Temperature scaling (T) and Platt scaling (a, b) are fit ONCE on the clean VALIDATION-set
logits (minimising binary NLL), then applied UNCHANGED to every clean and degraded TEST
condition. ECE/Brier are reported on the held-out test set. All transforms are monotonic,
so ROC-AUC and AUPRC are provably unchanged (verified numerically in main()).

Validation predictions come from tables/val_clean_predictions.csv, produced by
make_val_clean_predictions.py with the same checkpoint/preprocessing/AMP path as the
test run (957382). If that file is absent the script aborts (it will not silently fall
back to fitting on test).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, average_precision_score

import pubcommon as pc

R = pc.load_results()
VAL_CSV = pc.VAL_PREDICTIONS


def nll_at_T(T, logit, y):
    z = logit / T
    # stable BCE-with-logits
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def fit_temperature(logit, y):
    res = minimize_scalar(nll_at_T, bounds=(0.05, 20.0), args=(logit, y), method="bounded")
    return float(res.x)


def fit_platt(logit, y):
    """2-parameter logistic (Platt) scaling: p = sigmoid(a*logit + b). Adds an intercept,
    so it can correct the base-rate bias that a single temperature cannot."""
    from scipy.optimize import minimize

    def nll(params):
        a, b = params
        z = a * logit + b
        return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))

    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def ece(y, p, n_bins=15):
    return pc.expected_calibration_error(y, p, n_bins)


def reliability(ax, y, p, color, label):
    bins = np.linspace(0, 1, 11); xc, yc = [], []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < 9 else (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() > 0:
            xc.append(p[m].mean()); yc.append(y[m].mean())
    ax.plot(xc, yc, "o-", color=color, label=label)


def main():
    if not VAL_CSV.exists():
        sys.exit(f"ERROR: {VAL_CSV} not found. Run make_val_clean_predictions.py first "
                 "(calibration must be fit on the validation set, not on test).")

    # ── fit on clean VALIDATION ────────────────────────────────────────────
    val = pd.read_csv(VAL_CSV)
    if "logit" not in val.columns:
        sys.exit("val_clean_predictions.csv must contain a 'logit' column.")
    logit_v = val.logit.to_numpy(); y_v = val.label_45.to_numpy()
    T = fit_temperature(logit_v, y_v)
    a, b = fit_platt(logit_v, y_v)
    p_v_raw = 1.0 / (1.0 + np.exp(-logit_v))
    ece_val_raw = ece(y_v, p_v_raw)
    ece_val_temp = ece(y_v, 1.0 / (1.0 + np.exp(-logit_v / T)))
    ece_val_platt = ece(y_v, 1.0 / (1.0 + np.exp(-(a * logit_v + b))))
    print(f"  [fit on VALIDATION n={len(val)}, pos={int(y_v.sum())}]")
    print(f"  T={T:.4f}  Platt a={a:.4f} b={b:.4f}")
    print(f"  VAL ECE  uncal={ece_val_raw:.4f}  temp={ece_val_temp:.4f}  platt={ece_val_platt:.4f}")

    # ── evaluate on held-out TEST clean (for panel A + headline) ───────────
    cl = R[R.degradation_type == "clean"]
    logit_c = cl.logit.to_numpy(); y_c = cl.label_45.to_numpy()
    p_c_cal = 1.0 / (1.0 + np.exp(-logit_c / T))
    p_c_platt = 1.0 / (1.0 + np.exp(-(a * logit_c + b)))
    ece_before = ece(y_c, cl.prob.to_numpy())
    ece_after = ece(y_c, p_c_cal)
    ece_platt = ece(y_c, p_c_platt)
    print(f"  TEST clean ECE  uncal={ece_before:.4f}  temp={ece_after:.4f}  platt={ece_platt:.4f}")

    # ── recompute calibrated ECE/Brier per TEST condition (val-fit params) ──
    rows = []
    auc_ok = ap_ok = True
    for dtype, sev in pc.conditions():
        sub = R[(R.degradation_type == dtype) & (R.severity == sev)]
        y = sub.label_45.to_numpy(); lo = sub.logit.to_numpy()
        p_raw = sub.prob.to_numpy()
        p_cal = 1.0 / (1.0 + np.exp(-lo / T))
        p_platt = 1.0 / (1.0 + np.exp(-(a * lo + b)))
        # Monotonicity check: AUC/AP are invariant to any strictly-increasing transform
        # of the score. Use the logit itself as the rank-true reference (the stored prob
        # is rounded to 6 decimals in the CSV, which would create spurious tie-break diffs).
        auc_raw = roc_auc_score(y, lo); ap_raw = average_precision_score(y, lo)
        if not (np.isclose(auc_raw, roc_auc_score(y, p_cal), atol=1e-12)
                and np.isclose(auc_raw, roc_auc_score(y, p_platt), atol=1e-12)):
            auc_ok = False
        if not (np.isclose(ap_raw, average_precision_score(y, p_cal), atol=1e-12)
                and np.isclose(ap_raw, average_precision_score(y, p_platt), atol=1e-12)):
            ap_ok = False
        rows.append(dict(degradation_type=dtype, severity=sev, fit_on="validation",
                         T=T, platt_a=a, platt_b=b,
                         auc=auc_raw, ap=ap_raw,
                         ece_raw=ece(y, p_raw), ece_cal=ece(y, p_cal), ece_platt=ece(y, p_platt),
                         brier_raw=pc.brier(y, p_raw), brier_cal=pc.brier(y, p_cal),
                         brier_platt=pc.brier(y, p_platt)))
    tab = pd.DataFrame(rows)
    tab.to_csv(pc.TABLES / "calibration_temperature_scaled_table.csv", index=False)
    print("  wrote tables/calibration_temperature_scaled_table.csv")
    print(f"  AUC unchanged across all conditions: {auc_ok}   AUPRC unchanged: {ap_ok}")

    clean_row = tab[tab.degradation_type == "clean"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # A) reliability clean: uncalibrated vs temperature vs Platt
    ax = axes[0]
    reliability(ax, y_c, cl.prob.to_numpy(), "#d62728", f"uncalibrated (ECE={ece_before:.3f})")
    reliability(ax, y_c, p_c_cal, "#ff7f0e", f"temperature T={T:.2f} (ECE={ece_after:.3f})")
    reliability(ax, y_c, p_c_platt, "#2ca02c", f"Platt a={a:.2f},b={b:.2f} (ECE={ece_platt:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1.0, label="perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Held-out TEST clean reliability\n(params fit on validation)"); ax.legend(fontsize=7.5)
    pc.panel_label(ax, "A")

    # B) calibrated ECE vs severity (per degradation) with clean dashed
    ax = axes[1]
    for d in pc.DEG_ORDER:
        sub = tab[tab.degradation_type == d].sort_values("severity")
        xs = [0] + list(sub.severity); ys = [clean_row.ece_cal] + list(sub.ece_cal)
        ax.plot(xs, ys, marker=pc.DEG_MARKER[d], color=pc.DEG_COLOR[d], label=pc.DEG_LABEL[d])
    ax.axhline(clean_row.ece_cal, color="0.5", ls="--", lw=1.0)
    pc.severity_axis(ax); ax.set_ylabel("ECE (temperature-scaled)")
    ax.set_title("Calibrated ECE vs severity"); pc.panel_label(ax, "B")

    # C) calibrated Brier vs severity
    ax = axes[2]
    for d in pc.DEG_ORDER:
        sub = tab[tab.degradation_type == d].sort_values("severity")
        xs = [0] + list(sub.severity); ys = [clean_row.brier_cal] + list(sub.brier_cal)
        ax.plot(xs, ys, marker=pc.DEG_MARKER[d], color=pc.DEG_COLOR[d], label=pc.DEG_LABEL[d])
    ax.axhline(clean_row.brier_cal, color="0.5", ls="--", lw=1.0)
    pc.severity_axis(ax); ax.set_ylabel("Brier (temperature-scaled)")
    ax.set_title("Calibrated Brier vs severity"); pc.panel_label(ax, "C")
    axes[2].legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    pc.save_fig(fig, "FIG_calibration_temperature_scaled")

    cap = (
        f"Figure 8b. Post-hoc calibration fit on the validation set and evaluated on the held-out "
        f"test set. A single temperature T and Platt parameters (a, b) were fit ONCE on the clean "
        f"VALIDATION-set logits ({len(val)} images, {int(y_v.sum())} positive) by minimising binary "
        f"NLL, then applied UNCHANGED to every clean and degraded test condition. Fitted values: "
        f"T={T:.3f}; Platt a={a:.3f}, b={b:.3f}. (A) Reliability diagram on the held-out clean test "
        f"set: uncalibrated (ECE={ece_before:.3f}), temperature scaling p=sigmoid(logit/T) "
        f"(ECE={ece_after:.3f}), and Platt scaling p=sigmoid(a·logit+b) (ECE={ece_platt:.3f}). "
        f"Temperature scaling alone barely helps because the dominant miscalibration is a base-rate "
        f"bias from class-weighted training (pos_weight≈19): a single temperature rescales confidence "
        f"around p=0.5 but cannot shift the bias. Adding an intercept (Platt) corrects the base rate "
        f"and substantially lowers ECE. (B) Temperature-scaled ECE and (C) Brier versus severity on "
        f"the test set, one line per degradation (dashed = clean). Test set: {pc.N_TEST} images, "
        f"{pc.N_POS} positive. All transforms are monotonic, so ROC-AUC and AUPRC are unchanged on "
        f"every condition (verified numerically) — only probability calibration is affected. Even "
        f"after calibration, severe blur/JPEG2000/resolution degrade calibration markedly while "
        f"contrast stays near the clean baseline, so the calibration-robustness ranking matches the "
        f"discrimination-robustness ranking. Calibration parameters were fit on validation and "
        f"evaluated on the held-out test set (no test-set leakage into the calibration fit)."
    )
    pc.write_caption("FIG_calibration_temperature_scaled", cap)


if __name__ == "__main__":
    main()
