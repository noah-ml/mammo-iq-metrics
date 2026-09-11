#!/usr/bin/env python3
"""Regenerate FIG_auc_auprc_vs_severity (study-level CI bands) and
FIG_delta_auc_auprc_forest (paired study-level delta CIs), B=2000, seed 42,
so that the figure uncertainty matches the study-level bootstrap used for the
Chapter-5 tables. Verifies against the reported clean/JPEG2000-s6 CIs."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import pubcommon as pc

MAP = pc.ROOT / "master_splits_1024x384_complete.csv"
OUTDIR = pc.FIG_MAIN
B, SEED = 2000, 42

R = pc.load_results()
m = pd.read_csv(MAP)[["image_id", "study_id"]]
m["image_id"] = m["image_id"].astype(str)
R = R.merge(m, on="image_id", how="left")

clean = R[R.degradation_type == "clean"].set_index("image_id")
order = clean.index.to_numpy()
y = clean.label_45.to_numpy()
p_clean = clean.prob.to_numpy()
studies = clean.study_id.to_numpy()
uniq = np.unique(studies)
pos_by_study = {s: np.where(studies == s)[0] for s in uniq}

# Precompute the B study-level bootstrap index arrays ONCE (fixed seed),
# and reuse them for every condition -> avoids re-concatenating 244k times.
_rng = np.random.default_rng(SEED)
BOOT = []
for _ in range(B):
    pick = uniq[_rng.integers(0, len(uniq), len(uniq))]
    idx = np.concatenate([pos_by_study[s] for s in pick])
    if y[idx].sum() in (0, len(idx)):
        continue
    BOOT.append(idx)
YB = [y[idx] for idx in BOOT]               # cache resampled labels
PCB = [p_clean[idx] for idx in BOOT]        # cache resampled clean probs

def cond_prob(dtype, sev):
    sub = R[(R.degradation_type == dtype) & (R.severity == sev)].set_index("image_id")
    return sub.reindex(order).prob.to_numpy()

def ci_metric(p, fn):
    v = [fn(YB[b], p[BOOT[b]]) for b in range(len(BOOT))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def ci_delta(p_deg, fn):
    v = [fn(YB[b], p_deg[BOOT[b]]) - fn(YB[b], PCB[b]) for b in range(len(BOOT))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

# ---- assemble per-condition point + study-level CI ----
rows = {}
auc0 = roc_auc_score(y, p_clean); ap0 = average_precision_score(y, p_clean)
rows[("clean", 0)] = dict(auc=auc0, ap=ap0,
                          auc_ci=ci_metric(p_clean, roc_auc_score),
                          ap_ci=ci_metric(p_clean, average_precision_score))
for d in pc.DEG_ORDER:
    for s in range(1, 7):
        p = cond_prob(d, s)
        rows[(d, s)] = dict(
            auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
            auc_ci=ci_metric(p, roc_auc_score), ap_ci=ci_metric(p, average_precision_score),
            dauc=roc_auc_score(y, p) - auc0, dap=average_precision_score(y, p) - ap0,
            dauc_ci=ci_delta(p, roc_auc_score), dap_ci=ci_delta(p, average_precision_score))

import pickle as _pkl
_pkl.dump((rows, auc0, ap0), open(SCRIPTS / "_studylevel_rows.pkl", "wb"))
# ---- sanity vs Ch5 tables ----
c = rows[("clean", 0)]
print("clean AUC %.4f CI[%.3f,%.3f]  (Ch5: [0.780,0.866])" % (c["auc"], *c["auc_ci"]))
print("clean AP  %.4f CI[%.3f,%.3f]  (Ch5: [0.352,0.529])" % (c["ap"], *c["ap_ci"]))
j = rows[("jpeg2000", 6)]
print("jpeg2000 s6 dAUC %.3f CI[%.3f,%.3f]  (Ch5: -0.170 [-0.221,-0.120])" % (j["dauc"], *j["dauc_ci"]))
print("jpeg2000 s6 dAP  %.3f CI[%.3f,%.3f]  (Ch5: -0.314 [-0.393,-0.230])" % (j["dap"], *j["dap_ci"]))

# ---- FIG 3: AUC & AP vs severity, study-level bands ----
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(pc.FIGW, 4.5))
    specs = [("auc", "auc_ci", "ROC-AUC", auc0), ("ap", "ap_ci", "Average precision (AUPRC)", ap0)]
    for ax, (mk, cik, ylab, clean_val) in zip(axes, specs):
        for d in pc.DEG_ORDER:
            xs = [0] + list(range(1, 7))
            ys = [rows[("clean", 0)][mk] if s == 0 else rows[(d, s)][mk] for s in xs]
            lo = [rows[("clean", 0)][cik][0] if s == 0 else rows[(d, s)][cik][0] for s in xs]
            hi = [rows[("clean", 0)][cik][1] if s == 0 else rows[(d, s)][cik][1] for s in xs]
            ax.plot(xs, ys, marker=pc.DEG_MARKER[d], color=pc.DEG_COLOR[d], label=pc.DEG_LABEL[d])
            ax.fill_between(xs, lo, hi, color=pc.DEG_COLOR[d], alpha=0.13, lw=0)
        ax.axhline(clean_val, color="0.4", ls="--", lw=1.1, label="clean baseline")
        ax.axhline(clean_val - 0.05, color="0.6", ls=":", lw=1.0, label="baseline - 0.05")
        # sizes from thesisstyle (labels 10 pt, ticks 9.5 pt); the explicit
        # 11/10 pt that used to be here made this figure a step above the rest
        pc.severity_axis(ax); ax.set_ylabel(ylab)
    axes[1].yaxis.set_label_position("right"); axes[1].yaxis.tick_right()
    pc.panel_label(axes[0], "A"); pc.panel_label(axes[1], "B", dx=-0.045)
    # shared legend below both panels: at 6.10 in an in-axes 7-entry legend
    # would either overlap the curves or need a much smaller font
    h, l = axes[0].get_legend_handles_labels()
    # 3 columns, not 4: at 4 the last (longest) entry runs past the figure edge
    # 2-row shared legend (5 degradations + 2 baselines) at the standard 9.5 pt
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=pc.ts.SMALL,
               bbox_to_anchor=(0.5, 0.012), handlelength=1.3, columnspacing=0.8)
    fig.subplots_adjust(left=0.11, right=0.905, top=0.900, bottom=0.26, wspace=0.06)  # top was 0.935: the 11 pt panel letters overran the canvas
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"FIG_auc_auprc_vs_severity.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig); print("wrote FIG_auc_auprc_vs_severity")

# ---- FIG 4: paired delta forest at sev6, study-level paired CIs ----
def fig4():
    s6 = [(d, rows[(d, 6)]) for d in pc.DEG_ORDER]
    # stacked rather than side by side: the y-tick labels are degradation
    # names and need the full text width to stay legible at 9.5 pt
    fig, axes = plt.subplots(2, 1, figsize=(pc.FIGW, 4.3))
    for ax, mk, cik, lab in [(axes[0], "dauc", "dauc_ci", "$\\Delta$AUC (degraded $-$ clean) at severity 6"),
                             (axes[1], "dap", "dap_ci", "$\\Delta$AUPRC (degraded $-$ clean) at severity 6")]:
        s = sorted(s6, key=lambda t: t[1][mk])  # most negative at bottom
        ypos = np.arange(len(s))
        vals = [r[mk] for _, r in s]
        elo = [r[mk] - r[cik][0] for _, r in s]
        ehi = [r[cik][1] - r[mk] for _, r in s]
        ax.errorbar(vals, ypos, xerr=[elo, ehi], fmt="o", color="#333", ecolor="#888", capsize=2.5, ms=4)
        for yi, (d, r) in zip(ypos, s):
            ax.plot(r[mk], yi, "o", color=pc.DEG_COLOR[d], ms=6, zorder=5)
        ax.set_yticks(ypos); ax.set_yticklabels([pc.DEG_LABEL[d] for d, _ in s])
        ax.axvline(0, color="0.3", lw=1.1); ax.axvline(-0.05, color="0.6", ls=":", lw=1.0)
        ax.set_xlabel(lab)
    pc.panel_label(axes[0], "A"); pc.panel_label(axes[1], "B")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"FIG_delta_auc_auprc_forest.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig); print("wrote FIG_delta_auc_auprc_forest")

fig3(); fig4()
