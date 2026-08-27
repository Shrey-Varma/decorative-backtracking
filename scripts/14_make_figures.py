#!/usr/bin/env python
"""Stage 14: publication figures.

Reads {analysis-dir}/{stat_tests,activation_shift}.json and
continuations_flat.csv (all produced by stages 06/09/13); writes four
figures as PDF (vector) + 300-dpi PNG:

  fig1_causal_interventions_forest  every intervention arm, Δ accuracy vs.
                                    matched reference, 90% cluster-bootstrap
                                    CI, ±0.10 equivalence margin, BF01
  fig2_sufficiency_insertion        insertion rescue rates by population
                                    (wrong_point vs failed_chain)
  fig3_activation_shift             marker-locked residual shift (real) vs.
                                    AUROC for answer movement (chance)
  fig4_summary_2x2                  necessity/sufficiency x text/activation
                                    summary schematic

Sign convention: stat_tests deltas are (reference - arm); we plot
(arm - reference) so positive = arm improves accuracy.

  python scripts/14_make_figures.py \
      --analysis-dir results/exp1/analysis --out-dir results/figures
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--analysis-dir", default=str(ROOT / "results/exp1/analysis"))
parser.add_argument("--out-dir", default=str(ROOT / "results/figures"))
args = parser.parse_args()
AN = Path(args.analysis_dir)
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

stats = json.loads((AN / "stat_tests.json").read_text())
shift = json.loads((AN / "activation_shift.json").read_text())
allsplit = stats["splits"]["all"]

# Okabe-Ito colorblind-safe palette
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "grey": "#8a8a8a", "dark": "#333333",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.dpi": 130,
    "savefig.dpi": 300,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Figure 1
# Forest plot: every causal intervention, Δ accuracy (arm − ref), 90% CI,
# ±0.10 equivalence margin.
def flip(d):
    lo, hi = d["boot90_ci"]
    return -d["delta_mean"], (-hi, -lo)

per = allsplit["per_arm"]
rows = []  # (label, delta, ci, family, bf01)

# All ten SAE-family arms, because the family-wise claim in the text is about
# all ten; a curated subset would not show that the whole family sits on zero.
fam_necS = [
    ("SAE ablate (top feature)", "ablate_correction"),
    ("SAE ablate (candidate #2)", "ablate_correction/cand=12002"),
    ("SAE ablate (candidate #3)", "ablate_correction/cand=38107"),
    ("SAE ablate (candidate #4)", "ablate_correction/cand=59368"),
    ("SAE ablate (candidate #5)", "ablate_correction/cand=62649"),
    ("SAE ablate (joint, 5 feats)", "ablate_correction/joint"),
    ("SAE ablate (event-span only)$^\\dagger$", "ablate_correction/mask=event_only"),
    ("SAE ablate (mean mode)", "ablate_correction/mode=mean"),
    ("Control feature", "ablate_control_feature"),
    ("Control position", "ablate_correction_at_control_pos"),
]
fam_dir = [
    ("Direction ablate, L15", "ablate_direction/layer=15"),
    ("Direction ablate, L19", "ablate_direction/layer=19"),
    ("Direction ablate, L23", "ablate_direction/layer=23"),
]
fam_text = [
    ("Excise → neutral filler", "text_neutral_filler"),
    ("Keep correction (re-sample)", "text_with_correction"),
]
for lbl, k in fam_necS:
    rows.append((lbl, *flip(per[k]), "Necessity · SAE feature", per[k].get("bf01_jzs")))
for lbl, k in fam_dir:
    # This family is the one that is NOT null, so its BF01 lands below 1
    # (evidence for an effect). Blanking the cell reads as missing data, so we
    # print it and explain the sub-1 reading in the figure footnote. The effect
    # is positive: faithfulness predicts ablation HURTS, so the sign is wrong
    # for faithfulness either way.
    rows.append((lbl, *flip(per[k]), "Necessity · direction", per[k].get("bf01_jzs")))
for lbl, k in fam_text:
    d = allsplit["text_excision"][k]
    rows.append((lbl, *flip(d), "Necessity · text excision", d.get("bf01_jzs")))
ins_lbl = {
    "insert_correction/v=assert_mistake": 'Insert “I made a mistake…”',
    "insert_correction/v=minimal_marker": 'Insert “Hmm, wait.”',
    "insert_correction/v=soft_recheck": 'Insert “let me double-check…”',
    "insert_filler": "Insert filler (control)",
}
for k, lbl in ins_lbl.items():
    d = allsplit["text_insertion"][k]
    rows.append((lbl, *flip(d), "Sufficiency · text insertion", d.get("bf01_jzs")))
suf_lbl = {
    "clamp_on_feature/scale=1": "Clamp feature ON (1×)",
    "clamp_on_feature/scale=2": "Clamp feature ON (2×)",
    "add_direction/layer=19/scale=1": "Add direction, L19 (1×)",
    "add_direction/layer=19/scale=2": "Add direction, L19 (2×)",
}
for k, lbl in suf_lbl.items():
    d = allsplit["hooked_sufficiency"][k]
    rows.append((lbl, *flip(d), "Sufficiency · activation", d.get("bf01_jzs")))

_DIR_P = allsplit["familywise_maxT_direction_family"]["p_familywise"]

fam_color = {
    "Necessity · SAE feature": C["blue"],
    "Necessity · direction": C["sky"],
    "Necessity · text excision": C["green"],
    "Sufficiency · text insertion": C["orange"],
    "Sufficiency · activation": C["purple"],
}
is_control = {"Control feature", "Control position", "Insert filler (control)"}

# Layout: the data region is reserved for data. Family headers sit in the
# left margin, BF01 gets its own labelled column on the right, and the two
# explanatory notes live in empty plot space rather than on top of a CI.
X_LO, X_HI = -0.29, 0.19
X_BF = 0.215                      # BF column, outside the data region
FIG_W, AX_L, AX_R = 5.5, 0.315, 0.885
fig, ax = plt.subplots(figsize=(FIG_W, 5.3))
fig.subplots_adjust(left=AX_L, right=AX_R, top=0.935, bottom=0.095)


def left_of_axes(pt):
    """Axes-fraction x for a point `pt` typographic points left of the spine.

    Arm labels and family headers are both placed through this, so they hang
    from one left edge; right-aligned tick labels under a left-aligned header
    is the misalignment this replaces.
    """
    return -pt / (FIG_W * (AX_R - AX_L) * 72)


X_LBL, X_HDR = left_of_axes(116), left_of_axes(122)
ax.axvspan(-0.10, 0.10, color="0.94", zorder=0)
ax.axvline(0, color="0.4", lw=0.8, zorder=1)
for xm in (-0.10, 0.10):
    ax.axvline(xm, color="0.75", lw=0.8, ls="--", zorder=1)

ypos, ylabels, yfam = [], [], []
y = 0
last_fam = None
for lbl, delta, ci, fam, bf in rows:
    if fam != last_fam and last_fam is not None:
        y -= 1.0                   # family gap, room for the header
    if fam != last_fam:
        yfam.append((y, fam))
        last_fam = fam
    col = fam_color[fam]
    open_marker = lbl in is_control
    ax.plot(ci, [y, y], color=col, lw=1.8, zorder=2, solid_capstyle="round")
    ax.plot(delta, y, "o", ms=5.8, color="white" if open_marker else col,
            mec=col, mew=1.5, zorder=3)
    if bf is not None:
        txt = f"{bf:.2f}" if bf < 1 else f"{bf:.1f}"
        ax.annotate(txt, (X_BF, y), va="center", ha="center", fontsize=7.2,
                    color=col if fam == "Necessity · direction" else "0.35",
                    annotation_clip=False)
    ypos.append(y)
    ylabels.append(lbl)
    y -= 1

for fy, fam in yfam:
    ax.annotate(fam, (X_HDR, fy + 0.85), xycoords=ax.get_yaxis_transform(),
                fontsize=7.7, fontweight="bold", color=fam_color[fam],
                annotation_clip=False, ha="left", va="center")

ax.annotate("BF$_{01}$", (X_BF, 1.15), ha="center", va="center", fontsize=7.6,
            color="0.35", fontweight="bold", annotation_clip=False)

ax.annotate("equivalence margin $\\pm$0.10", (0, 1.15), ha="center",
            va="center", fontsize=7.2, color="0.45")

ax.set_yticks(ypos)
ax.set_yticklabels([])
for yv, lbl in zip(ypos, ylabels):
    ax.annotate(lbl, (X_LBL, yv), xycoords=ax.get_yaxis_transform(),
                fontsize=7.4, ha="left", va="center", annotation_clip=False)
ax.set_xlim(X_LO, X_HI)
ax.set_ylim(min(ypos) - 0.9, 1.7)
ax.set_xticks([-0.2, -0.1, 0.0, 0.1])
ax.set_xlabel("$\\Delta$ final-answer accuracy vs. matched reference"
              "   (90% cluster-bootstrap CI)", fontsize=8.6, labelpad=8)
fig.text(0.5, -0.004,
         "$\\dagger$ bit-identical to natural in 56.9% of its continuations, so "
         "its interval is tighter than its evidence.\n"
         "BF$_{01}$ below 1 favours an effect. Direction ablation is the one "
         "family that is not null, and it RAISES accuracy\n(family-wise "
         f"$p$ = {_DIR_P:.3f}), the opposite sign to the faithfulness prediction.",
         fontsize=6.2, color="0.45", ha="center", va="top", linespacing=1.6)
ax.set_title("Neither necessary nor sufficient for answer revision",
             pad=14, fontsize=9.4)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=8)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_bounds(X_LO, X_HI)
save(fig, "fig1_causal_interventions_forest")

# ------------------------------------------------------- Figure 1, compact
# Same 23 arms, collapsed to one row per family: a light span covering the
# union of the family's CIs, with every arm's point estimate over it. Keeps
# the "the whole family sits on zero" evidence that a curated subset would
# lose, at a third of the height. Per-arm numbers live in the appendix table.
FAM_P = {
    "Necessity · SAE feature": allsplit["familywise_maxT_sae_family"]["p_familywise"],
    "Necessity · direction": allsplit["familywise_maxT_direction_family"]["p_familywise"],
    "Sufficiency · text insertion":
        allsplit["familywise_maxT_insertion"]["p_familywise"],
    "Sufficiency · activation":
        allsplit["familywise_maxT_hooked_sufficiency"]["p_familywise"],
}
fams = []
for fam in fam_color:
    r = [x for x in rows if x[3] == fam]
    ds = [x[1] for x in r]
    bfs = [x[4] for x in r if x[4] is not None]
    fams.append((fam, ds, min(x[2][0] for x in r), max(x[2][1] for x in r),
                 bfs, len(r)))

figc, axc = plt.subplots(figsize=(FIG_W, 2.15))
figc.subplots_adjust(left=0.305, right=0.885, top=0.845, bottom=0.235)
axc.axvspan(-0.10, 0.10, color="0.94", zorder=0)
axc.axvline(0, color="0.4", lw=0.8, zorder=1)
for xm in (-0.10, 0.10):
    axc.axvline(xm, color="0.75", lw=0.8, ls="--", zorder=1)


def left_of_axes_c(pt):
    return -pt / (FIG_W * (0.885 - 0.305) * 72)


for i, (fam, ds, lo, hi, bfs, n) in enumerate(fams):
    y, col = -i, fam_color[fam]
    axc.plot([lo, hi], [y, y], color=col, lw=8.5, alpha=0.22,
             solid_capstyle="butt", zorder=2)
    axc.plot(ds, [y] * len(ds), "o", ms=3.9, color=col, mec="white", mew=0.5,
             zorder=3)
    axc.annotate(f"{fam}  ({n})", (left_of_axes_c(112), y),
                 xycoords=axc.get_yaxis_transform(), fontsize=7.6,
                 fontweight="bold", color=col, ha="left", va="center",
                 annotation_clip=False)
    if bfs:
        rng_txt = (f"{min(bfs):.2f}" if min(bfs) < 1 else f"{min(bfs):.1f}") + \
                  "–" + (f"{max(bfs):.1f}")
        axc.annotate(rng_txt, (X_BF, y), fontsize=7.0, ha="center", va="center",
                     color=col if fam == "Necessity · direction" else "0.35",
                     annotation_clip=False)
    fp = FAM_P.get(fam)
    if fp is not None:
        axc.annotate(f"family $p$ = {fp:.3f}", (left_of_axes_c(112), y - 0.34),
                     xycoords=axc.get_yaxis_transform(), fontsize=6.2,
                     color="0.45", ha="left", va="center", annotation_clip=False)
axc.annotate("BF$_{01}$ range", (X_BF, 0.92), fontsize=7.0, ha="center",
             va="center", color="0.35", fontweight="bold", annotation_clip=False)
axc.annotate("equivalence margin $\\pm$0.10", (0, 0.92), ha="center", va="center",
             fontsize=7.0, color="0.45")
axc.set_yticks([])
axc.set_xlim(X_LO, X_HI)
axc.set_ylim(-len(fams) + 0.35, 1.3)
axc.set_xticks([-0.2, -0.1, 0.0, 0.1])
axc.tick_params(axis="x", labelsize=7.8)
# one 95-char line overruns a 230pt-wide axes; two lines keep it inside the
# figure and off the neighbouring text.
axc.set_xlabel("$\\Delta$ final-answer accuracy vs. matched reference\n"
               "90% cluster-bootstrap CI, union over family", fontsize=8.0,
               labelpad=4)
axc.spines["left"].set_visible(False)
axc.set_title("Neither necessary nor sufficient for answer revision",
              fontsize=9.4, pad=9)
save(figc, "fig1b_forest_compact")

# ---------------------------------------------------------------- Figure 2
# Sufficiency by population: wrong_point (ceiling-bounded) vs failed_chain.
# The failed-chain panel plots the MATCHER-CORRECTED population (chains that
# are genuinely never rescued); the uncorrected population is contaminated by
# answer-matcher false negatives and is not the paper's estimand.
# continuations_flat.csv is 162 MB and not in the code release; stage 17
# writes the five columns this figure needs to release/insertion_rows.csv.
_flat = AN / "continuations_flat.csv"
_slice = AN.parent / "release/insertion_rows.csv"
if _flat.exists():
    df = pd.read_csv(_flat, usecols=["chain_id", "condition", "event_id",
                                     "meta", "is_correct"])
else:
    df = pd.read_csv(_slice)
ins = df[df.condition.str.startswith("insert_")].copy()
ins["population"] = ins.meta.map(lambda s: ast.literal_eval(s).get("population"))
del df
CLEAN = set(stats["failed_chain_purity"]["clean_chain_ids"])

ARMS = [
    ("insert_natural", "Natural", C["grey"]),
    ("insert_correction/v=assert_mistake", '"I made a\nmistake\u2026"', C["orange"]),
    ("insert_correction/v=soft_recheck", '"Let me\ndouble-\ncheck\u2026"', C["orange"]),
    ("insert_correction/v=minimal_marker", '"Hmm,\nwait."', C["orange"]),
    ("insert_filler", "Filler\n(control)", C["green"]),
]
rng = np.random.default_rng(0)


def boot_ci(g, n_boot=2000):
    """Cluster bootstrap over events of mean accuracy."""
    ev = [grp.is_correct.values for _, grp in g.groupby("event_id")]
    means = np.array([np.concatenate([ev[i] for i in
                     rng.integers(0, len(ev), len(ev))]).mean()
                     for _ in range(n_boot)])
    return np.percentile(means, [5, 95])


fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.85))
fig.subplots_adjust(wspace=0.28)
for ax, pop, title, note in [
    (axes[0], "wrong_point",
     "Self-correcting chains\n(truncated at a wrong intermediate answer)",
     "recovers with or without the marker"),
    (axes[1], "failed_chain",
     "Failed chains, matcher-corrected\n(never rescued; no recovery ceiling)",
     "no rescue from any phrasing"),
]:
    sub = ins[ins.population == pop]
    if pop == "failed_chain":
        sub = sub[sub.chain_id.isin(CLEAN)]
    tops = []
    for i, (arm, lbl, col) in enumerate(ARMS):
        g = sub[sub.condition == arm]
        acc = g.is_correct.mean()
        lo, hi = boot_ci(g)
        tops.append(hi)
        ax.bar(i, acc, width=0.62, color=col, alpha=0.85, zorder=2)
        ax.errorbar(i, acc, yerr=[[max(acc - lo, 0)], [max(hi - acc, 0)]], fmt="none",
                    ecolor=C["dark"], lw=1.1, capsize=2.5, zorder=3)
        ax.annotate(f"{acc*100:.1f}", (i, hi), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7.0,
                    zorder=4)
    nat = sub[sub.condition == "insert_natural"].is_correct.mean()
    ax.axhline(nat, color="0.35", lw=0.9, ls=":", zorder=1)
    # headroom for the value labels AND the italic note above them
    ax.set_ylim(0, max(tops) * 1.42)
    ax.set_xlim(-0.62, len(ARMS) - 0.38)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a[1] for a in ARMS], fontsize=6.4)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=8.2, pad=15)
    ax.annotate(note, (0.5, 1.005), xycoords="axes fraction", ha="center",
                va="bottom", fontsize=6.8, style="italic", color="0.4")
for _ax in axes:
    _ax.set_ylabel("Final-answer accuracy", fontsize=8.4)
    _ax.tick_params(axis="y", labelsize=7.6)
fig.suptitle("Inserting correction language does not cause correction", y=1.09,
             fontsize=9.8)
save(fig, "fig2_sufficiency_insertion")

# ---------------------------------------------------------------- Figure 3
# Internal state: real shift, but decoupled from answer changes.
W = "16"
layers = ["15", "19", "23"]
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.75))
fig.subplots_adjust(wspace=0.42)

ax = axes[0]
x = np.arange(len(layers))
ev = [shift["per_layer"][L][W]["mean_event_shift"] for L in layers]
ct = [shift["per_layer"][L][W]["mean_control_shift"] for L in layers]
# stage 13 renamed this key; runs predating the rename only carry the
# deprecated alias, and the two hold the same number (paired d_z).
def _dz(rec):
    return rec.get("cohens_dz", rec.get("cohens_d"))


ds = [_dz(shift["per_layer"][L][W]["all"]) for L in layers]
b1 = ax.bar(x - 0.18, ct, width=0.34, color=C["grey"], alpha=0.85,
            label="Matched sentence-start control")
b2 = ax.bar(x + 0.18, ev, width=0.34, color=C["blue"], alpha=0.9,
            label="Correction marker")
for xi, (e, c, d) in enumerate(zip(ev, ct, ds)):
    ax.annotate(f"$d_z$={d:.2f}\np<0.001", (xi + 0.18, e), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=6.4,
                color=C["blue"])
ax.set_xticks(x)
ax.set_xticklabels([f"Layer {L}" for L in layers])
ax.set_ylabel("Residual-stream shift\n1 − cos(before, after)", fontsize=8.2)
ax.set_ylim(0, 0.78)
ax.set_title("Markers coincide with a real\ninternal state change", fontsize=8.4)
ax.legend(fontsize=6.2, loc="upper left", handlelength=1.1,
          borderaxespad=0.2, frameon=False)

ax = axes[1]
metrics = [
    ("direction_projection_wrong_to_right", "Direction projection", C["purple"]),
    ("shift_magnitude_wrong_to_right", "Shift magnitude", C["red"]),
]
for j, (key, lbl, col) in enumerate(metrics):
    vals = [shift["auroc"][L][key] for L in layers]
    ax.plot(x + (j - 0.5) * 0.12, vals, "o", ms=6, color=col, label=lbl)
ax.axhline(0.5, color="0.4", lw=0.9, ls="--")
ax.annotate("chance", (2.35, 0.503), fontsize=7.5, color="0.4")
ax.set_xticks(x)
ax.set_xticklabels([f"Layer {L}" for L in layers])
ax.set_ylabel("AUROC: shift predicts\nwrong→right revision", fontsize=8.2)
ax.set_ylim(0.35, 0.65)
ax.set_xlim(-0.4, 2.75)
ax.set_title("…but the shift does not predict\nwhether the answer changes", fontsize=8.4)
ax.legend(fontsize=6.2, loc="upper left", frameon=False)
for _ax in axes:
    _ax.tick_params(labelsize=7.6)
fig.suptitle("Backtracking markers: genuine state transition, decoupled from revision",
             y=1.07, fontsize=9.6)
save(fig, "fig3_activation_shift")

# ---------------------------------------------------------------- Figure 4
# Summary schematic: 2x2 necessity/sufficiency x text/activation.
fig, ax = plt.subplots(figsize=(5.5, 2.8))
ax.set_axis_off()
# Numbers are READ FROM stat_tests.json, never typed in: a hardcoded summary
# figure silently goes stale the moment the analysis is re-run. Sign convention
# matches Table 2 and fig1 — Δ = arm − reference, positive = arm helps.
_ins = allsplit["text_insertion"]
_hook = allsplit["hooked_sufficiency"]
_corr = [v for k, v in _ins.items() if k.startswith("insert_correction")]
_exc = allsplit["text_excision"]["text_with_correction"]
_sae = allsplit["per_arm"]["ablate_correction"]
_saep = allsplit["familywise_maxT_sae_family"]["p_familywise"]
_dir = [allsplit["per_arm"][k] for k in allsplit["per_arm"] if k.startswith("ablate_direction")]
_dirp = allsplit["familywise_maxT_direction_family"]["p_familywise"]


def _fmt(x, sign=True):
    return f"{-x:+.3f}".replace("-", "−") if sign else f"{x:.2f}"


cells = [
    (0, 1, "Text level", "Excise correction sentence",
     "accuracy unchanged",
     f"Δ = {_fmt(_exc['delta_mean'])} · BF$_{{01}}$ = {_exc['bf01_jzs']:.1f}"),
    (1, 1, "Text level", "Insert correction phrases",
     "no rescue (3 phrasings)",
     f"|Δ| ≤ {max(abs(v['delta_mean']) for v in _corr):.3f} · "
     f"BF$_{{01}}$ ≥ {min(v['bf01_jzs'] for v in _corr):.1f}"),
    (0, 0, "Activation level", "Ablate SAE feature",
     "accuracy unchanged",
     f"Δ = {_fmt(_sae['delta_mean'])} · max-T p = {_saep:.2f}"),
    (1, 0, "Activation level", "Clamp feature ON / add direction",
     "no induced correction†",
     f"|Δ| ≤ {max(abs(v['delta_mean']) for v in _hook.values()):.3f} · "
     f"BF$_{{01}}$ ≥ {min(v['bf01_jzs'] for v in _hook.values()):.1f}"),
]
for cx, cy, lvl, test, res, num in cells:
    x0, y0 = 0.04 + cx * 0.49, 0.17 + cy * 0.405
    ax.add_patch(mpl.patches.FancyBboxPatch(
        (x0, y0), 0.44, 0.345, boxstyle="round,pad=0.012",
        fc="#f4f6f8", ec="0.7", lw=0.9, transform=ax.transAxes))
    ax.text(x0 + 0.22, y0 + 0.255, test, ha="center", fontsize=8.6,
            fontweight="bold", transform=ax.transAxes)
    ax.text(x0 + 0.22, y0 + 0.16, res, ha="center", fontsize=8.4,
            color=C["red"], transform=ax.transAxes)
    ax.text(x0 + 0.22, y0 + 0.07, num, ha="center", fontsize=7.6,
            color="0.35", transform=ax.transAxes)
ax.text(0.26, 0.975, "NOT NECESSARY", ha="center", fontsize=9.5,
        fontweight="bold", color=C["blue"], transform=ax.transAxes)
ax.text(0.75, 0.975, "NOT SUFFICIENT", ha="center", fontsize=9.5,
        fontweight="bold", color=C["orange"], transform=ax.transAxes)
ax.text(0.51, 0.095, "† dose-limited: interventions did not raise marker emission; "
        "text-level insertion carries the sufficiency claim",
        ha="center", fontsize=6.8, color="0.45", transform=ax.transAxes)
_dlo = min(abs(v["delta_mean"]) for v in _dir)
_dhi = max(abs(v["delta_mean"]) for v in _dir)
ax.text(0.51, -0.005, "Not shown: the SAE-free direction ablation is the one family that is "
        f"not null. It RAISES accuracy\n(Δ = +{_dlo:.2f} to +{_dhi:.2f}, family p = {_dirp:.3f}), "
        "the opposite sign to the faithfulness prediction.",
        ha="center", va="top", fontsize=6.8, color="0.45", linespacing=1.5,
        transform=ax.transAxes)
save(fig, "fig4_summary_2x2")

print("done ->", OUT)
