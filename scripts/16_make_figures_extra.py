#!/usr/bin/env python
"""Stage 16: the three figures that need raw trajectory/event data.

Kept out of stage 14 because the localization pass reads trajectories.jsonl
and events.jsonl end to end (~2.5M token positions), which would make every
routine figure rebuild slow.

  fig5_localization        where wrong->right revisions actually happen,
                           relative to the correction marker that claims them.
                           Uses the STRIDE-ONLY probe grid: see the note on the
                           event-boundary measurement bias below.
  fig6_partial_id          what the nulls exclude once selection purity is
                           accounted for, vs the pre-registered threshold
  fig7_worked_example      one real event: correction span, natural vs
                           ablated continuation, same final answer

  python scripts/16_make_figures_extra.py --run-dir results/exp1 \
      --out-dir ../paper/correction_faithfulness/figures
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.answers import answers_equal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default=str(ROOT / "results/exp1"))
ap.add_argument("--out-dir", default=str(ROOT / "results/figures"))
args = ap.parse_args()
RD, OUT = Path(args.run_dir), Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
STATS = json.loads((RD / "analysis/stat_tests.json").read_text())

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
     "grey": "#8a8a8a", "dark": "#333333"}
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "figure.dpi": 130, "savefig.dpi": 300,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def jl(p):
    with (RD / p).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------- Figure 5
# Where revisions happen relative to the marker that narrates them.
#
# A flip is only localized to the probe interval it was detected in, so its
# mass is spread uniformly over that interval's tokens rather than pinned to a
# point. That is the TOKEN-uniform view, which is the null the paper prefers:
# a longer interval has more room for the answer to move.
#
# IMPORTANT: this drops every probe point that sits on an event boundary and
# keeps only the stride-32 grid. Truncating a chain exactly at a marker onset
# lands on a clause boundary, and the forced answer read there is correct
# +2.0 points more often than at a position-matched mid-sentence stride point
# (up to +7.6 points in the middle deciles). Flips detected in the interval
# ENDING at an onset therefore inherit a measurement bias that manufactures
# apparent wrong->right movement immediately before the marker: on the full
# grid the pre-marker bin reads 1.90x, on the uniform grid 1.42x. Restricting
# to stride points costs resolution near markers and 160 of 1797 flips, and
# buys an estimator whose probe points are all measured the same way.
MAXD, BIN = 300, 15
edges = np.arange(-MAXD, MAXD + BIN, BIN)
ctr = edges[:-1] + BIN / 2
NB = len(ctr)
ABSMAX = 400

onsets, spans, boundary = {}, [], {}
for d in jl("events.jsonl"):
    onsets.setdefault(d["chain_id"], []).append(d["onset"])
    boundary.setdefault(d["chain_id"], set()).update((d["onset"], d["end"]))
    spans.append(d["end"] - d["onset"])
onsets = {k: np.array(sorted(v)) for k, v in onsets.items()}
# chains.jsonl is 61 MB and not in the code release; only the gold answers
# are needed here, and stage 17 writes them to release/gold_answers.json.
if (RD / "chains.jsonl").exists():
    gold = {d["chain_id"]: d["gold_answer"] for d in jl("chains.jsonl")}
else:
    gold = json.loads((RD / "release/gold_answers.json").read_text())
MEAN_SPAN = float(np.mean(spans))


def nearest_signed(pos, ons):
    """Signed token distance to the nearest onset (positive = after)."""
    i = np.searchsorted(ons, pos)
    lo = ons[np.clip(i - 1, 0, len(ons) - 1)]
    hi = ons[np.clip(i, 0, len(ons) - 1)]
    dlo, dhi = pos - lo, pos - hi
    return np.where(np.abs(dlo) <= np.abs(dhi), dlo, dhi)


per_chain = []   # (signed expo hist, signed flip hist, abs expo hist, abs flip)
n_flips = 0
for t in jl("trajectories.jsonl"):
    ons, g = onsets.get(t["chain_id"]), gold.get(t["chain_id"])
    if ons is None or g is None or len(ons) == 0:
        continue
    bad = boundary.get(t["chain_id"], set())
    pts = [p for p in t["points"] if p["position"] not in bad]
    if len(pts) < 2:
        continue
    P = np.array([p["position"] for p in pts])
    dom = np.arange(P[0] + 1, P[-1] + 1)
    if dom.size == 0:
        continue
    de = nearest_signed(dom, ons)
    fd, fw = [], []
    for a, b in zip(pts, pts[1:]):
        if a["answer"] is None or answers_equal(a["answer"], g):
            continue
        if not answers_equal(b["answer"], g) or answers_equal(a["answer"], b["answer"]):
            continue
        seg = np.arange(a["position"] + 1, b["position"] + 1)
        if seg.size == 0:
            continue
        n_flips += 1
        fd.append(nearest_signed(seg, ons))
        fw.append(np.full(seg.size, 1.0 / seg.size))
    fd = np.concatenate(fd) if fd else np.zeros(0)
    fw = np.concatenate(fw) if fw else np.zeros(0)
    per_chain.append((
        np.histogram(de, bins=edges)[0].astype(float),
        np.histogram(fd, bins=edges, weights=fw)[0],
        np.bincount(np.clip(np.abs(de), 0, ABSMAX).astype(int), minlength=ABSMAX + 1).astype(float),
        np.bincount(np.clip(np.abs(fd), 0, ABSMAX).astype(int), weights=fw, minlength=ABSMAX + 1),
    ))
print(f"  {len(per_chain)} chains, {n_flips} wrong->right flips")

SE = np.array([c[0] for c in per_chain])
SF = np.array([c[1] for c in per_chain])
AE = np.array([c[2] for c in per_chain])
AF = np.array([c[3] for c in per_chain])


def enrichment(se, sf):
    base = sf.sum() / se.sum()
    return (sf.sum(0) / np.maximum(se.sum(0), 1)) / base


def lorenz(ae, af):
    tok, flip = ae.sum(0).cumsum(), af.sum(0).cumsum()
    return tok / tok[-1], flip / flip[-1]


rng = np.random.default_rng(0)
NBOOT, n = 400, len(per_chain)
be = np.empty((NBOOT, NB))
bl = np.empty((NBOOT, ABSMAX + 1))
for i in range(NBOOT):
    idx = rng.integers(0, n, n)
    be[i] = enrichment(SE[idx], SF[idx])
    bl[i] = lorenz(AE[idx], AF[idx])[1]
enr, lo, hi = enrichment(SE, SF), *np.percentile(be, [5, 95], axis=0)
tok_share, flip_share = lorenz(AE, AF)
llo, lhi = np.percentile(bl, [5, 95], axis=0)
auc = float(np.trapezoid(flip_share, tok_share))

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.62))
fig.subplots_adjust(wspace=0.42)

ax = axes[0]
ax.axhspan(0, 1, color="0.955", zorder=0)
ax.axhline(1, color="0.45", lw=0.9, ls="--", zorder=1)
ax.axvspan(0, MEAN_SPAN, color=C["orange"], alpha=0.16, lw=0, zorder=1)
ax.fill_between(ctr, lo, hi, color=C["blue"], alpha=0.20, lw=0, zorder=2)
ax.plot(ctr, enr, color=C["blue"], lw=1.5, zorder=3)
ax.set_xlim(-MAXD * 0.55, MAXD * 0.55)
ax.set_ylim(0.35, 2.05)
ax.set_xlabel("tokens from correction-marker onset", fontsize=8.2)
ax.set_ylabel("revisions per token\n(× corpus rate)", fontsize=8.2)
ax.tick_params(labelsize=7.6)
pk = int(np.argmax(enr))
ax.annotate(f"{enr[pk]:.1f}× around\nthe marker", (ctr[pk], enr[pk]),
            xytext=(-158, 1.76), fontsize=6.9, color=C["blue"], linespacing=1.4,
            ha="left", arrowprops=dict(arrowstyle="-", lw=0.7, color=C["blue"]))
lo_i = int(np.argmin(np.where(ctr > 40, enr, 9)))
# Right-aligned against the axes edge: left-aligning here ran the block past
# the spine and into the neighbouring panel's y-label at \linewidth.
ax.annotate(f"{enr[lo_i]:.1f}× for the\n150 tokens after",
            (ctr[lo_i], enr[lo_i]), xytext=(MAXD * 0.53, 1.97), fontsize=7.0,
            color=C["red"], linespacing=1.45, ha="right", va="top",
            arrowprops=dict(arrowstyle="-", lw=0.7, color=C["red"],
                            connectionstyle="angle,angleA=0,angleB=90,rad=3"))
ax.annotate("marker\nspan", (MEAN_SPAN / 2 + 6, 0.52), fontsize=6.4, color=C["orange"],
            ha="left", va="center", linespacing=1.3)
ax.set_title("Enriched around the marker, depleted after", fontsize=8.6, pad=7)

ax = axes[1]
ax.plot([0, 1], [0, 1], color="0.45", lw=0.9, ls="--", zorder=1)
ax.fill_between(tok_share, llo, lhi, color=C["purple"], alpha=0.22, lw=0, zorder=2)
ax.plot(tok_share, flip_share, color=C["purple"], lw=1.5, zorder=3)
j = int(np.searchsorted(tok_share, STATS["localization"]["aligned_interval_token_share"]))
FULLGRID = STATS["localization"]
ax.plot([tok_share[j]], [flip_share[j]], "o", ms=4.6, color=C["purple"], zorder=4)
ax.annotate(f"nearest {tok_share[j]*100:.0f}% of tokens\nhold {flip_share[j]*100:.0f}% of revisions",
            (tok_share[j], flip_share[j]), xytext=(0.30, 0.20), fontsize=6.9,
            color=C["purple"], linespacing=1.4, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.7, color=C["purple"]))
ax.annotate(f"AUROC {auc:.2f}", (0.97, 0.06), fontsize=7.4, color="0.3", ha="right")
ax.annotate("chance", (0.72, 0.74), fontsize=6.9, color="0.45", rotation=34,
            ha="center", va="center")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("share of tokens, nearest marker first", fontsize=8.2)
ax.set_ylabel("share of wrong→right\nrevisions captured", fontsize=8.2)
ax.tick_params(labelsize=7.6)
ax.set_title("A weak localizer for a monitor", fontsize=8.6, pad=7)

fig.suptitle("Where the answer actually changes, relative to the “wait” that claims it",
             fontsize=9.4, y=1.05)
save(fig, "fig5_localization")

json.dump({"grid": "stride-only (event-boundary probe points dropped)",
           "n_flips": n_flips,
           "n_flips_full_grid": FULLGRID["n_wrong_to_right_flips"],
           "auc_distance_localizer": auc,
           "enrichment_bins": {"centre": ctr.tolist(), "enrichment": enr.tolist(),
                               "lo": lo.tolist(), "hi": hi.tolist()},
           "operating_point": {"token_share": float(tok_share[j]),
                               "flip_share": float(flip_share[j]),
                               "enrichment": float(flip_share[j] / tok_share[j])},
           "mean_marker_span_tokens": MEAN_SPAN},
          (RD / "analysis/localization_profile.json").open("w"), indent=1)
print("wrote analysis/localization_profile.json")


# ---------------------------------------------------------------- Figure 6
# What the nulls exclude. The measured quantity is an OBSERVED accuracy delta;
# the estimand faithfulness makes a claim about is the per-event effect. Three
# sources of non-differential error sit between them (probe reliability,
# event-detection contamination, censoring at the token cap), and all three
# attenuate rather than inflate, so an observed bound of +-b maps to a bound of
# +-b/pi on the truth for combined selection purity pi. This is the paper's
# partial-identification step, and it is currently prose only.
MARGIN, THRESH = 0.10, 0.30
BOUNDS = [
    (f"Observed $\\Delta_{{\\mathrm{{acc}}}}$\nwhat we measure", MARGIN, C["blue"],
     "TOST-equivalent, BF$_{01}$ 3.5–8.9"),
    ("True per-event effect\n$\\pi = 0.74$", MARGIN / 0.74, C["purple"],
     "probe 74% reliable · detector precision 0.88 · 22.5% censored"),
    ("True per-event effect\n$\\pi = 0.50$ (pessimistic)", MARGIN / 0.50, C["orange"],
     "every source at roughly double our measured rate"),
]
fig, ax = plt.subplots(figsize=(5.5, 2.32))
fig.subplots_adjust(left=0.29, right=0.985, top=0.80, bottom=0.235)
for x in (-THRESH, THRESH):
    ax.axvline(x, color=C["red"], lw=1.1, ls="--", zorder=3)
ax.axvspan(THRESH, 0.40, color=C["red"], alpha=0.075, lw=0, zorder=0)
ax.axvspan(-0.40, -THRESH, color=C["red"], alpha=0.075, lw=0, zorder=0)
ax.axvline(0, color="0.55", lw=0.8, zorder=1)
for i, (lbl, b, col, note) in enumerate(BOUNDS):
    y = -i
    ax.plot([-b, b], [y, y], color=col, lw=7.5, solid_capstyle="butt",
            alpha=0.9, zorder=2)
    ax.annotate(f"$\\pm${b:.2f}", (b, y), xytext=(7, 0), textcoords="offset points",
                va="center", ha="left", fontsize=7.6, color=col, fontweight="bold")
    ax.annotate(lbl, (-0.415, y), fontsize=7.6, ha="right", va="center",
                linespacing=1.35, annotation_clip=False)
    ax.annotate(note, (0, y - 0.40), fontsize=6.1, color="0.42", ha="center",
                va="center")
ax.annotate("faithful-effect threshold\n(pre-registered)", (THRESH, 0.80),
            fontsize=7.0, color=C["red"], ha="center", va="center", linespacing=1.35)
ax.annotate("excluded by every bound", (THRESH + 0.052, -1.0), fontsize=6.9,
            color=C["red"], ha="center", va="center", rotation=90)
ax.set_xlim(-0.40, 0.40); ax.set_ylim(-2.72, 0.42)
ax.set_yticks([])
ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
ax.set_xlabel("effect on final-answer accuracy", fontsize=8.6, labelpad=4)
ax.tick_params(axis="x", labelsize=7.8)
ax.spines["left"].set_visible(False)
ax.set_title("What the nulls exclude, and what they do not", fontsize=9.4, pad=9)
save(fig, "fig6_partial_id")
