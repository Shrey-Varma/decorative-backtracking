#!/usr/bin/env python
"""Stage 18: diagnostics for two claims the paper makes outside the main
statistical pipeline.

(a) DIRECTION-ABLATION MECHANISM (paper sec:results:necessity). Ablating the
    difference-in-means correction direction raises accuracy, and the paper
    attributes that to fewer chains being censored at the token cap rather
    than to suppressed reconsideration. That attribution is only checkable if
    we measure whether the ablation actually reduces reconsideration LANGUAGE.
    It does not: marker re-emission is near-unchanged while continuations get
    materially shorter, so the gain is a termination effect.

(c) SELECTION PURITY AT EVENT BOUNDARIES (paper app:probe). Events are
    selected by two probe reads at the event's own boundaries, and boundary
    reads carry the +2.0-point bias documented in 16_make_figures_extra.py.
    Re-deriving the informative set from the nearest STRIDE points outside the
    span instead retains far fewer events -- but that substitution also widens
    the window, so we bucket survival by how many extra tokens it introduces.
    Survival tracks the widening, not the bias, which is why purity enters the
    partial-identification bound as an unknown rather than a point estimate.

(b) PROBE RELIABILITY NEAR EVENTS (paper sec:methods:probe). The force-answer
    probe's instrument check is at the FINAL chain position, but the design
    selects events using probe reads at arbitrary mid-chain positions. There
    is no ground truth mid-chain, so we use volatility as a proxy: the rate at
    which ADJACENT grid points disagree on the intended answer, bucketed by
    distance to the nearest marker onset. If the probe were noisier at the
    moments selection relies on, that rate would rise near markers.

    Uses the STRIDE-ONLY grid. Probe points sitting on an event boundary carry
    the +2.0-point measurement bias documented in 16_make_figures_extra.py, so
    both onsets and span ends are dropped before measuring.

  python scripts/18_review_diagnostics.py --run-dir results/exp1
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import script_setup
from src.io_utils import write_json

MARKER_WINDOW_CHARS = 240   # matches the re-emission window used in the paper
DIST_BUCKETS = [(0, 32), (32, 96), (96, 256), (256, None)]


def jl(path):
    with path.open() as fh:
        for line in fh:
            yield json.loads(line)


def marker_pattern(lexicon):
    # identical construction to src.events._marker_pattern
    parts = sorted((re.escape(m) for m in lexicon), key=len, reverse=True)
    return re.compile(r"(?<![a-z])(" + "|".join(parts) + r")(?![a-z])", re.IGNORECASE)


def continuation_stats(run_dir, pat):
    agg = defaultdict(lambda: {"n": 0, "marker": 0, "chars": 0, "no_answer": 0})
    for d in jl(run_dir / "continuations.jsonl"):
        a = agg[d["condition"]]
        text = d.get("text") or ""
        a["n"] += 1
        a["chars"] += len(text)
        a["marker"] += bool(pat.search(text[:MARKER_WINDOW_CHARS]))
        a["no_answer"] += not d.get("final_answer")
    return {
        cond: {
            "n": a["n"],
            "marker_rate_within_240_chars": a["marker"] / a["n"],
            "mean_chars": a["chars"] / a["n"],
            "no_parseable_answer_rate": a["no_answer"] / a["n"],
        }
        for cond, a in sorted(agg.items())
    }


def probe_volatility(run_dir):
    onsets, boundary = {}, {}
    for d in jl(run_dir / "events.jsonl"):
        onsets.setdefault(d["chain_id"], []).append(d["onset"])
        boundary.setdefault(d["chain_id"], set()).update((d["onset"], d["end"]))
    onsets = {k: sorted(v) for k, v in onsets.items()}

    buckets = {f"{lo}-{hi if hi else 'inf'}": [0, 0] for lo, hi in DIST_BUCKETS}
    for t in jl(run_dir / "trajectories.jsonl"):
        cid = t["chain_id"]
        ons = onsets.get(cid)
        if not ons:
            continue
        bad = boundary.get(cid, set())
        pts = [p for p in t["points"] if p["position"] not in bad]
        for a, b in zip(pts, pts[1:]):
            mid = (a["position"] + b["position"]) / 2
            dist = min(abs(mid - o) for o in ons)
            for lo, hi in DIST_BUCKETS:
                if dist >= lo and (hi is None or dist < hi):
                    k = f"{lo}-{hi if hi else 'inf'}"
                    buckets[k][1] += 1
                    buckets[k][0] += a["answer"] != b["answer"]
                    break
    return {
        k: {"n_pairs": n, "answer_change_rate": (c / n) if n else None}
        for k, (c, n) in buckets.items()
    }


def selection_purity(run_dir):
    """Informative events re-derived on the stride-only grid, bucketed by how
    much extra window the substitution introduces."""
    events, boundary = {}, {}
    for d in jl(run_dir / "events.jsonl"):
        events.setdefault(d["chain_id"], []).append((d["event_id"], d["onset"], d["end"]))
        boundary.setdefault(d["chain_id"], set()).update((d["onset"], d["end"], d["end"] + 1))
    pts = {t["chain_id"]: {p["position"]: (p["answer"], p["correct"]) for p in t["points"]}
           for t in jl(run_dir / "trajectories.jsonl")}

    selected = []
    for cid, evs in events.items():
        P = pts.get(cid, {})
        for eid, on, en in evs:
            pre, post = P.get(on), P.get(en + 1)
            if pre and post and pre[0] is not None and not pre[1] and post[1]:
                selected.append((cid, on, en))

    buckets = {k: [0, 0] for k in ("<=8", "9-16", "17-32", ">32")}
    for cid, on, en in selected:
        P, bad = pts[cid], boundary[cid]
        before = [p for p in P if p not in bad and p < on]
        after = [p for p in P if p not in bad and p > en]
        if not before or not after:
            continue
        bp, ap = max(before), min(after)
        gap = max(on - bp, ap - en)
        k = "<=8" if gap <= 8 else "9-16" if gap <= 16 else "17-32" if gap <= 32 else ">32"
        buckets[k][1] += 1
        buckets[k][0] += P[bp][0] is not None and not P[bp][1] and P[ap][1]

    tot_n = sum(n for _, n in buckets.values())
    tot_k = sum(k for k, _ in buckets.values())
    return {
        "n_informative_boundary_grid": len(selected),
        "n_surviving_stride_grid": tot_k,
        "survival_overall": (tot_k / tot_n) if tot_n else None,
        "survival_by_extra_window_tokens": {
            k: {"n": n, "survived": c, "rate": (c / n) if n else None}
            for k, (c, n) in buckets.items()},
        "note": ("Survival rises monotonically as the substitute points sit closer to "
                 "the true boundaries, so most of the loss is the wider window rather "
                 "than the boundary measurement bias."),
    }


def main() -> None:
    cfg, _cfg_dict, run_dir = script_setup("18_review_diagnostics")
    pat = marker_pattern(cfg.experiment.events.lexicon)

    # continuations.jsonl is 174 MB and not in the code release. The volatility
    # half runs off trajectories.jsonl + events.jsonl, which are tracked, so
    # report what is computable rather than failing outright.
    if (run_dir / "continuations.jsonl").exists():
        conts = continuation_stats(run_dir, pat)
    else:
        conts = {}
        print("continuations.jsonl absent: skipping the direction-ablation half; "
              "its computed values are in the committed review_diagnostics.json\n")
    volat = probe_volatility(run_dir)
    purity = selection_purity(run_dir)

    ref = conts.get("natural_resample")
    direction = {k: v for k, v in conts.items() if k.startswith("ablate_direction")}
    summary = None
    if ref and direction:
        summary = {
            "reference": "natural_resample",
            "reference_marker_rate": ref["marker_rate_within_240_chars"],
            "direction_marker_rate_range": [
                min(v["marker_rate_within_240_chars"] for v in direction.values()),
                max(v["marker_rate_within_240_chars"] for v in direction.values()),
            ],
            "direction_length_ratio_range": [
                min(v["mean_chars"] for v in direction.values()) / ref["mean_chars"],
                max(v["mean_chars"] for v in direction.values()) / ref["mean_chars"],
            ],
            "note": ("Marker emission is near-unchanged under direction ablation while "
                     "continuations shorten, so the accuracy gain is a termination "
                     "effect and not suppressed reconsideration."),
        }

    record = {
        "marker_window_chars": MARKER_WINDOW_CHARS,
        "per_condition": conts,
        "direction_ablation_summary": summary,
        "continuations_available": bool(conts),
        "probe_volatility_by_distance_to_onset": volat,
        "selection_purity_at_boundaries": purity,
        "probe_volatility_note": ("Stride-only grid; probe points on an event onset or "
                                  "span end are dropped. A flat profile means the probe "
                                  "is no noisier at the positions event selection uses."),
    }
    out = run_dir / "analysis" / "review_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, record)
    print(f"-> {out}\n")

    if summary:
        lo, hi = summary["direction_marker_rate_range"]
        rlo, rhi = summary["direction_length_ratio_range"]
        print(f"direction ablation: marker {lo:.3f}-{hi:.3f} vs "
              f"{summary['reference_marker_rate']:.3f} natural; "
              f"length {100*(1-rhi):.0f}-{100*(1-rlo):.0f}% shorter")
    p = record["selection_purity_at_boundaries"]
    print(f"\nselection purity: {p['n_surviving_stride_grid']}/"
          f"{p['n_informative_boundary_grid']} survive the stride grid "
          f"({p['survival_overall']:.1%})")
    for k, v in p["survival_by_extra_window_tokens"].items():
        if v["n"]:
            print(f"  +{k:>6s} tok window  n={v['n']:4d}  survived={v['rate']:.1%}")

    print("\nprobe volatility (adjacent-point answer change):")
    for k, v in volat.items():
        print(f"  dist {k:>9s}  n={v['n_pairs']:6d}  rate={v['answer_change_rate']:.3f}")


if __name__ == "__main__":
    main()
