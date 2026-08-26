#!/usr/bin/env python
"""Stage 13 (#4): do verbalized corrections coincide with internal state
changes — and do those changes predict answer movement?
(see src/activation_shift.py for the metric and its controls)

For every chain with events (capped at features.corpus_max_chains), one
forward pass captures the residual stream at each target layer; each event's
before/after window shift is paired with a SENTENCE-START pseudo-span in the
same chain (position-decile-matched, clear of every event) — so 'state
changes at discourse boundaries' is subtracted out, not just 'state drifts
with depth'. Robustness over windows {8, 16, 32}. Reported overall and split
by whether the event is informative (straddles a wrong->right transition).

Also closes the correlational half of #4 with two AUROCs per layer:
  - projection of resid[onset] onto the diff-in-means correction direction
    -> does the internal correction signal predict answer movement?
  - the event's shift magnitude -> do BIGGER internal state changes at
    corrections predict answer movement?

  python scripts/13_activation_shift.py --run-dir runs/exp1
"""

import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.activation_shift import (
    auroc,
    event_control_pairs,
    sentence_start_positions,
    shift_stats,
    span_shift,
)
from src.cli import script_setup
from src.directions import _resid_capture_hooks, compute_correction_directions
from src.io_utils import group_events_by_chain, load_chains, load_events, write_json
from src.model_loading import load_hf_model
from src.trajectory import load_transitions

WINDOWS = (8, 16, 32)
W_PRIMARY = 16
# "length_matched" reproduces the shipped numbers: the pseudo-span has the same
# token length as the event, so the gap between the before/after windows is
# matched but the after-window boundary type is not (a real event span ends at a
# sentence end, a length-matched one usually does not). "sentence_end" builds
# the control span with the same rule that built the event span, removing that
# asymmetry at the cost of exact length matching. See
# activation_shift.span_end_at_sentence.
CONTROL_SPAN_END = "length_matched"


def _chain_rng(chain_id: str) -> random.Random:
    # process-independent seed (hash() is salted per process)
    return random.Random(int(hashlib.sha256(chain_id.encode()).hexdigest()[:8], 16))


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("13_activation_shift")
    layers = [15, 19, 23]
    fcfg = cfg.experiment.features
    max_chains = fcfg.corpus_max_chains
    print(f"activation shift: layers {layers}, windows {WINDOWS} (primary {W_PRIMARY})")

    all_chains = sorted(load_chains(run_dir / "chains.jsonl"), key=lambda c: c.chain_id)
    events_by_chain = group_events_by_chain(load_events(run_dir / "events.jsonl"))
    transitions = {tr.event_id: tr for tr in load_transitions(run_dir / "transitions.jsonl")}
    chains = [c for c in all_chains
              if c.trace_token_span is not None and events_by_chain.get(c.chain_id)]
    chains = chains[:max_chains]
    print(f"{len(chains)} chains with events")

    hf_model, _ = load_hf_model(cfg.model)
    device = next(hf_model.parameters()).device

    # diff-in-means directions for the projection AUROC (fit on marker spans,
    # scored at onsets; fit uses no answer labels, so no label circularity)
    directions = compute_correction_directions(
        hf_model, chains, events_by_chain, layers, fcfg, max_chains=max_chains,
    )

    subsets = ("all", "informative", "non_informative")
    # diffs[L][w][subset][chain_id] -> paired (event - control) shifts
    diffs = {L: {w: {s: {} for s in subsets} for w in WINDOWS} for L in layers}
    raw = {L: {w: {"event": [], "control": []} for w in WINDOWS} for L in layers}
    # per-event scores for the AUROCs (primary window)
    scores = {L: {"proj": [], "shift": [], "w2r": [], "flip": []} for L in layers}
    n_pairs = 0

    for i, chain in enumerate(chains):
        evs = events_by_chain[chain.chain_id]
        boundary_candidates = sentence_start_positions(chain)
        pairs = event_control_pairs(
            chain, evs, W_PRIMARY, _chain_rng(chain.chain_id),
            candidates=boundary_candidates,
            control_span_end=CONTROL_SPAN_END,
        )
        if not pairs:
            continue
        store: dict = {}
        handles = _resid_capture_hooks(hf_model, layers, store)
        try:
            with torch.no_grad():
                ids = torch.tensor([chain.token_ids], dtype=torch.long, device=device)
                try:
                    hf_model(ids, use_cache=False, logits_to_keep=1)
                except TypeError:
                    hf_model(ids, use_cache=False, num_logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        resid_by_layer = {L: store[L][0].float().cpu() for L in layers}
        del store

        for ev, ctrl, ctrl_end in pairs:
            tr = transitions.get(ev.event_id)
            for L in layers:
                resid = resid_by_layer[L]
                for w in WINDOWS:
                    s_ev = span_shift(resid, ev.onset, ev.end, w)
                    s_ct = span_shift(resid, ctrl, ctrl_end, w)
                    if s_ev is None or s_ct is None:
                        continue
                    raw[L][w]["event"].append(s_ev)
                    raw[L][w]["control"].append(s_ct)
                    sub = ("informative" if tr is not None and tr.wrong_to_right
                           else "non_informative")
                    for s in ("all", sub):
                        diffs[L][w][s].setdefault(chain.chain_id, []).append(s_ev - s_ct)
                    if L == layers[0] and w == W_PRIMARY:
                        n_pairs += 1
                # AUROC scores at the primary window, labels from transitions
                if tr is not None and tr.pre_answer is not None:
                    s_ev = span_shift(resid, ev.onset, ev.end, W_PRIMARY)
                    if s_ev is not None:
                        proj = float(resid[ev.onset] @ directions[L])
                        scores[L]["proj"].append(proj)
                        scores[L]["shift"].append(s_ev)
                        scores[L]["w2r"].append(bool(tr.wrong_to_right))
                        scores[L]["flip"].append(bool(tr.flipped))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(chains)} chains, {n_pairs} primary pairs", flush=True)

    results = {
        "windows": list(WINDOWS), "window_primary": W_PRIMARY, "layers": layers,
        "n_pairs_primary": n_pairs, "controls": f"sentence_start_matched/{CONTROL_SPAN_END}",
        "per_layer": {}, "auroc": {},
    }
    for L in layers:
        results["per_layer"][str(L)] = {
            str(w): {
                "mean_event_shift": float(np.mean(raw[L][w]["event"])) if raw[L][w]["event"] else None,
                "mean_control_shift": float(np.mean(raw[L][w]["control"])) if raw[L][w]["control"] else None,
                **{s: shift_stats(diffs[L][w][s]) for s in subsets},
            }
            for w in WINDOWS
        }
        sc = scores[L]
        results["auroc"][str(L)] = {
            "n_events": len(sc["proj"]),
            "direction_projection_wrong_to_right": auroc(sc["proj"], sc["w2r"]),
            "direction_projection_flipped": auroc(sc["proj"], sc["flip"]),
            "shift_magnitude_wrong_to_right": auroc(sc["shift"], sc["w2r"]),
            "shift_magnitude_flipped": auroc(sc["shift"], sc["flip"]),
        }

    out = run_dir / "analysis" / "activation_shift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, results)
    print(f"-> {out}")

    print("\n=== DIGEST ===")
    for L in layers:
        r = results["per_layer"][str(L)][str(W_PRIMARY)]
        a = r["all"] or {}
        print(f"layer {L} (w={W_PRIMARY}): event={r['mean_event_shift']:.4f} "
              f"control={r['mean_control_shift']:.4f} "
              f"Δ={a.get('mean_event_minus_control', float('nan')):.4f} "
              f"d={a.get('cohens_d')} p={a.get('p_two_sided')}")
        for s in ("informative", "non_informative"):
            st = r[s]
            if st:
                print(f"    {s}: Δ={st['mean_event_minus_control']:.4f} "
                      f"p={st['p_two_sided']:.4g} (n={st['n_chains']} chains)")
        au = results["auroc"][str(L)]
        print(f"    AUROC (n={au['n_events']}): "
              f"proj->w2r={au['direction_projection_wrong_to_right']} "
              f"shift->w2r={au['shift_magnitude_wrong_to_right']} "
              f"proj->flip={au['direction_projection_flipped']} "
              f"shift->flip={au['shift_magnitude_flipped']}")


if __name__ == "__main__":
    main()
