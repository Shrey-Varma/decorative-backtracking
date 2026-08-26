#!/usr/bin/env python
"""Stage 12: mechanistic sufficiency — clamp the correction carrier ON.

The knockout arms (stages 05/07) tested necessity; this stage tests the other
half of "faithful signal": at the SAME wrong_point insertion points as stage
11 (intended answer wrong, far from every event), does turning the carrier ON

  clamp_on_feature/scale=S   SAE feature clamped to S x its typical onset
                             magnitude (dose ladder S in {1, 2} — a
                             sufficiency null at one dose invites "you
                             underdosed"; a ladder + the induction
                             manipulation check does not)
  add_direction/layer=L/scale=S
                             residual pushed along the diff-in-means
                             direction (x' = x + α d̂, α = S x the raw
                             diff-in-means norm)

(a) induce a VERBALIZED correction in the continuation (manipulation check:
meta.marker_in_continuation vs the natural base rate), and (b) rescue the
wrong intended answer? Compared against natural_at_insert — the no-hook
baseline at the same truncation on the same backend with the same seeds.

The clamp window is the median informative-event span (the hooked analogue of
splicing one correction), applied event_only so the model is free to continue
naturally afterwards.

  python scripts/12_sufficiency_hooked.py --run-dir runs/exp1
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.answers import extract_explicit_answer
from src.cli import script_setup
from src.directions import _resid_capture_hooks, compute_correction_directions
from src.experiment import (
    CONTINUATION_CACHE_KEY,
    DirectionAdditionContinuationRunner,
    HFContinuationRunner,
    InterventionSpec,
    classify_outcome,
    select_informative,
)
from src.generation import HFHookedGenerator
from src.insertion import select_insertion_point, verbalizes_marker
from src.io_utils import (
    ContinuationRecord,
    JsonlCache,
    group_events_by_chain,
    load_chains,
    load_events,
    read_json,
    write_json,
)
from src.model_loading import get_sae_hook_info, load_hf_model, load_sae
from src.sae_interventions import PositionMask, _sae_dtype
from src.trajectory import load_trajectories, load_transitions


def typical_on_value(hf_model, sae, hook_layer: int, items) -> float:
    """Median over informative events of the feature's max activation within
    the real correction span — the magnitude the clamp-ON arm drives the
    feature to. `items` yields (chain, event, feature_id)."""
    device = next(hf_model.parameters()).device
    vals = []
    for chain, ev, fid in items:
        store: dict = {}
        handles = _resid_capture_hooks(hf_model, [hook_layer], store)
        try:
            with torch.no_grad():
                ids = torch.tensor([chain.token_ids[: ev.end + 1]],
                                   dtype=torch.long, device=device)
                try:
                    hf_model(ids, use_cache=False, logits_to_keep=1)
                except TypeError:
                    hf_model(ids, use_cache=False, num_logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        resid = store[hook_layer][0]                       # [seq, d]
        window = resid[ev.onset: ev.end + 1].to(_sae_dtype(sae, resid))
        with torch.no_grad():
            f = sae.encode(window)[..., fid]
        m = float(f.max())
        if m > 0:
            vals.append(m)
    if not vals:
        raise ValueError("correction feature never active inside event spans")
    return float(statistics.median(vals))


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("12_sufficiency_hooked")
    ncfg = cfg.experiment.necessity
    ccfg = cfg.generation.continuation
    fcfg = cfg.experiment.features
    lexicon = cfg.experiment.events.lexicon
    scfg = cfg_dict.get("sufficiency", {})
    scales = [float(s) for s in scfg.get("scales", [1.0, 2.0])]

    chains = {c.chain_id: c for c in load_chains(run_dir / "chains.jsonl")}
    events = load_events(run_dir / "events.jsonl")
    events_by_chain = group_events_by_chain(events)
    events_by_id = {ev.event_id: ev for ev in events}
    trajectories = load_trajectories(run_dir / "trajectories.jsonl")
    transitions = load_transitions(run_dir / "transitions.jsonl")

    informative = select_informative(transitions)
    informative.sort(key=lambda t: (t.chain_id, t.event_id))
    if ncfg.max_informative_events is not None:
        informative = informative[: ncfg.max_informative_events]
    informative = [t for t in informative if t.chain_id in chains and t.event_id in events_by_id]
    print(f"{len(informative)} informative events (same subset as stages 05/07/10/11)")
    if not informative:
        sys.exit("no informative events")

    # same deterministic insertion points as stage 11
    points = []  # (transition, event, t_ins, pre_answer)
    for tr in informative:
        chain, ev = chains[tr.chain_id], events_by_id[tr.event_id]
        sel = select_insertion_point(
            chain, events_by_chain[chain.chain_id], ev,
            ncfg.min_control_pos_distance, trajectories.get(chain.chain_id),
        )
        if sel is None:
            continue
        t_ins, pre = sel
        points.append((tr, ev, t_ins, pre if pre is not None else tr.pre_answer))
    print(f"{len(points)} usable insertion points")
    if not points:
        sys.exit("no insertion points")

    # clamp window: the hooked analogue of splicing ONE correction — the
    # median real correction span, event_only so the model continues freely
    window = int(scfg.get(
        "window", statistics.median(ev.end - ev.onset for _, ev, _, _ in points)
    ))
    print(f"clamp window: {window} tokens (event_only)")

    feat = read_json(run_dir / "features" / "candidate_features.json")
    fid = int(feat["candidates"][0]["feature_id"])

    sae = load_sae(cfg.model)
    hook_info = get_sae_hook_info(sae)
    hf_model, hf_tokenizer = load_hf_model(cfg.model)
    gen = HFHookedGenerator(hf_model, hf_tokenizer, hook_info.hook_layer)

    v_on = typical_on_value(
        hf_model, sae, hook_info.hook_layer,
        [(chains[tr.chain_id], ev, fid) for tr, ev, _, _ in points],
    )
    print(f"feature {fid} clamp-ON value: {v_on:.4g} (median of in-span maxima)")

    dir_layer = int(scfg.get("direction_layer", hook_info.hook_layer))
    dirs, raw_norms = compute_correction_directions(
        hf_model, list(chains.values()), events_by_chain, [dir_layer], fcfg,
        max_chains=fcfg.corpus_max_chains, return_raw=True,
    )
    print(f"direction addition: layer {dir_layer}, raw diff-in-means norm = "
          f"{raw_norms[dir_layer]:.4g}, scales {scales}")

    sae_runner = HFContinuationRunner(
        gen, sae, ccfg, cfg.generation.think_close
    )
    dir_runners = {
        s: DirectionAdditionContinuationRunner(
            gen, dirs[dir_layer], raw_norms[dir_layer] * s, dir_layer,
            ccfg, cfg.generation.think_close,
        )
        for s in scales
    }

    cache = JsonlCache(run_dir / "continuations.jsonl", CONTINUATION_CACHE_KEY)
    seeds_all = [ccfg.base_seed + k for k in range(ccfg.n_continuations)]
    new_keys: list[str] = []

    def run_arm(condition, ikey, tr, ev, t_ins, pre, run_fn, ablation_mode, extra_meta):
        new_keys.append(ikey)
        chain = chains[tr.chain_id]
        pending = [s for s in seeds_all
                   if cache.get(chain_id=chain.chain_id, event_id=ev.event_id,
                                condition=condition, seed=s, intervention_key=ikey) is None]
        if not pending:
            return 0
        texts = run_fn(chain, t_ins, pending)
        for seed, text in zip(pending, texts):
            final = extract_explicit_answer(text)
            cache.put(ContinuationRecord(
                chain_id=chain.chain_id, event_id=ev.event_id, condition=condition,
                seed=seed, truncation=t_ins, ablation_mode=ablation_mode,
                position_mask="event_only" if ablation_mode else None,
                feature_ids=[fid] if ablation_mode == "clamp_on" else [],
                text=text, final_answer=final,
                outcome=classify_outcome(final, pre, chain.gold_answer),
                intervention_key=ikey,
                meta={"gold_answer": chain.gold_answer, "pre_answer": pre,
                      "problem_id": chain.problem_id, "insert_pos": t_ins,
                      "marker_in_continuation": verbalizes_marker(text, lexicon),
                      **extra_meta},
            ))
        return len(pending)

    n_done = 0
    for i, (tr, ev, t_ins, pre) in enumerate(points):
        mask = PositionMask("event_only", t0=t_ins, t1=t_ins + window)
        n_done += run_arm(
            "natural_at_insert", f"ins_nat|t={t_ins}",
            tr, ev, t_ins, pre,
            lambda c, t, s: sae_runner.run_batch(c, t, s, None),
            None, {},
        )
        for s_ in scales:
            v = v_on * s_
            n_done += run_arm(
                f"clamp_on_feature/scale={s_:g}",
                f"ins_clamp|t={t_ins}|f={fid}|v={v:.4g}|w={window}",
                tr, ev, t_ins, pre,
                lambda c, t, sd, v=v: sae_runner.run_batch(
                    c, t, sd,
                    InterventionSpec([fid], "counterfactual", mask,
                                     counterfactual_values=[v]),
                ),
                "clamp_on", {"clamp_value": v, "scale": s_},
            )
            alpha = raw_norms[dir_layer] * s_
            n_done += run_arm(
                f"add_direction/layer={dir_layer}/scale={s_:g}",
                f"ins_diradd|t={t_ins}|layer={dir_layer}|a={alpha:.4g}|w={window}",
                tr, ev, t_ins, pre,
                lambda c, t, sd, s=s_: dir_runners[s].run_batch(c, t, sd, mask),
                "direction_addition", {"alpha": alpha, "layer": dir_layer, "scale": s_},
            )
        print(f"  [{i + 1}/{len(points)}] {ev.event_id}: {n_done} continuations so far",
              flush=True)

    akp = run_dir / "active_intervention_keys.json"
    active = set(read_json(akp)) if akp.exists() else set()
    active.update(new_keys)
    write_json(akp, sorted(active))
    write_json(run_dir / "sufficiency_meta.json", {
        "n_points": len(points), "feature_id": fid, "clamp_value_1x": v_on,
        "window": window, "direction_layer": dir_layer,
        "direction_raw_norm": raw_norms[dir_layer], "scales": scales,
    })
    print(f"sufficiency test complete ({n_done} new) -> {run_dir / 'continuations.jsonl'}")


if __name__ == "__main__":
    main()
