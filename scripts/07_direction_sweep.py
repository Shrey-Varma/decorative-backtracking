#!/usr/bin/env python
"""Stage 07 (§11 robustness): SAE-free correction-direction ablation swept
across layers.

Localizes the correction signal as a difference-in-means direction (NOT an SAE
feature, NOT vocabulary-thresholded) at each of `--set direction.layers` and
projection-ablates it during the continuation, on the SAME informative events
as stage 05. Appends `ablate_direction/layer=L` arms to continuations.jsonl and
registers their intervention keys, so stage 06 reports their Δ alongside the
SAE result. Reuses the cached natural_resample baseline.

  python scripts/07_direction_sweep.py --run-dir runs/exp1 --set direction.layers=[15,19,23]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answers import extract_explicit_answer
from src.cli import script_setup
from src.directions import compute_correction_directions
from src.experiment import (
    CONTINUATION_CACHE_KEY,
    DirectionContinuationRunner,
    classify_outcome,
    select_informative,
)
from src.generation import HFHookedGenerator
from src.io_utils import (
    ContinuationRecord,
    JsonlCache,
    group_events_by_chain,
    load_chains,
    load_events,
    read_json,
    write_json,
)
from src.model_loading import load_hf_model
from src.sae_interventions import PositionMask
from src.trajectory import load_transitions


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("07_direction_sweep")
    ncfg = cfg.experiment.necessity
    fcfg = cfg.experiment.features
    layers = [int(x) for x in cfg_dict.get("direction", {}).get("layers", [15, 19, 23])]
    print(f"direction sweep over layers {layers}")

    chains = {c.chain_id: c for c in load_chains(run_dir / "chains.jsonl")}
    events = load_events(run_dir / "events.jsonl")
    events_by_chain = group_events_by_chain(events)
    events_by_id = {ev.event_id: ev for ev in events}
    transitions = load_transitions(run_dir / "transitions.jsonl")

    informative = select_informative(transitions)
    informative.sort(key=lambda t: (t.chain_id, t.event_id))
    if ncfg.max_informative_events is not None:
        informative = informative[: ncfg.max_informative_events]
    informative = [t for t in informative if t.chain_id in chains and t.event_id in events_by_id]
    print(f"{len(informative)} informative events (same subset as stage 05)")
    if not informative:
        sys.exit("no informative events; run stages up to 05 first")

    hf_model, hf_tokenizer = load_hf_model(cfg.model)
    chain_list = list(chains.values())
    directions = compute_correction_directions(
        hf_model, chain_list, events_by_chain, layers, fcfg,
        max_chains=fcfg.corpus_max_chains,
    )
    write_json(
        run_dir / "directions_meta.json",
        {"layers": layers, "norm_check": {L: float(directions[L].norm()) for L in layers}},
    )

    gen = HFHookedGenerator(hf_model, hf_tokenizer, layers[0])
    cache = JsonlCache(run_dir / "continuations.jsonl", CONTINUATION_CACHE_KEY)
    n_cont = cfg.generation.continuation.n_continuations
    base_seed = cfg.generation.continuation.base_seed
    new_keys: list[str] = []

    for L in layers:
        runner = DirectionContinuationRunner(
            gen, directions[L], L, cfg.generation.continuation, cfg.generation.think_close
        )
        condition = f"ablate_direction/layer={L}"
        for tr in informative:
            chain = chains[tr.chain_id]
            ev = events_by_id[tr.event_id]
            ikey = f"dir|t={ev.onset}|layer={L}"
            new_keys.append(ikey)
            mask = PositionMask(ncfg.position_mask, t0=ev.onset, t1=ev.end)
            pending = [
                base_seed + k
                for k in range(n_cont)
                if cache.get(chain_id=chain.chain_id, event_id=ev.event_id,
                             condition=condition, seed=base_seed + k, intervention_key=ikey) is None
            ]
            if not pending:
                continue
            texts = runner.run_batch(chain, ev.onset, pending, mask)
            for seed, text in zip(pending, texts):
                final = extract_explicit_answer(text)
                cache.put(ContinuationRecord(
                    chain_id=chain.chain_id, event_id=ev.event_id, condition=condition,
                    seed=seed, truncation=ev.onset, ablation_mode="direction_projection",
                    position_mask=ncfg.position_mask, feature_ids=[],
                    text=text, final_answer=final,
                    outcome=classify_outcome(final, tr.pre_answer, chain.gold_answer),
                    intervention_key=ikey,
                    meta={"gold_answer": chain.gold_answer, "pre_answer": tr.pre_answer,
                          "problem_id": chain.problem_id, "layer": L},
                ))
        print(f"  layer {L}: done")

    # merge our keys into the active set so stage 06 keeps these rows AND the
    # existing stage-05 rows (natural_resample is shared and reused)
    akp = run_dir / "active_intervention_keys.json"
    active = set(read_json(akp)) if akp.exists() else set()
    active.update(new_keys)
    write_json(akp, sorted(active))
    print(f"direction sweep complete; {len(new_keys)} direction arms added -> {run_dir}")


if __name__ == "__main__":
    main()
