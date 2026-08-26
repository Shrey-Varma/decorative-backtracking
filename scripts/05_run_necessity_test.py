#!/usr/bin/env python
"""Stage 05 (§7): the knockout-and-continue necessity test on the hookable
TransformerLens backend. Informative events x 4 conditions x K shared seeds;
every continuation cached, so re-runs resume.

  python scripts/05_run_necessity_test.py --run-dir runs/exp1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.cli import script_setup
from src.experiment import (
    CONTINUATION_CACHE_KEY,
    HFContinuationRunner,
    NecessityInputs,
    run_necessity_test,
    select_informative,
)
from src.generation import HFHookedGenerator
from src.io_utils import (
    JsonlCache,
    group_events_by_chain,
    load_chains,
    load_events,
    read_json,
    write_json,
)
from src.model_loading import get_sae_hook_info, load_hf_model, load_sae
from src.trajectory import load_trajectories, load_transitions


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("05_run_necessity_test")
    ncfg = cfg.experiment.necessity

    chains = {c.chain_id: c for c in load_chains(run_dir / "chains.jsonl")}
    events = load_events(run_dir / "events.jsonl")
    events_by_chain = group_events_by_chain(events)
    events_by_id = {ev.event_id: ev for ev in events}
    trajectories = load_trajectories(run_dir / "trajectories.jsonl")
    transitions = load_transitions(run_dir / "transitions.jsonl")

    informative = select_informative(transitions)
    inputs_list = [
        NecessityInputs(
            chain=chains[tr.chain_id],
            event=events_by_id[tr.event_id],
            transition=tr,
            trajectory=trajectories.get(tr.chain_id),
            all_events=events_by_chain[tr.chain_id],
        )
        for tr in informative
        if tr.chain_id in chains and tr.event_id in events_by_id
    ]
    inputs_list.sort(key=lambda i: (i.chain.chain_id, i.event.event_id))
    if ncfg.max_informative_events is not None:
        inputs_list = inputs_list[: ncfg.max_informative_events]
    print(f"{len(inputs_list)} informative (wrong->right) events")
    if not inputs_list:
        # write the summary sentinel and exit 0: "no informative events" is a
        # legitimate empty result (common on small pilots), not a pipeline
        # error — otherwise the runner re-executes this stage forever
        write_json(run_dir / "active_intervention_keys.json", [])
        write_json(
            run_dir / "necessity_summary.json",
            {"n_informative_events": 0, "n_continuations": 0, "per_condition": {},
             "note": "no wrong->right corrected events; nothing to ablate"},
        )
        print("no wrong->right corrected events found — nothing to run")
        return

    feat = read_json(run_dir / "features" / "candidate_features.json")
    candidates = feat["candidates"]
    control_feature = int(feat["control_feature_id"])
    correction_features = (
        [c["feature_id"] for c in candidates]
        if ncfg.ablate_top_k_jointly
        else [candidates[0]["feature_id"]]
    )
    # §5: additional candidates get their own ablation arms so the causal
    # test — not the vocab ranking — decides which feature is the mechanism
    per_candidate = [
        c["feature_id"] for c in candidates[: max(ncfg.n_candidates_to_test, 1)]
    ]
    stats = torch.load(run_dir / "features" / "corpus_stats.pt", weights_only=True)
    corpus_means = stats["corpus_mean"]

    # interventional continuations run on the HF backend with an nn.Module
    # hook at the SAE layer (HFHookedGenerator): the intervention only edits
    # the layer-19 residual, and HF's fused decode is ~10-20x faster than the
    # TL loop at long context. Layer equivalence validated (cosine 0.9999).
    sae = load_sae(cfg.model)
    hook_info = get_sae_hook_info(sae)
    hf_model, hf_tokenizer = load_hf_model(cfg.model)
    runner = HFContinuationRunner(
        HFHookedGenerator(hf_model, hf_tokenizer, hook_info.hook_layer),
        sae, cfg.generation.continuation, cfg.generation.think_close,
        corpus_means=corpus_means,
    )

    counterfactual_fn = None
    if ncfg.ablation_mode == "counterfactual" or "counterfactual" in ncfg.secondary_ablation_modes:
        # counterfactual targets need TL activations — load TL only here
        from src.features import make_tl_encode_fn
        from src.model_loading import load_all

        loaded = load_all(cfg.model, need_tl=True)
        encode_fn = make_tl_encode_fn(
            loaded["tl_model"], loaded["sae"], loaded["hook_info"]
        )
        from src.experiment import choose_control_position

        def counterfactual_fn(inputs: NecessityInputs):
            """§7.1 counterfactual targets: the features' activations over a
            matched no-correction window of the same chain."""
            t_ctrl = choose_control_position(
                inputs.chain, inputs.all_events, inputs.event,
                ncfg.min_control_pos_distance, inputs.trajectory,
            )
            if t_ctrl is None:
                return None
            span = inputs.event.end - inputs.event.onset
            f = encode_fn(inputs.chain.token_ids[: t_ctrl + span + 1])
            window = f[t_ctrl: t_ctrl + span + 1]
            return window.mean(dim=0)[correction_features].tolist()

    cache = JsonlCache(run_dir / "continuations.jsonl", CONTINUATION_CACHE_KEY)
    records = run_necessity_test(
        inputs_list,
        runner,
        correction_features,
        control_feature,
        ncfg,
        n_continuations=cfg.generation.continuation.n_continuations,
        base_seed=cfg.generation.continuation.base_seed,
        cache=cache,
        counterfactual_values_fn=counterfactual_fn,
        per_candidate_features=per_candidate,
    )
    counts: dict[str, int] = {}
    for r in records:
        counts[r.condition] = counts.get(r.condition, 0) + 1

    # continuations.jsonl is an append-only cache: a prior run under a
    # different feature set / config left rows with other intervention_keys in
    # it. Record exactly the keys THIS run used so stage 06 analyzes only the
    # current intervention and never pools superseded continuations.
    active_keys = sorted({r.intervention_key for r in records})
    write_json(run_dir / "active_intervention_keys.json", active_keys)
    write_json(
        run_dir / "necessity_summary.json",
        {
            "n_informative_events": len(inputs_list),
            "n_continuations": len(records),
            "per_condition": counts,
            "correction_features": correction_features,
            "control_feature": control_feature,
            "n_active_intervention_keys": len(active_keys),
        },
    )
    print(f"{len(records)} continuations -> {run_dir / 'continuations.jsonl'}")


if __name__ == "__main__":
    main()
