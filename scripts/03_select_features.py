#!/usr/bin/env python
"""Stage 03 (§5): score every SAE feature (vocabulary-association +
answer-predictive), select the candidate correction features and the matched
control feature. Scoring uses the PILOT split only, so selection never sees
the chains the causal claim is evaluated on.

RUN AFTER STAGE 04: the answer-predictive labels (did the intended answer
change across the event?) are read from 04's transitions.jsonl. This keeps
every force-answer probe on the single fast backend — mixing TL and vLLM
probes within one trajectory would let backend numerics masquerade as answer
flips at event boundaries.

Deliverable: <run>/features/candidate_features.json (+ corpus stats tensors).

  python scripts/03_select_features.py --run-dir runs/exp1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.cli import script_setup
from src.features import (
    answer_predictive_score,
    collect_top_examples,
    compute_corpus_stats_and_windows,
    make_tl_encode_fn,
    select_candidates,
    select_control_feature,
    write_dashboards,
)
from src.io_utils import group_events_by_chain, load_chains, load_events, write_json
from src.metrics import split_chains
from src.model_loading import load_all
from src.trajectory import load_transitions


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("03_select_features")
    fcfg = cfg.experiment.features

    transitions_path = run_dir / "transitions.jsonl"
    if not transitions_path.exists():
        sys.exit(
            "transitions.jsonl not found — run scripts/04_run_trajectory.py first "
            "(stage 03 consumes its event transitions for the answer-predictive score)"
        )
    event_flips = {tr.event_id: tr.flipped for tr in load_transitions(transitions_path)}

    chains = load_chains(run_dir / "chains.jsonl")
    events = load_events(run_dir / "events.jsonl")
    events_by_chain = group_events_by_chain(events)

    split = split_chains(
        [c.chain_id for c in chains],
        cfg.experiment.splits.pilot_fraction,
        cfg.experiment.splits.split_seed,
    )
    pilot = [c for c in chains if split[c.chain_id] == "pilot" and events_by_chain.get(c.chain_id)]
    if not pilot:
        print("WARNING: empty pilot split; scoring on all chains with events")
        pilot = [c for c in chains if events_by_chain.get(c.chain_id)]

    loaded = load_all(cfg.model, need_tl=True)
    tl_model, sae, hook_info = loaded["tl_model"], loaded["sae"], loaded["hook_info"]
    encode_fn = make_tl_encode_fn(tl_model, sae, hook_info, fcfg.activation_batch_tokens)

    print(f"scoring on {len(pilot)} pilot chains (hook: {hook_info.hook_name})")
    stats, window_acts = compute_corpus_stats_and_windows(
        pilot, events_by_chain, encode_fn, fcfg
    )

    auroc = None
    try:
        auroc = answer_predictive_score(window_acts, event_flips)
    except ValueError as e:
        print(f"WARNING: answer-predictive score unavailable ({e})")

    candidates = select_candidates(stats, auroc, fcfg)
    control_id = select_control_feature(
        stats, candidates,
        cfg.experiment.necessity.control_feature,
        cfg.experiment.necessity.control_feature_seed,
    )

    examples = collect_top_examples(
        pilot, encode_fn, [c.feature_id for c in candidates] + [control_id], fcfg
    )
    for c in candidates:
        c.top_examples = examples.get(c.feature_id, [])

    feat_dir = run_dir / "features"
    write_dashboards(feat_dir, candidates, control_id, stats)
    torch.save(
        {
            "corpus_mean": stats.corpus_mean,
            "activation_freq": stats.activation_freq,
            "vocab_score": stats.vocab_score,
        },
        feat_dir / "corpus_stats.pt",
    )
    write_json(feat_dir / "control_feature_examples.json", examples.get(control_id, []))

    for c in candidates:
        print(
            f"feature {c.feature_id}: vocab={c.vocab_score:.3f} "
            f"auroc={c.answer_predictive_auroc} freq={c.activation_freq:.5f}"
        )
    print(f"control feature: {control_id}\n-> {feat_dir / 'candidate_features.json'}")


if __name__ == "__main__":
    main()
