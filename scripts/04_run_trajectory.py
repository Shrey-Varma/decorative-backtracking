#!/usr/bin/env python
"""Stage 04 (§6): force-answer trajectory over a prefix grid + every event
boundary; localize wrong->right flips and whether they align to events (the
position-confound diagnostic). Also emits per-event transitions — the
wrong->right subset is the input to the necessity test (§9 criterion c).

Runs on the fast backend: probes are non-interventional.

  python scripts/04_run_trajectory.py --run-dir runs/exp1
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import script_setup
from src.generation import make_cached_batch_prober, make_fast_backend
from src.io_utils import (
    JsonlCache,
    append_jsonl,
    group_events_by_chain,
    load_chains,
    load_events,
    write_json,
)
from src.model_loading import load_all
from src.trajectory import compute_trajectory, event_transitions, localization_summary


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("04_run_trajectory")
    chains = load_chains(run_dir / "chains.jsonl")
    events_by_chain = group_events_by_chain(load_events(run_dir / "events.jsonl"))

    loaded = load_all(cfg.model, need_fast=True, need_tl=False)
    tokenizer = loaded["tokenizer"]
    backend = make_fast_backend(loaded["fast_model"], tokenizer, cfg.model.fast_backend)
    probe_cache = JsonlCache(run_dir / "probe_cache.jsonl", ["chain_id", "position", "template"])
    prober = make_cached_batch_prober(backend, tokenizer, cfg.generation.probe, probe_cache)

    traj_path = run_dir / "trajectories.jsonl"
    trans_path = run_dir / "transitions.jsonl"
    for p in (traj_path, trans_path):
        if p.exists():
            p.unlink()

    trajectories = []
    n_w2r = 0
    for chain in chains:
        evs = events_by_chain.get(chain.chain_id)
        if not evs or chain.trace_token_span is None:
            continue
        traj = compute_trajectory(chain, evs, prober, cfg.experiment.trajectory)
        trajectories.append(traj)
        append_jsonl(traj_path, dataclasses.asdict(traj))
        for tr in event_transitions(chain, evs, prober):
            d = dataclasses.asdict(tr)
            d["flipped"] = tr.flipped
            d["wrong_to_right"] = tr.wrong_to_right
            append_jsonl(trans_path, d)
            n_w2r += tr.wrong_to_right

    summary = localization_summary(trajectories)
    summary["n_trajectories"] = len(trajectories)
    summary["n_wrong_to_right_events"] = n_w2r
    write_json(run_dir / "trajectory_summary.json", summary)
    print(f"{len(trajectories)} trajectories, {n_w2r} wrong->right events\n{summary}")


if __name__ == "__main__":
    main()
