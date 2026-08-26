#!/usr/bin/env python
"""Stage 02 (§5 lexicon seed, §9 criterion b): detect correction events in
each chain's reasoning trace; emit the human-validation sample (§11
event-detection circularity).

  python scripts/02_detect_events.py --run-dir runs/exp1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import script_setup
from src.events import detect_events, sample_validation_events
from src.io_utils import append_jsonl, load_chains, write_json


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("02_detect_events")
    chains = load_chains(run_dir / "chains.jsonl")

    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()
    all_events = []
    n_chains_with_events = 0
    for chain in chains:
        events = detect_events(chain, cfg.experiment.events)
        if events:
            n_chains_with_events += 1
        for ev in events:
            append_jsonl(events_path, ev)
            all_events.append(ev)

    validation = sample_validation_events(
        all_events, cfg.experiment.events.validation_sample
    )
    write_json(run_dir / "events_for_validation.json", validation)
    write_json(
        run_dir / "event_summary.json",
        {
            "n_chains": len(chains),
            "n_chains_with_events": n_chains_with_events,
            "n_events": len(all_events),
            "marker_counts": _marker_counts(all_events),
        },
    )
    print(
        f"{len(all_events)} events across {n_chains_with_events}/{len(chains)} "
        f"chains -> {events_path}\n"
        f"label events_for_validation.json (is_correction: true/false) for the "
        f"detection-precision estimate"
    )


def _marker_counts(events) -> dict:
    counts: dict[str, int] = {}
    for ev in events:
        counts[ev.marker] = counts.get(ev.marker, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
