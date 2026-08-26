#!/usr/bin/env python
"""Stage 15: publishable provenance record for the run.

The pre-registration claim in the paper is "decision rules were frozen before
the test split was unsealed", and its only evidence is a file in a directory.
File mtimes are mutable and the project is not under version control, so that
claim is currently unfalsifiable. Hashing the pre-registration and the frozen
config, and publishing the digest alongside the paper, does not prove WHEN the
file was written — but it does pin WHAT was registered, so any later edit to
the registered rules is detectable by anyone holding the digest.

This also audits manifest completeness: every stage writes one via
`cli.script_setup`, so a stage whose outputs are present but whose manifest is
missing means the results directory was pulled incompletely and the config that
produced those rows is not recorded locally.

  python scripts/15_provenance.py --run-dir results/exp1
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import script_setup
from src.io_utils import write_json

# stage -> an output that only that stage produces
STAGE_OUTPUTS = {
    "01_generate_chains": ["chains.jsonl"],
    "02_detect_events": ["events.jsonl"],
    "03_select_features": ["features/candidate_features.json"],
    "04_run_trajectory": ["trajectories.jsonl"],
    "05_run_necessity_test": ["necessity_summary.json"],
    "06_analyze": ["analysis/analysis.json"],
    "07_direction_sweep": ["directions_meta.json"],
    "09_stat_tests": ["analysis/stat_tests.json"],
    "11_insertion_text": ["sufficiency_meta.json"],
    "13_activation_shift": ["analysis/activation_shift.json"],
}

HASHED = [
    "prereg_sufficiency.json",
    "necessity_summary.json",
    "sufficiency_meta.json",
    "event_summary.json",
    "trajectory_summary.json",
    "analysis/stat_tests.json",
    "analysis/activation_shift.json",
    "analysis/event_validation_summary.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    _cfg, cfg_dict, run_dir = script_setup("15_provenance")

    digests = {}
    for rel in HASHED:
        p = run_dir / rel
        digests[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size} if p.exists() else None

    manifest_dir = run_dir / "manifests"
    present = {d.name for d in manifest_dir.iterdir() if d.is_dir()} if manifest_dir.exists() else set()
    missing = {
        stage: outs for stage, outs in STAGE_OUTPUTS.items()
        if stage not in present and any((run_dir / o).exists() for o in outs)
    }

    record = {
        "run_dir": str(run_dir),
        "config_sha256": hashlib.sha256(
            json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest(),
        "file_digests": digests,
        "manifests_present": sorted(present),
        "manifests_missing_for_completed_stages": sorted(missing),
        "note": ("A digest pins WHAT was registered, not WHEN. Publish it with the "
                 "paper so later edits to the registered rules are detectable. "
                 "Stages listed as missing a manifest produced outputs in this "
                 "directory but their manifest was not pulled from the machine "
                 "that ran them — the config behind those rows is unrecorded here."),
    }
    write_json(run_dir / "provenance.json", record)
    print(f"-> {run_dir / 'provenance.json'}")
    for rel, d in digests.items():
        print(f"  {rel}: {d['sha256'][:16] if d else 'MISSING'}")
    if missing:
        print(f"\nWARNING: outputs present but manifest missing for: {sorted(missing)}")
    else:
        print("\nmanifest coverage complete")


if __name__ == "__main__":
    main()
