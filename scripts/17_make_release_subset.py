#!/usr/bin/env python
"""Stage 17: the small derived files a code release needs to rebuild figures.

`chains.jsonl` (61 MB) and `analysis/continuations_flat.csv` (162 MB) are far
too large to ship, but the figure scripts only touch a narrow slice of each:
gold answers keyed by chain, and five columns of the insertion rows. This
writes those slices to `release/`, which the figure scripts fall back to when
the full files are absent.

  python scripts/17_make_release_subset.py --run-dir results/exp1
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default="results/exp1")
args = ap.parse_args()
RD = Path(args.run_dir)
OUT = RD / "release"
OUT.mkdir(parents=True, exist_ok=True)

gold = {}
with (RD / "chains.jsonl").open() as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            gold[d["chain_id"]] = d["gold_answer"]
(OUT / "gold_answers.json").write_text(json.dumps(gold, sort_keys=True))
print(f"gold_answers.json      {len(gold):5d} chains  "
      f"{(OUT / 'gold_answers.json').stat().st_size / 1024:.0f} KB")

cols = ["chain_id", "condition", "event_id", "meta", "is_correct"]
df = pd.read_csv(RD / "analysis/continuations_flat.csv", usecols=cols)
ins = df[df.condition.str.startswith("insert_")]
ins.to_csv(OUT / "insertion_rows.csv", index=False)
print(f"insertion_rows.csv     {len(ins):5d} rows    "
      f"{(OUT / 'insertion_rows.csv').stat().st_size / 1024 / 1024:.1f} MB")
