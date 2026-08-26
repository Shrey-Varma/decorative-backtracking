#!/usr/bin/env python
"""Stage 08 (§11): LLM-judge first pass on the event-detection validation
sample.

The lexicon-seeded detector is vocabulary-keyed, so a reviewer will ask what
fraction of flagged events are GENUINE self-corrections vs incidental marker
words ("wait" used non-correctively). This labels the sampled events with an
LLM judge and reports detection precision. This is a FIRST PASS — a proxy for,
not a replacement of, human validation (state that in any writeup).

Runs locally (no GPU). Needs an OpenAI-compatible endpoint:
  configure experiment.verifier.{enabled,base_url,model,api_key_env}, or pass
  --set experiment.verifier.enabled=true experiment.verifier.model=...

  python scripts/08_validate_events.py --run-dir runs/exp1
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import script_setup
from src.events import detection_quality
from src.io_utils import read_json, write_json

JUDGE_PROMPT = (
    "You are auditing an automatic detector of self-CORRECTION events in a "
    "reasoning model's chain of thought. A genuine correction event is where "
    "the model reconsiders and revises a PRIOR step/claim/answer (e.g. catches "
    "a mistake, recomputes, changes direction). Incidental uses of words like "
    "'wait' or 'hmm' that do NOT revise a prior step are NOT corrections.\n\n"
    "Passage (the marker is near the middle):\n---\n{ctx}\n---\n\n"
    "Does this passage show a genuine self-correction? Answer exactly YES or NO."
)


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("08_validate_events")
    vcfg = cfg.experiment.verifier
    path = run_dir / "events_for_validation.json"
    if not path.exists():
        sys.exit("events_for_validation.json not found (run stage 02)")
    sample = read_json(path)

    if not vcfg.enabled or not vcfg.model:
        sys.exit(
            "verifier not configured — set experiment.verifier.enabled=true, "
            ".model=<judge model>, .base_url=<endpoint>, .api_key_env=<VAR>"
        )
    from openai import OpenAI

    client = OpenAI(base_url=vcfg.base_url, api_key=os.environ.get(vcfg.api_key_env, "unset"))

    labeled = 0
    for ev in sample:
        if ev.get("is_correction") is not None:
            continue  # keep any human labels already present
        resp = client.chat.completions.create(
            model=vcfg.model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(ctx=ev["context"])}],
            max_tokens=4, temperature=0.0,
        )
        ans = resp.choices[0].message.content.strip().upper()
        ev["is_correction"] = ans.startswith("YES")
        ev["labeled_by"] = f"llm:{vcfg.model}"
        labeled += 1

    write_json(path, sample)
    q = detection_quality(sample)
    write_json(run_dir / "event_detection_precision.json",
               {**q, "judge": vcfg.model, "note": "LLM first-pass; not human-validated"})
    print(f"labeled {labeled} events by {vcfg.model}")
    print(f"detection precision (LLM first pass): {q}")


if __name__ == "__main__":
    main()
