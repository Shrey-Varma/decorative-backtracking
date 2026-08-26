#!/usr/bin/env python
"""Stage 10: text-level excision test (see src/excision.py for the design).

Runs the three text arms — with-correction / without-correction /
neutral-filler — on the SAME informative events as stage 05, all on the vLLM
backend with shared per-request seeds (the whole experiment is one batched
submission; no hooks needed, so this is fast and cheap). Appends rows to
continuations.jsonl and registers their intervention keys for stages 06/09.

  python scripts/10_excision_test.py --run-dir runs/exp1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answers import extract_explicit_answer
from src.cli import script_setup
from src.excision import TEXT_ARMS, excision_prefixes
from src.experiment import CONTINUATION_CACHE_KEY, classify_outcome, select_informative
from src.generation import make_fast_backend
from src.io_utils import (
    ContinuationRecord,
    JsonlCache,
    group_events_by_chain,
    load_chains,
    load_events,
    read_json,
    write_json,
)
from src.model_loading import load_all
from src.trajectory import load_transitions


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("10_excision_test")
    ncfg = cfg.experiment.necessity
    ccfg = cfg.generation.continuation

    chains = {c.chain_id: c for c in load_chains(run_dir / "chains.jsonl")}
    events_by_id = {ev.event_id: ev for ev in load_events(run_dir / "events.jsonl")}
    transitions = load_transitions(run_dir / "transitions.jsonl")

    informative = select_informative(transitions)
    informative.sort(key=lambda t: (t.chain_id, t.event_id))
    if ncfg.max_informative_events is not None:
        informative = informative[: ncfg.max_informative_events]
    informative = [t for t in informative if t.chain_id in chains and t.event_id in events_by_id]
    print(f"{len(informative)} informative events (same subset as stages 05/07)")
    if not informative:
        sys.exit("no informative events")

    loaded = load_all(cfg.model, need_fast=True, need_tl=False)
    tokenizer = loaded["tokenizer"]
    backend = make_fast_backend(
        loaded["fast_model"], tokenizer, cfg.model.fast_backend, progress=True
    )

    cache = JsonlCache(run_dir / "continuations.jsonl", CONTINUATION_CACHE_KEY)
    seeds_all = [ccfg.base_seed + k for k in range(ccfg.n_continuations)]

    # build the full request list (one batched submission; vLLM scheduler
    # handles the rest), skipping anything already cached
    requests = []  # (transition, event, arm, ikey, prefix, seed)
    new_keys: list[str] = []
    for tr in informative:
        chain, ev = chains[tr.chain_id], events_by_id[tr.event_id]
        prefixes = excision_prefixes(chain, ev, tokenizer)
        for arm in TEXT_ARMS:
            ikey = f"text|{arm}|t0={ev.onset}|t1={ev.end}"
            new_keys.append(ikey)
            for seed in seeds_all:
                if cache.get(chain_id=chain.chain_id, event_id=ev.event_id,
                             condition=arm, seed=seed, intervention_key=ikey) is None:
                    requests.append((tr, ev, arm, ikey, prefixes[arm], seed))
    print(f"{len(requests)} continuations to generate "
          f"({len(informative) * len(TEXT_ARMS) * len(seeds_all) - len(requests)} cached)")

    if requests:
        outs = backend.batch_generate(
            [r[4] for r in requests],
            max_new_tokens=ccfg.max_new_tokens,
            temperature=ccfg.temperature,
            top_p=ccfg.top_p,
            seeds=[r[5] for r in requests],
        )
        for (tr, ev, arm, ikey, prefix, seed), out in zip(requests, outs):
            chain = chains[tr.chain_id]
            final = extract_explicit_answer(out.text)
            cache.put(ContinuationRecord(
                chain_id=chain.chain_id, event_id=ev.event_id, condition=arm,
                seed=seed, truncation=len(prefix), ablation_mode="text_excision",
                position_mask=None, feature_ids=[],
                text=out.text, final_answer=final,
                outcome=classify_outcome(final, tr.pre_answer, chain.gold_answer),
                intervention_key=ikey,
                meta={"gold_answer": chain.gold_answer, "pre_answer": tr.pre_answer,
                      "problem_id": chain.problem_id, "arm": arm},
            ))

    akp = run_dir / "active_intervention_keys.json"
    active = set(read_json(akp)) if akp.exists() else set()
    active.update(new_keys)
    write_json(akp, sorted(active))
    print(f"excision test complete -> {run_dir / 'continuations.jsonl'}")


if __name__ == "__main__":
    main()
