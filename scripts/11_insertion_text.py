#!/usr/bin/env python
"""Stage 11: text-level INSERTION test (sufficiency; see src/insertion.py).

Two populations, five arms each, all on the vLLM backend with shared
per-request seeds:

  wrong_point    matched 'intended answer wrong, far from every event'
                 truncations of the informative-event chains
  failed_chain   mid-trace truncations of chains whose FINAL answer is wrong
                 (the model never rescued itself — no ceiling on the effect)

  insert_natural / insert_correction/v={assert_mistake,soft_recheck,
  minimal_marker} / insert_filler

Appends rows to continuations.jsonl and registers their intervention keys for
stages 06/09. Each record's meta carries the population and whether the
CONTINUATION verbalizes a correction marker (base rate for stage 12's
induction metric).

  python scripts/11_insertion_text.py --run-dir runs/exp1
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answers import extract_explicit_answer
from src.cli import script_setup
from src.excision import DEFAULT_FILLER
from src.experiment import CONTINUATION_CACHE_KEY, classify_outcome, select_informative
from src.generation import make_fast_backend
from src.insertion import (
    CORRECTION_VARIANTS,
    INSERTION_ARMS,
    failed_chains,
    insertion_prefixes,
    select_failed_chain_point,
    select_insertion_point,
    verbalizes_marker,
)
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
from src.trajectory import load_trajectories, load_transitions

N_FAILED_CHAINS = 60  # failed_chain population size (deterministic prefix)


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("11_insertion_text")
    ncfg = cfg.experiment.necessity
    ccfg = cfg.generation.continuation
    lexicon = cfg.experiment.events.lexicon

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
    print(f"{len(informative)} informative events (same subset as stages 05/07/10)")

    # ---- population 1: wrong points in self-correcting chains -------------
    # (chain, event_id, t_ins, pre_answer, population)
    points: list[tuple] = []
    skipped = 0
    for tr in informative:
        chain, ev = chains[tr.chain_id], events_by_id[tr.event_id]
        sel = select_insertion_point(
            chain, events_by_chain[chain.chain_id], ev,
            ncfg.min_control_pos_distance, trajectories.get(chain.chain_id),
        )
        if sel is None:
            skipped += 1
            continue
        t_ins, pre = sel
        points.append((chain, ev.event_id, t_ins,
                       pre if pre is not None else tr.pre_answer, "wrong_point"))
    print(f"wrong_point: {len(points)} insertion points ({skipped} skipped)")

    # ---- population 2: failed chains (final answer wrong) ------------------
    informative_chain_ids = {tr.chain_id for tr in informative}
    fails = [c for c in failed_chains(sorted(chains.values(), key=lambda c: c.chain_id))
             if c.chain_id not in informative_chain_ids]
    n_p2 = 0
    for chain in fails:
        if n_p2 >= N_FAILED_CHAINS:
            break
        sel = select_failed_chain_point(
            chain, events_by_chain.get(chain.chain_id, []),
            ncfg.min_control_pos_distance,
        )
        if sel is None:
            continue
        t_ins, pre = sel
        points.append((chain, f"{chain.chain_id}__failins", t_ins, pre, "failed_chain"))
        n_p2 += 1
    print(f"failed_chain: {n_p2} insertion points "
          f"(of {len(fails)} failed chains available)")
    if not points:
        sys.exit("no insertion points")

    loaded = load_all(cfg.model, need_fast=True, need_tl=False)
    tokenizer = loaded["tokenizer"]
    backend = make_fast_backend(
        loaded["fast_model"], tokenizer, cfg.model.fast_backend, progress=True
    )

    # changing any spliced text must MISS the cache, never silently reuse
    text_fp = hashlib.sha256(
        ("|".join(f"{k}={v}" for k, v in sorted(CORRECTION_VARIANTS.items()))
         + f"|{DEFAULT_FILLER}").encode()
    ).hexdigest()[:8]

    cache = JsonlCache(run_dir / "continuations.jsonl", CONTINUATION_CACHE_KEY)
    seeds_all = [ccfg.base_seed + k for k in range(ccfg.n_continuations)]

    requests = []  # (chain, event_id, arm, ikey, prefix, seed, pre, t_ins, population)
    new_keys: list[str] = []
    for chain, event_id, t_ins, pre, population in points:
        prefixes = insertion_prefixes(chain, t_ins, tokenizer)
        for arm in INSERTION_ARMS:
            ikey = f"ins_text|{arm}|t={t_ins}|p={population}|c={text_fp}"
            new_keys.append(ikey)
            for seed in seeds_all:
                if cache.get(chain_id=chain.chain_id, event_id=event_id,
                             condition=arm, seed=seed, intervention_key=ikey) is None:
                    requests.append(
                        (chain, event_id, arm, ikey, prefixes[arm], seed, pre, t_ins, population)
                    )
    print(f"{len(requests)} continuations to generate")

    if requests:
        outs = backend.batch_generate(
            [r[4] for r in requests],
            max_new_tokens=ccfg.max_new_tokens,
            temperature=ccfg.temperature,
            top_p=ccfg.top_p,
            seeds=[r[5] for r in requests],
        )
        for (chain, event_id, arm, ikey, prefix, seed, pre, t_ins, population), out in zip(
            requests, outs
        ):
            final = extract_explicit_answer(out.text)
            cache.put(ContinuationRecord(
                chain_id=chain.chain_id, event_id=event_id, condition=arm,
                seed=seed, truncation=len(prefix), ablation_mode="text_insertion",
                position_mask=None, feature_ids=[],
                text=out.text, final_answer=final,
                outcome=classify_outcome(final, pre, chain.gold_answer),
                intervention_key=ikey,
                meta={"gold_answer": chain.gold_answer, "pre_answer": pre,
                      "problem_id": chain.problem_id, "arm": arm,
                      "population": population, "insert_pos": t_ins,
                      "marker_in_continuation": verbalizes_marker(out.text, lexicon)},
            ))

    akp = run_dir / "active_intervention_keys.json"
    active = set(read_json(akp)) if akp.exists() else set()
    active.update(new_keys)
    write_json(akp, sorted(active))
    print(f"insertion test complete -> {run_dir / 'continuations.jsonl'}")


if __name__ == "__main__":
    main()
