#!/usr/bin/env python
"""Stage 01 (§9): generate R1-distill chains on the fast backend.

Keeps chains with a parseable final answer (criterion a); event and
wrong->right filters happen in stages 02/04. Persists raw token ids +
per-token char offsets so all downstream truncation points are reproducible.

  python scripts/01_generate_chains.py --run-dir runs/exp1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answers import AnswerVerifier, extract_answer
from src.cli import script_setup
from src.generation import generate_chains, make_fast_backend
from src.io_utils import append_jsonl, set_seed, write_json
from src.model_loading import load_all


def load_problems(cfg) -> list[dict]:
    from datasets import load_dataset

    problems = []
    for ds_cfg in cfg.experiment.datasets:
        ds = load_dataset(ds_cfg.hf_path, split=ds_cfg.split)
        if ds_cfg.limit:
            ds = ds.select(range(min(ds_cfg.limit, len(ds))))
        for i, row in enumerate(ds):
            pid = str(row[ds_cfg.id_field]) if ds_cfg.id_field else str(i)
            problems.append(
                {
                    "problem_id": f"{ds_cfg.name}__{pid}",
                    "dataset": ds_cfg.name,
                    "prompt": row[ds_cfg.problem_field],
                    "gold_answer": str(row[ds_cfg.answer_field]),
                }
            )
    return problems


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("01_generate_chains")
    set_seed(cfg.generation.chain.base_seed)

    problems = load_problems(cfg)
    loaded = load_all(cfg.model, need_fast=True, need_tl=False)
    tokenizer = loaded["tokenizer"]
    # progress=True: the whole run is one cross-problem submission, so
    # vLLM's request-level tqdm is the only progress signal
    backend = make_fast_backend(
        loaded["fast_model"], tokenizer, cfg.model.fast_backend, progress=True
    )

    chains = generate_chains(backend, tokenizer, problems, cfg.generation)

    verifier = AnswerVerifier(run_dir / "answer_cache.jsonl", cfg.experiment.verifier)
    kept, dropped = 0, 0
    out_path = run_dir / "chains.jsonl"
    if out_path.exists():
        out_path.unlink()  # regenerate cleanly instead of appending duplicates
    for chain in chains:
        final = extract_answer(chain.text)
        if final is None:
            dropped += 1
            continue
        v = verifier.verify(chain.problem_id, chain.text, chain.gold_answer)
        chain.final_answer = v["parsed"]
        chain.final_correct = v["correct"]
        append_jsonl(out_path, chain)
        kept += 1

    write_json(
        run_dir / "chain_summary.json",
        {"n_problems": len(problems), "n_generated": len(chains),
         "n_kept_parseable": kept, "n_dropped_unparseable": dropped},
    )
    print(f"kept {kept}/{len(chains)} chains -> {out_path}")


if __name__ == "__main__":
    main()
