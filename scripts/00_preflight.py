#!/usr/bin/env python
"""Stage 00 (spec build-order step 1): verify the foundation before spending
GPU-hours on anything downstream.

  --backend tl    load model + SAE on TransformerLens, read hook_name/hook_layer
                  from the SAE config, verify the error-preserving identity and
                  reconstruction quality, check tokenizer parity vs HF
  --backend fast  boot the fast backend (vLLM or HF) and sample a few tokens
  --backend both  both, sequentially (frees the TL model before booting vLLM)

Run the two backends as separate invocations on a single GPU if `both` runs
tight on memory.

  python scripts/00_preflight.py --backend tl
  python scripts/00_preflight.py --backend fast
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


def check_tl(cfg) -> None:
    from transformers import AutoTokenizer

    from src.model_loading import load_all, verify_tokenizer_consistency

    loaded = load_all(cfg.model, need_tl=True)
    print(f"TL + SAE preflight OK: {loaded['recon_stats']}")
    hf_tok = AutoTokenizer.from_pretrained(
        cfg.model.model_name, revision=cfg.model.model_revision
    )
    verify_tokenizer_consistency(
        hf_tok, loaded["tl_model"], "Wait, let me reconsider 12 + 35."
    )
    print("tokenizer parity OK (fast backend tokenizer == TL tokenizer)")
    # free everything before a possible fast-backend boot on the same GPU
    import gc

    import torch

    del loaded
    gc.collect()
    torch.cuda.empty_cache()


def check_fast(cfg) -> None:
    if cfg.model.fast_backend == "vllm":
        from vllm import SamplingParams

        from src.model_loading import load_vllm_model

        llm = load_vllm_model(cfg.model)
        out = llm.generate(
            ["What is 6 * 7? Answer briefly."],
            SamplingParams(temperature=0.0, max_tokens=32),
            use_tqdm=False,
        )
        print(f"vLLM preflight OK: {out[0].outputs[0].text.strip()[:80]!r}")
    else:
        from src.model_loading import load_hf_model

        load_hf_model(cfg.model)
        print("HF fast backend preflight OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend", choices=["tl", "fast", "both"], default="tl")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.backend in ("tl", "both"):
        check_tl(cfg)
    if args.backend in ("fast", "both"):
        check_fast(cfg)
    print("preflight passed")


if __name__ == "__main__":
    main()
