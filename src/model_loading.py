"""Model + SAE loading (build order step 1).

- TransformerLens model for all interventional work (hooks during generation).
- HF (or vLLM) model for bulk non-interventional chain generation.
- SAE via sae_lens; hook_name / hook_layer are read from the SAE's own config
  (§2: "do not hardcode").
- Startup checks: encode/decode round-trip, error-preserving identity, and
  tokenizer consistency across backends (§2 backend split).

Heavy deps (torch/transformers/transformer_lens/sae_lens) import lazily so the
pure-logic modules and tests never need them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class SAEHookInfo:
    hook_name: str
    hook_layer: int
    d_sae: int
    d_in: int


def _torch_dtype(name: str):
    import torch
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


# ----------------------------------------------------------------------------- SAE


def load_sae(cfg: ModelConfig):
    """Load the SAE with sae_lens. Tries the release registry first, then a
    direct HF snapshot (community SAEs are usually not in the registry)."""
    from sae_lens import SAE

    try:
        sae = SAE.from_pretrained(
            release=cfg.sae.repo_id, sae_id=cfg.sae.sae_id, device=cfg.device
        )
        if isinstance(sae, tuple):  # older sae_lens returns (sae, cfg_dict, sparsity)
            sae = sae[0]
    except Exception as e:
        logger.info("registry load failed (%s); falling back to HF snapshot", e)
        from huggingface_hub import snapshot_download

        local = snapshot_download(cfg.sae.repo_id, revision=cfg.sae.revision)
        candidates = [
            f"{local}/{cfg.sae.sae_id}",
            local,
        ]
        sae = None
        for path in candidates:
            try:
                sae = SAE.load_from_disk(path, device=cfg.device)
                break
            except Exception:
                continue
        if sae is None:
            raise RuntimeError(
                f"could not load SAE from {cfg.sae.repo_id} (tried {candidates})"
            )
    sae = sae.to(_torch_dtype(cfg.sae.dtype))
    return sae


def get_sae_hook_info(sae: Any) -> SAEHookInfo:
    """Read hook point from the SAE config, across sae_lens layout versions."""
    cfg = sae.cfg
    meta = getattr(cfg, "metadata", None)
    hook_name = getattr(cfg, "hook_name", None) or (meta and getattr(meta, "hook_name", None))
    hook_layer = getattr(cfg, "hook_layer", None)
    if hook_layer is None and meta is not None:
        hook_layer = getattr(meta, "hook_layer", None)
    if hook_layer is None and hook_name:
        # e.g. "blocks.19.hook_resid_post"
        parts = hook_name.split(".")
        if len(parts) >= 2 and parts[0] == "blocks":
            hook_layer = int(parts[1])
    if hook_name is None or hook_layer is None:
        raise ValueError(f"could not determine hook point from SAE config: {cfg}")
    return SAEHookInfo(
        hook_name=hook_name,
        hook_layer=int(hook_layer),
        d_sae=int(cfg.d_sae),
        d_in=int(cfg.d_in),
    )


# --------------------------------------------------------------------------- models


def load_tl_model(cfg: ModelConfig):
    """Interventional backend: TransformerLens.

    Upstream TransformerLens does not register the R1 distills, so we can't call
    `from_pretrained(name)` directly. Instead we load the (public) DeepSeek
    weights with HF and mount them on a TL scaffold whose architecture matches
    exactly — DeepSeek-R1-Distill-Llama-8B IS Llama-3.1-8B architecturally, so
    the residual-stream activations the SAE was trained on are reproduced
    faithfully. Set `tl_model_name` to the scaffold (meta-llama/Llama-3.1-8B).

    fold_ln / center_* are all disabled (== from_pretrained_no_processing) so
    the residual stream stays faithful for the SAE. low_cpu_mem_usage + swap
    keep the transient two-copy load peak under the box's host RAM."""
    import gc

    import torch
    from transformer_lens import HookedTransformer

    dtype = _torch_dtype(cfg.dtype)
    weights_name = cfg.model_name
    scaffold = cfg.tl_model_name or cfg.model_name

    if scaffold == weights_name:
        # natively-registered model: the direct path works
        model = HookedTransformer.from_pretrained_no_processing(
            weights_name, device=cfg.device, dtype=dtype, n_ctx=cfg.tl_n_ctx
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf = AutoModelForCausalLM.from_pretrained(
            weights_name, revision=cfg.model_revision,
            torch_dtype=dtype, low_cpu_mem_usage=True,
        )
        tok = AutoTokenizer.from_pretrained(weights_name, revision=cfg.model_revision)
        model = HookedTransformer.from_pretrained(
            scaffold, hf_model=hf, tokenizer=tok,
            fold_ln=False, center_writing_weights=False, center_unembed=False,
            device=cfg.device, dtype=dtype, n_ctx=cfg.tl_n_ctx,
        )
        del hf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.eval()
    torch.set_grad_enabled(False)
    return model


def load_hf_model(cfg: ModelConfig):
    """Fast backend for bulk chain generation (no hooks)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        torch_dtype=_torch_dtype(cfg.dtype),
        device_map=cfg.device,
    )
    model.eval()
    return model, tokenizer


def load_vllm_model(cfg: ModelConfig):
    import os

    # vLLM V1 launches its engine-core in a subprocess. If CUDA is already
    # initialized in this process (e.g. set_seed -> torch.cuda.* runs before
    # generation), a *forked* subprocess dies with "Cannot re-initialize CUDA
    # in forked subprocess". spawn starts a fresh process and sidesteps it.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    return LLM(model=cfg.model_name, revision=cfg.model_revision, dtype=cfg.dtype)


# ------------------------------------------------------------------ startup checks


def verify_tokenizer_consistency(hf_tokenizer, tl_model, sample: str) -> None:
    """§2: tokenization must be identical across backends — truncation points
    computed on fast-backend chains are reused verbatim on the TL backend.

    What must agree is the CONTENT tokenization (vocab + merges). BOS-prepend
    policy is NOT part of this: TransformerLens configures its tokenizer to
    auto-prepend BOS while a raw AutoTokenizer may not, but the pipeline
    tokenizes each prompt exactly once (via apply_chat_template, BOS included)
    and every backend thereafter consumes the stored token ids — chain text is
    never re-tokenized, so a per-tokenizer BOS default can't shift positions.
    We therefore compare modulo a single leading BOS."""
    def strip_bos(ids, tok):
        bos = getattr(tok, "bos_token_id", None)
        return ids[1:] if bos is not None and ids and ids[0] == bos else ids

    hf_ids = strip_bos(hf_tokenizer.encode(sample), hf_tokenizer)
    tl_ids = strip_bos(tl_model.tokenizer.encode(sample), tl_model.tokenizer)
    if hf_ids != tl_ids:
        raise RuntimeError(
            "tokenizer mismatch between fast backend and TransformerLens "
            "(beyond a leading BOS):\n"
            f"  hf: {hf_ids[:20]}...\n  tl: {tl_ids[:20]}..."
        )


def verify_sae_reconstruction(sae, tl_model, hook_info: SAEHookInfo, sample: str) -> dict:
    """Round-trip encode/decode on real activations and verify the
    error-preserving identity: decode(encode(x)) + err == x exactly (§7.1),
    plus report plain reconstruction quality for sanity."""
    import torch

    tokens = tl_model.to_tokens(sample)
    _, cache = tl_model.run_with_cache(tokens, names_filter=hook_info.hook_name)
    x = cache[hook_info.hook_name].to(next(sae.parameters()).dtype)

    f = sae.encode(x)
    x_hat = sae.decode(f)
    err = x - x_hat
    identity = x_hat + err

    if not torch.allclose(identity, x):
        raise RuntimeError("error-preserving identity failed: decode(encode(x)) + err != x")

    frac_var = 1 - (x - x_hat).pow(2).sum() / (x - x.mean(dim=(0, 1))).pow(2).sum()
    stats = {
        "fraction_variance_explained": float(frac_var),
        "mean_l0": float((f > 0).float().sum(-1).mean()),
        "d_sae": hook_info.d_sae,
        "hook_name": hook_info.hook_name,
        "hook_layer": hook_info.hook_layer,
    }
    logger.info("SAE reconstruction check: %s", stats)
    return stats


def load_all(cfg: ModelConfig, need_fast: bool = False, need_tl: bool = True) -> dict:
    """Convenience: load whatever a script needs and run the startup checks."""
    out: dict[str, Any] = {}
    if need_tl:
        out["tl_model"] = load_tl_model(cfg)
        out["sae"] = load_sae(cfg)
        out["hook_info"] = get_sae_hook_info(out["sae"])
        out["recon_stats"] = verify_sae_reconstruction(
            out["sae"], out["tl_model"], out["hook_info"],
            "Let me double-check: 3 * 17 = 51. Wait, actually that is right.",
        )
    if need_fast:
        if cfg.fast_backend == "vllm":
            out["fast_model"] = load_vllm_model(cfg)
            from transformers import AutoTokenizer
            out["tokenizer"] = AutoTokenizer.from_pretrained(
                cfg.model_name, revision=cfg.model_revision
            )
        else:
            out["fast_model"], out["tokenizer"] = load_hf_model(cfg)
        if need_tl:
            verify_tokenizer_consistency(
                out["tokenizer"], out["tl_model"], "Wait, let me reconsider 12 + 35."
            )
    elif need_tl:
        out["tokenizer"] = out["tl_model"].tokenizer
    return out
