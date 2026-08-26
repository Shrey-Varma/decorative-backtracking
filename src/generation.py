"""Chain generation, seeded continuation, force-answer probe (step 3, §6, §9).

Backend split (§2): bulk chain generation runs on a fast backend (HF generate
or vLLM); every *interventional* continuation runs on TransformerLens where
residual-stream hooks work during generation. Both share one HF tokenizer, and
all positions are absolute indices into ChainRecord.token_ids.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from .answers import extract_probe_answer
from .config import ChainGenConfig, ContinuationConfig, GenerationConfig, ProbeConfig
from .io_utils import ChainRecord, ProbeResult
from .sae_interventions import ClampHook

logger = logging.getLogger(__name__)


@dataclass
class GenOut:
    token_ids: list[int]   # completion tokens only
    text: str              # decoded completion


class Backend(Protocol):
    """Minimal generation interface shared by HF / vLLM / TL backends.

    `seeds` (one per prefix) is honored only by backends that advertise
    `supports_per_request_seeds = True`: it makes each request's randomness a
    function of the request alone, independent of batch composition. Backends
    without it (HF) get one `seed` per call covering the whole batch draw."""

    def batch_generate(
        self,
        prefixes: list[list[int]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        greedy: bool = False,
        seeds: Optional[Sequence[int]] = None,
    ) -> list[GenOut]: ...


# ----------------------------------------------------------------- token/char maps


def compute_token_char_offsets(tokenizer, completion_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """Decode the completion and produce per-token char spans into it.

    Fast path: re-encode the decoded text with offset mapping and check the ids
    round-trip. Fallback: incremental decode (always correct, slower)."""
    text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        if list(enc["input_ids"]) == list(completion_ids):
            return text, [tuple(o) for o in enc["offset_mapping"]]
    except Exception:
        pass
    offsets: list[tuple[int, int]] = []
    prev = ""
    for i in range(len(completion_ids)):
        cur = tokenizer.decode(completion_ids[: i + 1], skip_special_tokens=False)
        offsets.append((len(prev), len(cur)))
        prev = cur
    return prev, offsets


def char_to_completion_token(offsets: Sequence[tuple[int, int]], char_idx: int) -> int:
    """Index of the completion token whose span contains char_idx."""
    starts = [s for s, _ in offsets]
    i = bisect_right(starts, char_idx) - 1
    return max(0, i)


def find_trace_span(
    chain_text: str,
    offsets: Sequence[tuple[int, int]],
    prompt_token_len: int,
    think_open: str,
    think_close: str,
) -> tuple[int, int]:
    """Absolute [start, end) token span of the reasoning trace (§3): between
    the think delimiters, excluding the final answer block. If the close tag
    never appears, the whole completion counts as trace (flagged upstream)."""
    open_pos = chain_text.find(think_open)
    if open_pos >= 0:
        start_char = open_pos + len(think_open)
        start_tok = char_to_completion_token(offsets, start_char)
        if offsets[start_tok][0] < start_char:
            start_tok += 1
    else:
        start_tok = 0
    close_pos = chain_text.find(think_close)
    end_tok = (
        char_to_completion_token(offsets, close_pos) if close_pos >= 0 else len(offsets)
    )
    start_tok = min(start_tok, end_tok)
    return (prompt_token_len + start_tok, prompt_token_len + end_tok)


# ----------------------------------------------------------------------- HF backend


def _eos_ids(model, tokenizer) -> list[int]:
    """All terminator ids the model may stop on — generation_config can list
    several (e.g. a chat </think>/end-of-turn token distinct from eos)."""
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    gen_cfg = getattr(model, "generation_config", None)
    eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg else None
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(e) for e in eos)
    return list(ids)


class HFGenerator:
    supports_per_request_seeds = False

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def batch_generate(
        self,
        prefixes: list[list[int]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        greedy: bool = False,
        seeds: Optional[Sequence[int]] = None,
    ) -> list[GenOut]:
        import torch

        if seeds is not None:
            raise ValueError(
                "HF backend samples the whole batch from one RNG and cannot "
                "honor per-request seeds; callers must use the per-batch path"
            )

        device = next(self.model.parameters()).device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        maxlen = max(len(p) for p in prefixes)
        input_ids = torch.full((len(prefixes), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(prefixes), maxlen), dtype=torch.long)
        for i, p in enumerate(prefixes):  # left-pad
            input_ids[i, maxlen - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[i, maxlen - len(p):] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)

        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=temperature if not greedy else None,
                top_p=top_p if not greedy else None,
                pad_token_id=pad_id,
            )
        # a model may stop on any of its configured terminator ids (R1/Llama
        # chat templates often list several); generate() then right-pads the
        # finished row. Cut at the first real terminator (keep it), else strip
        # trailing pads — otherwise pad tokens get baked into
        # ChainRecord.token_ids and throw off every downstream position index.
        stop_ids = set(_eos_ids(self.model, self.tokenizer))
        results = []
        for i in range(len(prefixes)):
            comp = out[i, maxlen:].tolist()
            cut = next((j for j, t in enumerate(comp) if t in stop_ids), None)
            if cut is not None:
                comp = comp[: cut + 1]
            else:
                while comp and comp[-1] == pad_id:
                    comp.pop()
            results.append(GenOut(comp, self.tokenizer.decode(comp, skip_special_tokens=False)))
        return results


class VLLMGenerator:
    """vLLM seeds are PER-REQUEST: a batch of identical prompts under one seed
    would return identical completions. Explicit `seeds` (one per prefix) take
    precedence; otherwise seed+i is derived per request. `progress=True` shows
    vLLM's request-level tqdm — use it for long submissions (stage 01)."""

    supports_per_request_seeds = True

    def __init__(self, llm, tokenizer, progress: bool = False):
        self.llm = llm
        self.tokenizer = tokenizer
        self.progress = progress

    def batch_generate(
        self,
        prefixes: list[list[int]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        greedy: bool = False,
        seeds: Optional[Sequence[int]] = None,
    ) -> list[GenOut]:
        from vllm import SamplingParams

        if seeds is not None and len(seeds) != len(prefixes):
            raise ValueError(f"{len(seeds)} seeds for {len(prefixes)} prefixes")

        def seed_for(i: int) -> Optional[int]:
            if greedy:
                return seed
            if seeds is not None:
                return seeds[i]
            return (seed + i) if seed is not None else None

        params = [
            SamplingParams(
                temperature=0.0 if greedy else temperature,
                top_p=1.0 if greedy else top_p,
                max_tokens=max_new_tokens,
                seed=seed_for(i),
            )
            for i in range(len(prefixes))
        ]
        outs = self.llm.generate(
            [{"prompt_token_ids": list(p)} for p in prefixes],  # TokensPrompt form
            sampling_params=params,
            use_tqdm=self.progress,
        )
        return [
            GenOut(list(o.outputs[0].token_ids), o.outputs[0].text) for o in outs
        ]


# ----------------------------------------------------------------------- TL backend


class TLGenerator:
    """Hookable generation on TransformerLens with an explicit KV cache and an
    explicit absolute-position tracker, so ClampHooks see true positions."""

    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer

    def generate_hooked(
        self,
        prefix_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: Optional[int] = None,
        greedy: bool = False,
        hook_name: Optional[str] = None,
        clamp_hook: Optional[ClampHook] = None,
        early_stop_regex: Optional[str] = None,
        early_stop_check_every: int = 32,
        think_close: Optional[str] = None,
    ) -> GenOut:
        import torch

        # class was renamed HookedTransformerKeyValueCache ->
        # TransformerLensKeyValueCache; support both across TL versions
        try:
            from transformer_lens import TransformerLensKeyValueCache as _KVCache
        except ImportError:
            from transformer_lens import HookedTransformerKeyValueCache as _KVCache

        device = self.model.cfg.device
        fwd_hooks = []
        if clamp_hook is not None:
            assert hook_name is not None
            clamp_hook.tracker.reset()
            fwd_hooks.append((hook_name, clamp_hook))

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        cache = _KVCache.init_cache(
            self.model.cfg, device, batch_size=1
        )
        tokens = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        stop_pattern = re.compile(early_stop_regex) if early_stop_regex else None
        eos_id = self.tokenizer.eos_token_id
        completion: list[int] = []

        with torch.no_grad(), self.model.hooks(fwd_hooks=fwd_hooks):
            logits = self.model(tokens, past_kv_cache=cache)
            for step in range(max_new_tokens):
                next_id = _sample(logits[0, -1], temperature, top_p, greedy, generator)
                completion.append(next_id)
                if next_id == eos_id:
                    break
                if stop_pattern and (step + 1) % early_stop_check_every == 0:
                    text_so_far = self.tokenizer.decode(completion, skip_special_tokens=False)
                    scope = text_so_far
                    if think_close:
                        idx = text_so_far.find(think_close)
                        scope = text_so_far[idx + len(think_close):] if idx >= 0 else ""
                    if stop_pattern.search(scope):
                        break
                step_tokens = torch.tensor([[next_id]], dtype=torch.long, device=device)
                logits = self.model(step_tokens, past_kv_cache=cache)

        return GenOut(completion, self.tokenizer.decode(completion, skip_special_tokens=False))

    def generate_hooked_batch(
        self,
        prefix_ids: list[int],
        seeds: Sequence[int],
        max_new_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        greedy: bool = False,
        hook_name: Optional[str] = None,
        clamp_hook: Optional[ClampHook] = None,
        early_stop_regex: Optional[str] = None,
        early_stop_check_every: int = 32,
        think_close: Optional[str] = None,
    ) -> list[GenOut]:
        """K seeded continuations from ONE prefix in ONE batched decode.

        Single-sequence decode is memory-bandwidth bound (all weights stream
        per token for one sequence); the K seeds of an (event, condition) share
        the same prefix and the same clamp mask, so decoding them as a batch is
        a ~Kx throughput win with no change to the intervention semantics:

        - prefill runs once at batch 1 (bounds attention memory on long
          prefixes), then the cached prefix K/V is replicated across rows;
        - each row samples from its own CPU generator (per-seed determinism,
          same draw algorithm as the single path);
        - the clamp hook is position-keyed and batch-agnostic: during decode it
          sees [B, 1, d_model] at one absolute position and clamps all rows,
          exactly matching K single runs;
        - finished rows (EOS / early-stop) are fed EOS as padding input and
          their completions frozen, until all rows finish or the cap hits.

        Note batched kernels reduce in a different order than batch-1 kernels,
        so a batched continuation is not bit-identical to its unbatched
        counterpart — same class of bf16 noise as any resampling; all
        conditions run under the same regime so comparisons stay paired.
        """
        import torch

        try:
            from transformer_lens import TransformerLensKeyValueCache as _KVCache
        except ImportError:
            from transformer_lens import HookedTransformerKeyValueCache as _KVCache

        n_rows = len(seeds)
        device = self.model.cfg.device
        fwd_hooks = []
        if clamp_hook is not None:
            assert hook_name is not None
            clamp_hook.tracker.reset()
            fwd_hooks.append((hook_name, clamp_hook))

        generators = [
            torch.Generator(device="cpu").manual_seed(int(s)) for s in seeds
        ]
        stop_pattern = re.compile(early_stop_regex) if early_stop_regex else None
        eos_id = self.tokenizer.eos_token_id

        cache = _KVCache.init_cache(self.model.cfg, device, batch_size=1)
        tokens = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        completions: list[list[int]] = [[] for _ in range(n_rows)]
        finished = [False] * n_rows

        with torch.no_grad(), self.model.hooks(fwd_hooks=fwd_hooks):
            logits = self.model(tokens, past_kv_cache=cache)  # prefill once
            _replicate_kv_cache(cache, n_rows)
            last_logits = logits[0, -1].unsqueeze(0).expand(n_rows, -1)
            for step in range(max_new_tokens):
                sampled = _sample_batch(
                    last_logits, temperature, top_p, greedy, generators, finished
                )
                next_ids = []
                for i in range(n_rows):
                    if finished[i]:
                        next_ids.append(eos_id)  # pad input; output ignored
                        continue
                    nid = sampled[i]
                    completions[i].append(nid)
                    if nid == eos_id:
                        finished[i] = True
                    next_ids.append(nid)
                if stop_pattern and (step + 1) % early_stop_check_every == 0:
                    for i in range(n_rows):
                        if finished[i]:
                            continue
                        text_so_far = self.tokenizer.decode(
                            completions[i], skip_special_tokens=False
                        )
                        scope = text_so_far
                        if think_close:
                            idx = text_so_far.find(think_close)
                            scope = (
                                text_so_far[idx + len(think_close):] if idx >= 0 else ""
                            )
                        if stop_pattern.search(scope):
                            finished[i] = True
                if all(finished):
                    break
                step_tokens = torch.tensor(
                    next_ids, dtype=torch.long, device=device
                ).unsqueeze(1)
                last_logits = self.model(step_tokens, past_kv_cache=cache)[:, -1]

        return [
            GenOut(
                completions[i],
                self.tokenizer.decode(completions[i], skip_special_tokens=False),
            )
            for i in range(n_rows)
        ]

    supports_per_request_seeds = True

    def batch_generate(
        self,
        prefixes: list[list[int]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        greedy: bool = False,
        seeds: Optional[Sequence[int]] = None,
    ) -> list[GenOut]:
        if seeds is not None and len(seeds) != len(prefixes):
            raise ValueError(f"{len(seeds)} seeds for {len(prefixes)} prefixes")
        return [
            self.generate_hooked(
                p, max_new_tokens, temperature=temperature, top_p=top_p,
                seed=seeds[i] if seeds is not None else seed, greedy=greedy,
            )
            for i, p in enumerate(prefixes)
        ]


class HFHookedGenerator:
    """Batched hooked generation on a plain HF model — the fast interventional
    backend for stage 05.

    The knockout only needs to EDIT the layer-`hook_layer` residual stream,
    and an nn.Module forward hook on `model.model.layers[hook_layer]` does
    exactly that: a Llama decoder layer's output[0] IS the post-block residual
    (== TL's blocks.N.hook_resid_post; equivalence validated at cosine 0.9999
    per position). HF's fused-SDPA decode is ~10-20x faster per step than the
    TransformerLens loop at long context, which is what makes the full
    necessity grid affordable.

    Reuses ClampHook verbatim (it is batch-agnostic and position-keyed) via a
    thin adapter to the nn.Module hook signature, and _sample_batch for
    per-row seeded sampling — same draw algorithm, same determinism contract
    as the TL path.
    """

    def __init__(self, model, tokenizer, hook_layer: int):
        self.model = model
        self.tokenizer = tokenizer
        self.hook_layer = hook_layer

    def generate_hooked_batch(
        self,
        prefix_ids: list[int],
        seeds: Sequence[int],
        max_new_tokens: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        greedy: bool = False,
        clamp_hook=None,
        early_stop_regex: Optional[str] = None,
        early_stop_check_every: int = 32,
        think_close: Optional[str] = None,
        hook_layer: Optional[int] = None,
    ) -> list[GenOut]:
        import torch
        from transformers import DynamicCache

        n_rows = len(seeds)
        device = next(self.model.parameters()).device
        generators = [
            torch.Generator(device="cpu").manual_seed(int(s)) for s in seeds
        ]
        stop_pattern = re.compile(early_stop_regex) if early_stop_regex else None
        eos_id = self.tokenizer.eos_token_id

        # clamp_hook is any position-keyed resid editor with a .tracker
        # (ClampHook for SAE features, DirectionAblationHook for the diff-in-
        # means sweep); hook_layer lets the sweep target other layers.
        layer = self.hook_layer if hook_layer is None else hook_layer
        handle = None
        if clamp_hook is not None:
            clamp_hook.tracker.reset()

            def _adapter(module, args, output):
                # decoder layers return (hidden_states, ...) in older
                # transformers and the bare hidden_states tensor in newer ones
                if isinstance(output, tuple):
                    return (clamp_hook(output[0]),) + tuple(output[1:])
                return clamp_hook(output)

            handle = self.model.model.layers[layer].register_forward_hook(_adapter)

        completions: list[list[int]] = [[] for _ in range(n_rows)]
        finished = [False] * n_rows

        def _forward(tok):
            # logits_to_keep=1: only run the LM head on the LAST position. Without
            # it, prefilling an ~7k-token event x 8 rows materializes an
            # 8 x 7k x 128k-vocab logits tensor (~14 GiB) and OOMs the 48GB GPU,
            # when we only ever use the final position's logits.
            try:
                return self.model(tok, past_key_values=cache, use_cache=True,
                                  logits_to_keep=1)
            except TypeError:  # older transformers spelling
                return self.model(tok, past_key_values=cache, use_cache=True,
                                  num_logits_to_keep=1)

        try:
            cache = DynamicCache()
            tokens = torch.tensor([prefix_ids] * n_rows, dtype=torch.long, device=device)
            with torch.no_grad():
                out = _forward(tokens)
                last_logits = out.logits[:, -1]
                for step in range(max_new_tokens):
                    sampled = _sample_batch(
                        last_logits, temperature, top_p, greedy, generators, finished
                    )
                    next_ids = []
                    for i in range(n_rows):
                        if finished[i]:
                            next_ids.append(eos_id)
                            continue
                        nid = sampled[i]
                        completions[i].append(nid)
                        if nid == eos_id:
                            finished[i] = True
                        next_ids.append(nid)
                    if stop_pattern and (step + 1) % early_stop_check_every == 0:
                        for i in range(n_rows):
                            if finished[i]:
                                continue
                            text_so_far = self.tokenizer.decode(
                                completions[i], skip_special_tokens=False
                            )
                            scope = text_so_far
                            if think_close:
                                idx = text_so_far.find(think_close)
                                scope = (
                                    text_so_far[idx + len(think_close):]
                                    if idx >= 0 else ""
                                )
                            if stop_pattern.search(scope):
                                finished[i] = True
                    if all(finished):
                        break
                    step_tokens = torch.tensor(
                        next_ids, dtype=torch.long, device=device
                    ).unsqueeze(1)
                    out = _forward(step_tokens)
                    last_logits = out.logits[:, -1]
        finally:
            if handle is not None:
                handle.remove()

        return [
            GenOut(
                completions[i],
                self.tokenizer.decode(completions[i], skip_special_tokens=False),
            )
            for i in range(n_rows)
        ]


def _replicate_kv_cache(cache, batch_size: int) -> None:
    """Expand a batch-1 TL KV cache to `batch_size` rows in place.

    The K seeded continuations of one (event, condition) share an identical
    prefix, so we prefill ONCE at batch 1 (bounding attention memory) and
    repeat the cached prefix K/V across rows for the batched decode."""
    if batch_size == 1:
        return
    for entry in cache.entries:
        for attr in ("past_keys", "past_values"):
            t = getattr(entry, attr)
            setattr(entry, attr, t.repeat(batch_size, *([1] * (t.ndim - 1))))
    mask = getattr(cache, "previous_attention_mask", None)
    if mask is not None:
        cache.previous_attention_mask = mask.repeat(
            batch_size, *([1] * (mask.ndim - 1))
        )


def _sample_batch(
    logits,  # [B, vocab]
    temperature: float,
    top_p: float,
    greedy: bool,
    generators: list,          # one CPU generator per row (per-seed determinism)
    skip: list[bool],          # finished rows: don't draw, don't advance their RNG
) -> list[Optional[int]]:
    import torch

    B = logits.shape[0]
    if greedy or temperature <= 0:
        ids = logits.argmax(-1)
        return [None if skip[i] else int(ids[i].item()) for i in range(B)]
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_idx = probs.sort(-1, descending=True)  # one batched sort
    cum = sorted_probs.cumsum(-1)
    out: list[Optional[int]] = []
    for i in range(B):
        if skip[i]:
            out.append(None)
            continue
        cutoff = int((cum[i] < top_p).sum().item()) + 1
        kept = sorted_probs[i, :cutoff].cpu()
        pick = torch.multinomial(kept / kept.sum(), 1, generator=generators[i])
        out.append(int(sorted_idx[i, int(pick.item())].item()))
    return out


def _sample(logits, temperature: float, top_p: float, greedy: bool, generator) -> int:
    import torch

    if greedy or temperature <= 0:
        return int(logits.argmax().item())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        # transfer only the nucleus to CPU (typically <1k of ~128k entries):
        # the CPU generator keeps sampling deterministic without paying a
        # full-vocab sync + copy on every generated token
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cum = sorted_probs.cumsum(-1)
        cutoff = int((cum < top_p).sum().item()) + 1
        kept = sorted_probs[:cutoff].cpu()
        pick = torch.multinomial(kept / kept.sum(), 1, generator=generator)
        return int(sorted_idx[int(pick.item())].item())
    idx = torch.multinomial(probs.cpu(), 1, generator=generator)
    return int(idx.item())


def make_fast_backend(fast_model, tokenizer, fast_backend: str, progress: bool = False) -> Backend:
    """One construction point for the fast backend so stages 01 and 04 can't
    drift (constructor arguments, backend additions)."""
    if fast_backend == "vllm":
        return VLLMGenerator(fast_model, tokenizer, progress=progress)
    return HFGenerator(fast_model, tokenizer)


# ------------------------------------------------------------------ chain assembly


def build_prompt_ids(tokenizer, problem: str) -> list[int]:
    messages = [{"role": "user", "content": problem}]
    out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    # normalize to a flat List[int]: depending on transformers version / chat
    # template, apply_chat_template can return a BatchEncoding or a nested
    # [[...]] batch instead of a bare list — iterating those yields dict keys
    # (strings) or inner lists, which corrupts the token-id prompt downstream.
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if len(out) and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def make_chain_record(
    tokenizer,
    gen_cfg: GenerationConfig,
    chain_id: str,
    problem_id: str,
    dataset: str,
    prompt: str,
    gold_answer: str,
    seed: int,
    prompt_ids: list[int],
    out: GenOut,
    generation_config: dict,
) -> ChainRecord:
    text, offsets = compute_token_char_offsets(tokenizer, out.token_ids)
    trace_span = find_trace_span(
        text, offsets, len(prompt_ids), gen_cfg.think_open, gen_cfg.think_close
    )
    return ChainRecord(
        chain_id=chain_id,
        problem_id=problem_id,
        dataset=dataset,
        prompt=prompt,
        gold_answer=gold_answer,
        seed=seed,
        token_ids=list(prompt_ids) + list(out.token_ids),
        prompt_token_len=len(prompt_ids),
        text=text,
        token_char_offsets=offsets,
        trace_token_span=trace_span,
        generation_config=generation_config,
    )


def generate_chains(
    backend: Backend,
    tokenizer,
    problems: list[dict],   # {problem_id, dataset, prompt, gold_answer}
    gen_cfg: GenerationConfig,
) -> list[ChainRecord]:
    """Bulk chain generation on the fast backend (§9).

    Seed-capable backends (vLLM, TL) get every (problem, chain) request in ONE
    submission — cross-problem batching keeps the engine's scheduler saturated
    instead of feeding it n_chains_per_problem-wide batches. Each request's
    seed is keyed to (problem_index, chain_index), so seed assignment is a pure
    function of the request, independent of batch composition. The HF fallback
    keeps the per-problem loop (one torch seed covers a whole batch draw there).

    Either way, chains persist raw token ids + offsets: nothing downstream
    depends on regeneration, only on what was actually sampled.
    """
    cc: ChainGenConfig = gen_cfg.chain
    base_cfg = {
        "temperature": cc.temperature, "top_p": cc.top_p,
        "max_new_tokens": cc.max_new_tokens,
    }
    chains: list[ChainRecord] = []

    if getattr(backend, "supports_per_request_seeds", False):
        prompt_ids_by_problem = [build_prompt_ids(tokenizer, p["prompt"]) for p in problems]
        requests = [
            (p_idx, j, cc.base_seed + p_idx * cc.n_chains_per_problem + j)
            for p_idx in range(len(problems))
            for j in range(cc.n_chains_per_problem)
        ]
        outs = backend.batch_generate(
            [prompt_ids_by_problem[p_idx] for p_idx, _, _ in requests],
            max_new_tokens=cc.max_new_tokens,
            temperature=cc.temperature,
            top_p=cc.top_p,
            seeds=[s for _, _, s in requests],
        )
        for (p_idx, j, s), out in zip(requests, outs):
            prob = problems[p_idx]
            chains.append(
                make_chain_record(
                    tokenizer, gen_cfg,
                    chain_id=f"{prob['problem_id']}__c{j}",
                    problem_id=prob["problem_id"],
                    dataset=prob["dataset"],
                    prompt=prob["prompt"],
                    gold_answer=prob["gold_answer"],
                    seed=s,
                    prompt_ids=prompt_ids_by_problem[p_idx],
                    out=out,
                    generation_config={
                        **base_cfg, "seed_scheme": "per_request",
                        "seed": s, "chain_index": j,
                    },
                )
            )
        return chains

    for p_idx, prob in enumerate(problems):
        prompt_ids = build_prompt_ids(tokenizer, prob["prompt"])
        seed = cc.base_seed + p_idx
        outs = backend.batch_generate(
            [prompt_ids] * cc.n_chains_per_problem,
            max_new_tokens=cc.max_new_tokens,
            temperature=cc.temperature,
            top_p=cc.top_p,
            seed=seed,
        )
        for j, out in enumerate(outs):
            chains.append(
                make_chain_record(
                    tokenizer, gen_cfg,
                    chain_id=f"{prob['problem_id']}__c{j}",
                    problem_id=prob["problem_id"],
                    dataset=prob["dataset"],
                    prompt=prob["prompt"],
                    gold_answer=prob["gold_answer"],
                    seed=seed,
                    prompt_ids=prompt_ids,
                    out=out,
                    generation_config={
                        **base_cfg, "seed_scheme": "hf_batch",
                        "batch_seed": seed, "batch_index": j,
                    },
                )
            )
    return chains


# -------------------------------------------------------------- force-answer probe


def batch_force_answer_probe(
    backend: Backend,
    tokenizer,
    chain: ChainRecord,
    positions: list[int],
    probe_cfg: ProbeConfig,
) -> list[ProbeResult]:
    template_ids = tokenizer.encode(probe_cfg.template, add_special_tokens=False)
    prefixes = [chain.token_ids[:t] + template_ids for t in positions]
    outs = backend.batch_generate(
        prefixes, max_new_tokens=probe_cfg.max_new_tokens, greedy=True
    )
    results = []
    for t, out in zip(positions, outs):
        parsed = extract_probe_answer(out.text, probe_cfg.stop_string)
        results.append(
            ProbeResult(chain_id=chain.chain_id, position=t, raw=out.text, parsed=parsed)
        )
    return results


def make_cached_batch_prober(backend: Backend, tokenizer, probe_cfg: ProbeConfig, cache):
    """BatchProber (trajectory.py) backed by the probe JsonlCache, so every
    (chain, position) is probed exactly once across scripts 03/04. Cache keys
    include a template fingerprint: probes are only comparable under one fixed
    template (§6)."""
    import hashlib

    # the cached `parsed` value depends on stop_string too (via
    # extract_probe_answer), so it must be part of the fingerprint — otherwise
    # changing stop_string returns answers parsed under the old setting
    template_hash = hashlib.sha256(
        f"{probe_cfg.template}|{probe_cfg.max_new_tokens}|{probe_cfg.stop_string}".encode()
    ).hexdigest()[:8]

    def batch_prober(chain: ChainRecord, positions: list[int]) -> dict[int, Optional[str]]:
        answers: dict[int, Optional[str]] = {}
        missing = []
        for t in positions:
            hit = cache.get(chain_id=chain.chain_id, position=t, template=template_hash)
            if hit is not None:
                answers[t] = hit["parsed"]
            else:
                missing.append(t)
        if missing:
            for res in batch_force_answer_probe(backend, tokenizer, chain, missing, probe_cfg):
                cache.put(
                    {
                        "chain_id": res.chain_id,
                        "position": res.position,
                        "template": template_hash,
                        "raw": res.raw,
                        "parsed": res.parsed,
                    }
                )
                answers[res.position] = res.parsed
        return answers

    return batch_prober
