"""Candidate correction-feature scoring + selection (build order step 6, §5).

Two independent scores per SAE feature:

1. vocabulary-association — differential activation on correction-marker
   tokens vs position-matched non-marker trace tokens (necessary, not
   sufficient: a pure "fires on the word wait" feature is exactly the
   decorative case).
2. answer-predictive — AUROC of the feature's activation around an event for
   predicting whether the intended answer actually changes across the event
   (labels come from force-answer probes, §6).

Selection: rank by vocab score for the shortlist, carry the top
`n_candidate_features` into the causal test, record BOTH scores — the causal
test, not either score, decides faithfulness.

Activation access is abstracted as `encode_tokens(token_ids) -> [n, d_sae]`
so tests inject synthetic activations and the real runner wraps TL+SAE.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

from .config import FeaturesConfig
from .io_utils import ChainRecord, EventRecord, write_json

logger = logging.getLogger(__name__)

EncodeFn = Callable[[Sequence[int]], torch.Tensor]  # token_ids -> [n_tokens, d_sae]


def make_tl_encode_fn(
    tl_model, sae, hook_info, batch_tokens: int = 4096, context_overlap: int = 512
) -> EncodeFn:
    """Real activation runner: residual stream at the SAE hook -> SAE features.

    Long sequences are chunked to bound memory; each chunk is re-run with
    `context_overlap` tokens of left context (dropped from the output) so
    activations near chunk starts aren't computed as if the chunk began the
    sequence. Exact for context windows longer than the overlap is not
    guaranteed — raise `context_overlap` (or `batch_tokens`) if that matters."""

    def encode_tokens(token_ids: Sequence[int]) -> torch.Tensor:
        chunks = []
        ids = list(token_ids)
        for lo in range(0, len(ids), batch_tokens):
            ctx = min(lo, context_overlap)
            window = ids[lo - ctx: lo + batch_tokens]
            tokens = torch.tensor([window], dtype=torch.long, device=tl_model.cfg.device)
            _, cache = tl_model.run_with_cache(
                tokens,
                names_filter=hook_info.hook_name,
                # the SAE reads one residual site — layers above it and the
                # unembed are dead weight (~1.7x speedup on every encode)
                stop_at_layer=hook_info.hook_layer + 1,
            )
            x = cache[hook_info.hook_name][0].to(next(sae.parameters()).dtype)
            f = sae.encode(x)
            chunks.append(f[ctx:])
        return torch.cat(chunks, dim=0).cpu()

    return encode_tokens


# ------------------------------------------------------------------ streaming stats


@dataclass
class _GroupAcc:
    total: torch.Tensor
    sq: torch.Tensor
    n: int = 0

    @classmethod
    def zeros(cls, d: int) -> "_GroupAcc":
        return cls(torch.zeros(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64))

    def add(self, f: torch.Tensor) -> None:  # f: [k, d_sae]
        # upcast BEFORE reducing: summing thousands of bf16/fp16 activations in
        # native precision stalls the accumulator, skewing corpus_mean (the
        # mean-ablation target) and making var() = E[x^2]-mean^2 inconsistent
        fd = f.double()
        self.total += fd.sum(0)
        self.sq += (fd ** 2).sum(0)
        self.n += f.shape[0]

    def mean(self) -> torch.Tensor:
        return self.total / max(self.n, 1)

    def var(self) -> torch.Tensor:
        m = self.mean()
        return (self.sq / max(self.n, 1)) - m**2


@dataclass
class CorpusStats:
    corpus_mean: torch.Tensor        # [d_sae] mean over all trace tokens (mean ablation §7.1)
    activation_freq: torch.Tensor    # [d_sae] fraction of trace tokens with f > 0
    vocab_score: torch.Tensor        # [d_sae] Cohen's-d of marker vs matched non-marker
    marker_mean: torch.Tensor
    nonmarker_mean: torch.Tensor
    n_marker_tokens: int
    n_nonmarker_tokens: int
    n_chains: int


def _trace_rel_range(chain: ChainRecord) -> range:
    s, e = chain.trace_token_span
    return range(s - chain.prompt_token_len, e - chain.prompt_token_len)


def _marker_rel_positions(chain: ChainRecord, events: list[EventRecord]) -> set[int]:
    out: set[int] = set()
    for ev in events:
        for t in range(ev.onset, ev.end + 1):
            out.add(t - chain.prompt_token_len)
    return out


def compute_corpus_stats(
    chains: list[ChainRecord],
    events_by_chain: dict[str, list[EventRecord]],
    encode_fn: EncodeFn,
    cfg: FeaturesConfig,
    seed: int = 0,
) -> CorpusStats:
    return compute_corpus_stats_and_windows(chains, events_by_chain, encode_fn, cfg, seed)[0]


def compute_corpus_stats_and_windows(
    chains: list[ChainRecord],
    events_by_chain: dict[str, list[EventRecord]],
    encode_fn: EncodeFn,
    cfg: FeaturesConfig,
    seed: int = 0,
) -> tuple[CorpusStats, dict[str, torch.Tensor]]:
    """ONE streaming pass over the corpus producing both the corpus stats
    (means + vocab-association score) and the per-event window activations for
    the answer-predictive score — encoding a chain's activations is the
    expensive part ([n_tokens, d_sae] TL forward + SAE encode), so everything
    that needs them shares the same pass. Non-marker comparison tokens are
    matched on within-trace position bins so position effects don't leak into
    the vocab score."""
    rng = random.Random(seed)
    window_acts: dict[str, torch.Tensor] = {}
    d_sae: Optional[int] = None
    all_acc = marker_acc = nonmarker_acc = None
    active_count: Optional[torch.Tensor] = None
    n_tokens = 0

    for chain in chains[: cfg.corpus_max_chains]:
        if chain.trace_token_span is None:
            continue
        rel = _trace_rel_range(chain)
        if len(rel) == 0:
            continue
        f_completion = encode_fn(chain.token_ids)[chain.prompt_token_len:]
        window_acts.update(
            _event_windows(f_completion, chain, events_by_chain.get(chain.chain_id, []), cfg)
        )
        f_trace = f_completion[rel.start: rel.stop]
        if d_sae is None:
            d_sae = f_trace.shape[-1]
            all_acc = _GroupAcc.zeros(d_sae)
            marker_acc = _GroupAcc.zeros(d_sae)
            nonmarker_acc = _GroupAcc.zeros(d_sae)
            active_count = torch.zeros(d_sae, dtype=torch.float64)

        all_acc.add(f_trace)
        active_count += (f_trace > 0).double().sum(0)
        n_tokens += f_trace.shape[0]

        marker_rel = {
            p for p in _marker_rel_positions(chain, events_by_chain.get(chain.chain_id, []))
            if rel.start <= p < rel.stop
        }
        if not marker_rel:
            continue
        # position bins over the trace for matched sampling
        span = max(len(rel), 1)
        bins: dict[int, list[int]] = {}
        for p in rel:
            if p in marker_rel:
                continue
            b = min(cfg.position_match_bins - 1, (p - rel.start) * cfg.position_match_bins // span)
            bins.setdefault(b, []).append(p)

        matched: list[int] = []
        for p in marker_rel:
            b = min(cfg.position_match_bins - 1, (p - rel.start) * cfg.position_match_bins // span)
            pool = bins.get(b) or [q for qs in bins.values() for q in qs]
            if pool:
                matched.extend(rng.choice(pool) for _ in range(cfg.nonmarker_per_marker))

        m_idx = torch.tensor(sorted(marker_rel), dtype=torch.long) - rel.start
        marker_acc.add(f_trace[m_idx])
        if matched:
            nm_idx = torch.tensor(matched, dtype=torch.long) - rel.start
            nonmarker_acc.add(f_trace[nm_idx])

    if d_sae is None:
        raise ValueError("no usable chains for corpus stats")
    if marker_acc.n == 0 or nonmarker_acc.n == 0:
        # an all-zero vocab_score would make select_candidates return features
        # 0..k-1 by topk tie-break — a silent garbage candidate set. Fail loud.
        raise ValueError(
            f"vocab score undefined: {marker_acc.n} marker / {nonmarker_acc.n} "
            "non-marker tokens — check the lexicon, event detection, and trace spans"
        )

    # clamp_min(0): E[x^2]-mean^2 can round tiny-negative for near-constant
    # features, and sqrt of that is NaN — which topk would rank FIRST
    pooled_var = ((marker_acc.var() + nonmarker_acc.var()) / 2).clamp_min(0)
    vocab_score = (marker_acc.mean() - nonmarker_acc.mean()) / (pooled_var.sqrt() + 1e-6)
    stats = CorpusStats(
        corpus_mean=all_acc.mean().float(),
        activation_freq=(active_count / max(n_tokens, 1)).float(),
        vocab_score=vocab_score.float(),
        marker_mean=marker_acc.mean().float(),
        nonmarker_mean=nonmarker_acc.mean().float(),
        n_marker_tokens=marker_acc.n,
        n_nonmarker_tokens=nonmarker_acc.n,
        n_chains=min(len(chains), cfg.corpus_max_chains),
    )
    return stats, window_acts


# -------------------------------------------------------------- answer-predictive


def _event_windows(
    f_completion: torch.Tensor,
    chain: ChainRecord,
    events: list[EventRecord],
    cfg: FeaturesConfig,
) -> dict[str, torch.Tensor]:
    """Max feature activation over [t0 - prewindow, t1] per event: the
    'at/just-before the event' signal (§5)."""
    out = {}
    for ev in events:
        lo = max(0, ev.onset - cfg.prewindow_tokens - chain.prompt_token_len)
        hi = min(len(f_completion) - 1, ev.end - chain.prompt_token_len)
        if hi < lo:
            continue
        out[ev.event_id] = f_completion[lo: hi + 1].max(dim=0).values
    return out


def event_window_activations(
    chain: ChainRecord,
    events: list[EventRecord],
    encode_fn: EncodeFn,
    cfg: FeaturesConfig,
) -> dict[str, torch.Tensor]:
    """Standalone wrapper around _event_windows (one encode per call); the
    pipeline gets windows from compute_corpus_stats_and_windows instead, which
    shares a single encode pass per chain."""
    if not events:
        return {}
    f_completion = encode_fn(chain.token_ids)[chain.prompt_token_len:]
    return _event_windows(f_completion, chain, events, cfg)


def answer_predictive_score(
    window_acts: dict[str, torch.Tensor],   # event_id -> [d_sae]
    event_flips: dict[str, bool],           # event_id -> did intended answer change
) -> torch.Tensor:
    """Per-feature AUROC (Mann-Whitney) of window activation predicting an
    intended-answer flip across the event. NaN when a class is empty."""
    ids = [e for e in window_acts if e in event_flips]
    if not ids:
        raise ValueError("no events with both activations and flip labels")
    X = torch.stack([window_acts[e] for e in ids]).double()   # [n_events, d_sae]
    y = torch.tensor([bool(event_flips[e]) for e in ids])
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return torch.full((X.shape[1],), float("nan"))
    ranks = _midranks(X)  # exact midranks so ties (sparse zeros!) don't bias U
    r_pos = ranks[y].sum(0)
    auc = (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc.float()


def _midranks(X: torch.Tensor) -> torch.Tensor:
    """Column-wise midranks (1-based, ties averaged), fully vectorized —
    equivalent to scipy.stats.rankdata(..., method='average') per column."""
    n, d = X.shape
    sorted_x, order = X.sort(dim=0)
    idx = torch.arange(n, dtype=torch.float64).unsqueeze(1).expand(n, d)
    new_group = torch.ones(n, d, dtype=torch.bool)
    new_group[1:] = sorted_x[1:] != sorted_x[:-1]
    # index of each element's tie-group start, in sorted order
    group_start = torch.where(new_group, idx, torch.zeros_like(idx)).cummax(dim=0).values
    # group end = (index of the next group's start) - 1; a row is "next group
    # start" for everything at or before it, so reverse-cummin propagates it back
    is_last_in_group = torch.cat([new_group[1:], torch.ones(1, d, dtype=torch.bool)])
    next_start = torch.where(is_last_in_group, idx + 1, torch.full_like(idx, float(n)))
    group_end = next_start.flip(0).cummin(dim=0).values.flip(0) - 1
    midrank_sorted = (group_start + group_end) / 2 + 1  # 1-based average rank
    ranks = torch.empty(n, d, dtype=torch.float64)
    ranks.scatter_(0, order, midrank_sorted)
    return ranks


# ----------------------------------------------------------------------- selection


@dataclass
class CandidateFeature:
    feature_id: int
    vocab_score: float
    answer_predictive_auroc: Optional[float]
    corpus_mean: float
    activation_freq: float
    dashboard_path: Optional[str] = None
    top_examples: list[dict] = field(default_factory=list)


def select_candidates(
    stats: CorpusStats,
    auroc: Optional[torch.Tensor],
    cfg: FeaturesConfig,
) -> list[CandidateFeature]:
    top = torch.topk(stats.vocab_score, cfg.n_candidate_features)
    out = []
    for score, fid in zip(top.values.tolist(), top.indices.tolist()):
        a = None
        if auroc is not None and not torch.isnan(auroc[fid]):
            a = float(auroc[fid])
        out.append(
            CandidateFeature(
                feature_id=int(fid),
                vocab_score=float(score),
                answer_predictive_auroc=a,
                corpus_mean=float(stats.corpus_mean[fid]),
                activation_freq=float(stats.activation_freq[fid]),
            )
        )
    return out


def select_control_feature(
    stats: CorpusStats,
    candidates: list[CandidateFeature],
    strategy: str,
    seed: int = 0,
    exclude_top_vocab: int = 100,
) -> int:
    """Generic-disruption control (§7.3): a feature that is NOT
    correction-associated, matched to the primary candidate on corpus stats
    (activation frequency + mean) so ablating it is comparably disruptive."""
    d_sae = stats.vocab_score.shape[0]
    excluded = set(torch.topk(stats.vocab_score, min(exclude_top_vocab, d_sae)).indices.tolist())
    excluded.update(c.feature_id for c in candidates)

    if strategy == "random":
        rng = random.Random(seed)
        alive = [i for i in range(d_sae) if i not in excluded and stats.activation_freq[i] > 0]
        return rng.choice(alive)

    primary = candidates[0]
    target = torch.tensor(
        [
            torch.log1p(torch.tensor(primary.activation_freq * 1e4)),
            torch.log1p(torch.tensor(primary.corpus_mean)),
        ]
    )
    feats = torch.stack(
        [torch.log1p(stats.activation_freq * 1e4), torch.log1p(stats.corpus_mean)], dim=1
    )
    dist = (feats - target).pow(2).sum(1)
    dist[list(excluded)] = float("inf")
    dist[stats.activation_freq <= 0] = float("inf")
    return int(dist.argmin())


# ---------------------------------------------------------------------- dashboards


def collect_top_examples(
    chains: list[ChainRecord],
    encode_fn: EncodeFn,
    feature_ids: list[int],
    cfg: FeaturesConfig,
) -> dict[int, list[dict]]:
    """Fallback dashboard data: top activating trace tokens with context.

    Restricted to the trace span: examples humans use to vet a candidate
    correction feature must be trace-internal, not post-</think> answer tokens
    (a feature firing on '\\boxed{' would otherwise dominate the dashboard)."""
    heaps: dict[int, list[dict]] = {fid: [] for fid in feature_ids}
    fid_tensor = torch.tensor(feature_ids, dtype=torch.long)
    for chain in chains[: cfg.corpus_max_chains]:
        if chain.trace_token_span is None:
            continue
        rel_start = chain.trace_token_span[0] - chain.prompt_token_len
        rel_end = chain.trace_token_span[1] - chain.prompt_token_len
        f_completion = encode_fn(chain.token_ids)[chain.prompt_token_len:, fid_tensor]
        f = f_completion[rel_start:rel_end]  # trace tokens only, [n_trace, k]
        if f.shape[0] == 0:
            continue
        for k, fid in enumerate(feature_ids):
            col = f[:, k]
            topv, topi = torch.topk(col, min(3, col.shape[0]))
            for v, i in zip(topv.tolist(), topi.tolist()):
                if v <= 0:
                    continue
                # i is trace-relative; token_char_offsets is completion-relative
                lo_c, hi_c = chain.token_char_offsets[rel_start + i]
                ctx_lo, ctx_hi = max(0, lo_c - 80), min(len(chain.text), hi_c + 80)
                heaps[fid].append(
                    {
                        "chain_id": chain.chain_id,
                        "activation": v,
                        "token_text": chain.text[lo_c:hi_c],
                        "context": chain.text[ctx_lo:ctx_hi],
                    }
                )
    for fid in feature_ids:
        heaps[fid] = sorted(heaps[fid], key=lambda d: -d["activation"])[
            : cfg.top_examples_per_feature
        ]
    return heaps


def write_dashboards(
    out_dir: str | Path,
    candidates: list[CandidateFeature],
    control_feature_id: int,
    stats: CorpusStats,
) -> None:
    """Deliverable: candidate_features.json (+ per-feature example files)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in candidates:
        p = out_dir / f"feature_{c.feature_id}.json"
        write_json(p, {"feature_id": c.feature_id, "top_examples": c.top_examples})
        c.dashboard_path = str(p)
    write_json(
        out_dir / "candidate_features.json",
        {
            "candidates": [c.__dict__ for c in candidates],
            "control_feature_id": control_feature_id,
            "n_marker_tokens": stats.n_marker_tokens,
            "n_nonmarker_tokens": stats.n_nonmarker_tokens,
            "n_chains": stats.n_chains,
        },
    )
