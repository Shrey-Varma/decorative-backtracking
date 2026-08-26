"""Do verbalized corrections coincide with internal state changes? (#4)

The causal arms test whether the correction signal is load-bearing; this
module tests the weaker correlational claim directly: is a correction marker
even accompanied by a change in the model's internal state, or is it surface
text all the way down?

Metric — representational shift across a span at layer L:

    shift = 1 - cos( mean resid[t0-w : t0] ,  mean resid[t1+1 : t1+1+w] )

The windows straddle the span but EXCLUDE it, so the lexical identity of the
marker tokens ("wait", ...) cannot drive the metric — it measures whether the
ONGOING reasoning state after the correction differs from the state before
it. Controls are pseudo-spans of the same length at position-decile-matched
locations whose windows are clear of every event span, so 'state drifts with
depth' is subtracted out. Everything is paired per event and inference is a
chain-level sign-flip permutation.

Interpretation either way closes reviewer checklist #4:
  shift(events) >  shift(controls): markers are internally real (they mark a
      representational transition) — but per the causal nulls, not one that
      is answer-relevant.
  shift(events) ~= shift(controls): the verbalization isn't even accompanied
      by a distinctive internal state change.
"""

from __future__ import annotations

import random
import re
from typing import Optional, Sequence

import numpy as np
import torch

from .generation import char_to_completion_token
from .io_utils import ChainRecord, EventRecord

_SENTENCE_END = re.compile(r"[.!?\n]")


def span_shift(resid: torch.Tensor, t0: int, t1: int, w: int) -> Optional[float]:
    """1 - cosine between the mean residual over [t0-w, t0) and over
    (t1, t1+w]. `resid` is [seq, d] for one layer; indices are absolute rows
    of that tensor. None when a window would run off either end."""
    if t0 - w < 0 or t1 + 1 + w > resid.shape[0]:
        return None
    a = resid[t0 - w: t0].float().mean(0)
    b = resid[t1 + 1: t1 + 1 + w].float().mean(0)
    denom = float(a.norm() * b.norm())
    if denom == 0.0:
        return None
    return float(1.0 - float(a @ b) / denom)


def sentence_start_positions(chain: ChainRecord) -> list[int]:
    """Absolute token positions that begin a sentence inside the trace.
    Correction markers live at sentence starts, so controls drawn from the
    SAME kind of location subtract out any generic
    'state changes at discourse boundaries' effect."""
    if chain.trace_token_span is None:
        return []
    start, end = chain.trace_token_span
    out = []
    for m in _SENTENCE_END.finditer(chain.text):
        tok = char_to_completion_token(chain.token_char_offsets, m.end())
        if chain.token_char_offsets[tok][0] < m.end():
            tok += 1
        p = chain.prompt_token_len + tok
        if start < p < end:
            out.append(p)
    return sorted(set(out))


def span_end_at_sentence(chain: ChainRecord, onset: int, max_span_tokens: int) -> int:
    """End a pseudo-span the way `events.detect_events` ends a real one: at the
    sentence boundary after the onset, capped.

    Needed because the two window ends are not otherwise matched. A real event
    span stops at a sentence end, so its AFTER-window starts at a fresh
    sentence; a length-matched pseudo-span usually stops mid-sentence, so its
    after-window does not. The sentence-start control therefore subtracts
    "state changes at a discourse boundary" on the before side only, and the
    residual asymmetry inflates the event-minus-control shift by an unknown
    amount. Pass control_span_end="sentence_end" to remove it (this changes the
    measured shift, so it requires re-running stage 13).
    """
    start, end = chain.trace_token_span
    rel = onset - chain.prompt_token_len
    if rel < 0 or rel >= len(chain.token_char_offsets):
        return min(onset + max_span_tokens, end - 1)
    char = chain.token_char_offsets[rel][1]
    m = _SENTENCE_END.search(chain.text, char)
    if m is None:
        return min(onset + max_span_tokens, end - 1)
    tok = char_to_completion_token(chain.token_char_offsets, max(m.end() - 1, 0))
    return max(onset, min(chain.prompt_token_len + tok, onset + max_span_tokens, end - 1))


def sample_control_span(
    chain: ChainRecord,
    events: Sequence[EventRecord],
    event: EventRecord,
    w: int,
    rng: random.Random,
    n_bins: int = 10,
    max_tries: int = 200,
    candidates: Optional[Sequence[int]] = None,
) -> Optional[int]:
    """Onset of a pseudo-span for `event`: same span length, inside the trace
    with window room, position-decile-matched to the real onset when possible,
    and with [onset-w, onset+span+w] clear of EVERY event span.

    When `candidates` is given (e.g. sentence-start positions), controls are
    drawn from it — matched decile first, then any candidate — before falling
    back to uniform rejection sampling over the trace."""
    start, end = chain.trace_token_span
    span_len = event.end - event.onset
    lo, hi = start + w, end - span_len - w - 1
    if hi <= lo:
        return None
    trace = max(end - start, 1)
    b = min(n_bins - 1, (event.onset - start) * n_bins // trace)
    bin_lo = max(lo, start + b * trace // n_bins)
    bin_hi = min(hi, start + (b + 1) * trace // n_bins)

    def clear(p: int) -> bool:
        return all(ev.end < p - w or ev.onset > p + span_len + w for ev in events)

    if candidates:
        in_range = [p for p in candidates if lo <= p < hi and clear(p)]
        in_bin = [p for p in in_range if bin_lo <= p < bin_hi]
        for pool in (in_bin, in_range):
            if pool:
                return rng.choice(pool)

    for lo_i, hi_i in ((bin_lo, bin_hi), (lo, hi)):
        if hi_i > lo_i:
            for _ in range(max_tries):
                p = rng.randrange(lo_i, hi_i)
                if clear(p):
                    return p
    return None


def event_control_pairs(
    chain: ChainRecord,
    events: Sequence[EventRecord],
    w: int,
    rng: random.Random,
    candidates: Optional[Sequence[int]] = None,
    control_span_end: str = "length_matched",
) -> list[tuple[EventRecord, int, int]]:
    """(event, control_onset, control_end) for every event with window room and
    a clear matched control.

    control_span_end="length_matched" gives the pseudo-span the same token
    length as the event (matches the window gap, leaves the after-window
    boundary type unmatched); "sentence_end" builds it with the same rule that
    built the event span (matches the boundary type, lets the length vary).
    See span_end_at_sentence.
    """
    if control_span_end not in ("length_matched", "sentence_end"):
        raise ValueError(f"unknown control_span_end: {control_span_end}")
    out = []
    for ev in events:
        c = sample_control_span(chain, events, ev, w, rng, candidates=candidates)
        if c is None:
            continue
        if control_span_end == "sentence_end":
            end = span_end_at_sentence(chain, c, max(ev.end - ev.onset, 1))
        else:
            end = c + (ev.end - ev.onset)
        out.append((ev, c, end))
    return out


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Rank-based AUROC (ties get midranks). None when one class is empty."""
    y = np.asarray(labels, dtype=bool)
    x = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    sx = x[order]
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2 + 1  # midrank, 1-indexed
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def shift_stats(
    diffs_by_chain: dict[str, list[float]],
    n_perm: int = 10000,
    seed: int = 7,
) -> Optional[dict]:
    """Chain-level inference on paired (event - control) shifts: events
    average within chain, sign-flip permutation over chains (chains are the
    exchangeable unit — events within a chain share state)."""
    per_chain = np.array([float(np.mean(v)) for v in diffs_by_chain.values() if v])
    n_pairs = sum(len(v) for v in diffs_by_chain.values())
    if len(per_chain) < 5:
        return None
    obs = float(per_chain.mean())
    sd = float(per_chain.std(ddof=1))
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(per_chain)))
    perm = np.abs((flips * per_chain).mean(axis=1))
    p = float((np.sum(perm >= abs(obs) - 1e-12) + 1) / (n_perm + 1))
    return {
        "n_chains": int(len(per_chain)),
        "n_pairs": int(n_pairs),
        "mean_event_minus_control": obs,
        # d_z: the PAIRED effect size (mean chain-level difference over the SD of
        # chain-level differences), not a between-condition d. It is the right
        # statistic for this design and systematically larger than d_av, so it
        # must be reported as d_z rather than "Cohen's d".
        "cohens_dz": obs / sd if sd > 0 else None,
        "cohens_d": obs / sd if sd > 0 else None,  # deprecated alias
        "p_two_sided": p,
    }
