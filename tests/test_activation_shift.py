import pytest
import random

import torch

from src.activation_shift import (
    auroc,
    event_control_pairs,
    sample_control_span,
    sentence_start_positions,
    shift_stats,
    span_shift,
)
from src.config import EventsConfig
from src.events import detect_events
from tests.conftest import DEFAULT_LEXICON, make_chain

W = 3


def _chain_and_events(tokenizer):
    filler = "step " * 30
    chain = make_chain(
        tokenizer,
        f"<think> the total is 7. {filler} wait, that's wrong, recheck the sum. "
        f"{filler} so 42. </think> \\boxed{{42}}",
        gold="42",
    )
    events = detect_events(chain, EventsConfig(lexicon=DEFAULT_LEXICON))
    assert len(events) == 1
    return chain, events


def test_span_shift_ignores_span_content():
    # identical state before and after; WILD content inside the span — the
    # metric must be ~0 because windows exclude the span (token identity of
    # "wait" cannot drive it)
    resid = torch.ones(50, 8)
    resid[20:25] = torch.randn(5, 8) * 9
    s = span_shift(resid, 20, 24, w=5)
    assert s is not None and abs(s) < 1e-6


def test_span_shift_detects_state_change():
    resid = torch.zeros(50, 8)
    resid[:20, 0] = 1.0   # state before: e0
    resid[25:, 1] = 1.0   # state after: e1 (orthogonal)
    s = span_shift(resid, 20, 24, w=5)
    assert s is not None and s > 0.99


def test_span_shift_no_window_room_returns_none():
    resid = torch.randn(10, 4)
    assert span_shift(resid, 2, 4, w=5) is None    # pre-window off the start
    assert span_shift(resid, 5, 8, w=4) is None    # post-window off the end


def test_control_span_is_clear_and_inside_trace(tokenizer):
    chain, events = _chain_and_events(tokenizer)
    ev = events[0]
    p = sample_control_span(chain, events, ev, w=W, rng=random.Random(0))
    assert p is not None
    span_len = ev.end - ev.onset
    start, end = chain.trace_token_span
    assert start + W <= p and p + span_len + W < end
    # the pseudo-span plus its windows never touches the real event span
    assert ev.end < p - W or ev.onset > p + span_len + W


def test_control_span_deterministic_given_rng(tokenizer):
    chain, events = _chain_and_events(tokenizer)
    ev = events[0]
    p1 = sample_control_span(chain, events, ev, w=W, rng=random.Random(7))
    p2 = sample_control_span(chain, events, ev, w=W, rng=random.Random(7))
    assert p1 == p2


def test_event_control_pairs_skips_events_without_room(tokenizer):
    chain, events = _chain_and_events(tokenizer)
    pairs = event_control_pairs(chain, events, w=W, rng=random.Random(0))
    assert len(pairs) == 1
    ev, onset, end = pairs[0]
    # default mode length-matches the pseudo-span to the event span
    assert end - onset == ev.end - ev.onset
    # a window far larger than the trace leaves no room for anything
    assert event_control_pairs(chain, events, w=10_000, rng=random.Random(0)) == []


def test_event_control_pairs_sentence_end_mode(tokenizer):
    """The alternative control construction ends the pseudo-span at a sentence
    boundary, the same rule that ends a real event span — removing the
    after-window boundary asymmetry at the cost of exact length matching."""
    chain, events = _chain_and_events(tokenizer)
    pairs = event_control_pairs(chain, events, w=W, rng=random.Random(0),
                                control_span_end="sentence_end")
    assert len(pairs) == 1
    ev, onset, end = pairs[0]
    assert end >= onset
    assert end - onset <= max(ev.end - ev.onset, 1)
    with pytest.raises(ValueError):
        event_control_pairs(chain, events, w=W, rng=random.Random(0),
                            control_span_end="nonsense")


def test_shift_stats_reports_dz(tokenizer):
    diffs = {f"c{i}": [0.30 + 0.01 * (i % 3)] for i in range(12)}
    st = shift_stats(diffs)
    # the paired effect size is d_z and must be labelled as such
    assert st["cohens_dz"] == st["cohens_d"]


def test_shift_stats_detects_consistent_positive_effect():
    diffs = {f"c{i}": [0.30 + 0.01 * (i % 3)] for i in range(12)}
    st = shift_stats(diffs)
    assert st is not None
    assert st["mean_event_minus_control"] > 0.25
    assert st["p_two_sided"] < 0.01
    assert st["n_chains"] == 12 and st["n_pairs"] == 12


def test_shift_stats_null_is_nonsignificant():
    rng = random.Random(3)
    diffs = {f"c{i}": [rng.gauss(0.0, 0.1) for _ in range(3)] for i in range(20)}
    st = shift_stats(diffs)
    assert st is not None and st["p_two_sided"] > 0.05


def test_shift_stats_requires_min_chains():
    assert shift_stats({"c1": [0.1], "c2": [0.2]}) is None


def test_sentence_start_positions_inside_trace(tokenizer):
    chain, _ = _chain_and_events(tokenizer)
    starts = sentence_start_positions(chain)
    assert starts, "chain has sentences -> must find sentence starts"
    lo, hi = chain.trace_token_span
    assert all(lo < p < hi for p in starts)
    # each position begins right after a sentence-ending char
    for p in starts[:3]:
        rel = p - chain.prompt_token_len
        prev_text = chain.text[: chain.token_char_offsets[rel][0]]
        assert prev_text.rstrip()[-1:] in ".!?\n" or prev_text.endswith("\n")


def test_control_span_prefers_candidates(tokenizer):
    chain, events = _chain_and_events(tokenizer)
    ev = events[0]
    starts = sentence_start_positions(chain)
    p = sample_control_span(chain, events, ev, w=W, rng=random.Random(0),
                            candidates=starts)
    assert p in starts  # drawn from the candidate pool, not uniform fallback


def test_auroc_extremes_and_chance():
    assert auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
    assert auroc([0.1, 0.2], [True, True]) is None  # one class empty
