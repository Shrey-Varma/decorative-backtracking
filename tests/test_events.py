from src.config import EventsConfig
from src.events import (
    detect_events,
    detect_events_from_feature,
    detection_quality,
    sample_validation_events,
)
from tests.conftest import DEFAULT_LEXICON, make_chain

TRACE = (
    "<think> First I compute 6 * 7 = 43. Then the total is 43. "
    "Wait, that's wrong. 6 * 7 is 42, let me recheck the sum. "
    "So the answer is 42. </think> The answer is \\boxed{42}."
)


def test_detects_events_inside_trace(tokenizer, events_cfg):
    chain = make_chain(tokenizer, TRACE)
    events = detect_events(chain, events_cfg)
    assert len(events) >= 1
    ev = events[0]
    # onset is absolute and points at "Wait,"
    rel = ev.onset - chain.prompt_token_len
    lo, hi = chain.token_char_offsets[rel]
    assert chain.text[lo:hi].strip().lower().startswith("wait")
    assert ev.end >= ev.onset
    assert ev.marker == "wait"


def test_adjacent_markers_merge_into_one_event(tokenizer, events_cfg):
    chain = make_chain(
        tokenizer,
        "<think> total 43. Wait, actually that's wrong, let me recheck it now. "
        "It is 42. </think> \\boxed{42}",
    )
    events = detect_events(chain, events_cfg)
    assert len(events) == 1  # wait/actually/that's wrong pile-up = one event


def test_no_events_outside_trace(tokenizer, events_cfg):
    chain = make_chain(
        tokenizer,
        "<think> compute 6*7=42 directly. </think> Wait, the answer is \\boxed{42}.",
    )
    assert detect_events(chain, events_cfg) == []


def test_word_boundary_no_false_positive(tokenizer, events_cfg):
    chain = make_chain(
        tokenizer,
        "<think> we are waiting for the awaited result 42 </think> \\boxed{42}",
    )
    assert detect_events(chain, events_cfg) == []


def test_span_cap(tokenizer):
    cfg = EventsConfig(lexicon=DEFAULT_LEXICON, max_span_tokens=3)
    long_tail = "wait " + "word " * 50 + ". done"
    chain = make_chain(tokenizer, f"<think> x is 1. {long_tail} </think> \\boxed{{1}}")
    events = detect_events(chain, cfg)
    assert len(events) == 1
    assert events[0].end - events[0].onset <= 3


def test_distant_markers_stay_separate(tokenizer, events_cfg):
    filler = "step " * 40
    chain = make_chain(
        tokenizer,
        f"<think> a is 3. wait, recheck. {filler} actually it is 4. {filler} "
        f"</think> \\boxed{{4}}",
    )
    events = detect_events(chain, events_cfg)
    assert len(events) == 2
    assert events[0].event_id != events[1].event_id


def test_feature_based_detection(tokenizer, events_cfg):
    chain = make_chain(tokenizer, TRACE)
    n_completion = len(chain.token_ids) - chain.prompt_token_len
    acts = [0.0] * n_completion
    s, e = chain.trace_token_span
    mid = (s + e) // 2 - chain.prompt_token_len
    for i in range(mid, mid + 3):
        acts[i] = 5.0
    events = detect_events_from_feature(chain, acts, threshold=1.0, cfg=events_cfg)
    assert len(events) == 1
    assert events[0].source == "feature"
    assert events[0].onset == chain.prompt_token_len + mid


def test_validation_sampling_and_quality():
    from src.io_utils import EventRecord

    events = [
        EventRecord("c", f"c__e{i}", 10, 12, "wait", "Wait", "ctx") for i in range(20)
    ]
    sample = sample_validation_events(events, 5, seed=1)
    assert len(sample) == 5
    assert all(s["is_correction"] is None for s in sample)
    for i, s in enumerate(sample):
        s["is_correction"] = i < 4
    q = detection_quality(sample)
    assert q["n_labeled"] == 5
    assert abs(q["precision"] - 0.8) < 1e-9
