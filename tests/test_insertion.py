from src.config import EventsConfig
from src.events import _marker_pattern, detect_events
from src.insertion import (
    CORRECTION_VARIANTS,
    INSERTION_ARMS,
    INSERTION_REF,
    failed_chains,
    first_marker_offset,
    insertion_prefixes,
    select_failed_chain_point,
    select_insertion_point,
    snap_to_sentence_start,
    verbalizes_marker,
)
from src.trajectory import Trajectory, TrajectoryPoint
from tests.conftest import DEFAULT_LEXICON, make_chain


def _chain_and_event(tokenizer):
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


def test_insertion_prefix_construction(tokenizer):
    chain, events = _chain_and_event(tokenizer)
    t = chain.trace_token_span[0] + 5
    p = insertion_prefixes(chain, t, tokenizer)
    assert set(p) == set(INSERTION_ARMS)
    # natural: plain truncation at the insertion point
    assert p[INSERTION_REF] == chain.token_ids[:t]
    # each correction variant: natural + its synthetic correction verbatim
    for v, text in CORRECTION_VARIANTS.items():
        corr = tokenizer.encode(text, add_special_tokens=False)
        assert p[f"insert_correction/v={v}"] == chain.token_ids[:t] + corr
    # filler is at least as long as every correction arm (generous
    # extra-tokens control) and shares the pre-insertion prefix
    max_arm = max(len(p[f"insert_correction/v={v}"]) for v in CORRECTION_VARIANTS)
    assert len(p["insert_filler"]) == max_arm
    assert p["insert_filler"][:t] == chain.token_ids[:t]


def test_every_variant_contains_marker_filler_does_not(tokenizer):
    chain, events = _chain_and_event(tokenizer)
    t = chain.trace_token_span[0] + 5
    p = insertion_prefixes(chain, t, tokenizer)
    pattern = _marker_pattern(list(DEFAULT_LEXICON))
    for v in CORRECTION_VARIANTS:
        text = tokenizer.decode(p[f"insert_correction/v={v}"][t:])
        assert pattern.search(text) is not None, (
            f"variant {v} must verbalize a marker: {text!r}")
    filler_text = tokenizer.decode(p["insert_filler"][t:])
    assert pattern.search(filler_text) is None, (
        f"filler must not contain correction markers: {filler_text!r}")


def test_snap_to_sentence_start(tokenizer):
    chain = make_chain(
        tokenizer,
        "<think> one two three. four five six seven eight nine. </think> \\boxed{42}",
    )
    t = chain.prompt_token_len + 5  # mid-sentence, at "five"
    s = snap_to_sentence_start(chain, t)
    assert s == chain.prompt_token_len + 4  # "four", right after "three."
    # already at a sentence start -> unchanged
    assert snap_to_sentence_start(chain, s) == s


def test_select_insertion_point_prefers_wrong_trajectory_points(tokenizer):
    chain, events = _chain_and_event(tokenizer)
    ev = events[0]
    t_wrong = chain.trace_token_span[0] + 6
    assert t_wrong < ev.onset - 8  # genuinely far from the event
    traj = Trajectory(
        chain_id=chain.chain_id,
        points=[TrajectoryPoint(position=t_wrong, answer="7", correct=False)],
    )
    sel = select_insertion_point(chain, events, ev, min_distance=8, trajectory=traj)
    assert sel is not None
    t, pre = sel
    # sentence snapping may move the cut backwards, never forwards, and
    # never inside the event
    assert chain.trace_token_span[0] < t <= t_wrong
    assert t < ev.onset - 8
    # the REVERT anchor is the answer probed at the ORIGINAL point
    assert pre == "7"


def test_failed_chain_selection(tokenizer):
    filler = "step " * 40
    ok = make_chain(tokenizer, f"<think> {filler} </think> \\boxed{{42}}",
                    chain_id="p1__c0", gold="42")
    bad = make_chain(tokenizer, f"<think> {filler} </think> \\boxed{{41}}",
                     chain_id="p2__c0", gold="42")
    unparsed = make_chain(tokenizer, f"<think> {filler} </think> no answer",
                          chain_id="p3__c0", gold="42")
    fails = failed_chains([ok, bad, unparsed])
    assert [c.chain_id for c in fails] == ["p2__c0"]

    sel = select_failed_chain_point(bad, [], min_distance=4)
    assert sel is not None
    t, pre = sel
    start, end = bad.trace_token_span
    assert start + 4 < t < end - 4
    assert pre == "41"  # REVERT anchor: the chain's wrong final answer


def test_marker_detection_helpers():
    assert first_marker_offset("so 6. Wait, that's wrong.", DEFAULT_LEXICON) == 6
    assert first_marker_offset("all smooth sailing here", DEFAULT_LEXICON) is None
    assert verbalizes_marker("hmm... wait, redo it", DEFAULT_LEXICON)
    # marker only far into the text does not count as "induced at the start"
    late = "x" * 500 + " wait, hold on"
    assert not verbalizes_marker(late, DEFAULT_LEXICON, within_chars=240)
    # word-boundary: "waiting" must not fire
    assert not verbalizes_marker("waiting for the result", DEFAULT_LEXICON)
