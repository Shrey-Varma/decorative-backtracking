from src.config import EventsConfig
from src.events import _marker_pattern, detect_events
from src.excision import TEXT_ARMS, excision_prefixes
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
    return chain, events[0]


def test_excision_prefix_construction(tokenizer):
    chain, ev = _chain_and_event(tokenizer)
    p = excision_prefixes(chain, ev, tokenizer)
    assert set(p) == set(TEXT_ARMS)
    # A: everything through the correction span, inclusive
    assert p["text_with_correction"] == chain.token_ids[: ev.end + 1]
    # B: correction never happened
    assert p["text_without_correction"] == chain.token_ids[: ev.onset]
    # C: same length as A (length/position control), same pre-onset content
    assert len(p["text_neutral_filler"]) == len(p["text_with_correction"])
    assert p["text_neutral_filler"][: ev.onset] == chain.token_ids[: ev.onset]
    # ...but the span content differs (the correction is gone)
    assert p["text_neutral_filler"][ev.onset:] != chain.token_ids[ev.onset: ev.end + 1]


def test_filler_contains_no_correction_markers(tokenizer):
    chain, ev = _chain_and_event(tokenizer)
    p = excision_prefixes(chain, ev, tokenizer)
    filler_ids = p["text_neutral_filler"][ev.onset:]
    filler_text = tokenizer.decode(filler_ids)
    assert _marker_pattern(EventsConfig(lexicon=DEFAULT_LEXICON).lexicon).search(
        filler_text
    ) is None, f"filler must not itself contain correction markers: {filler_text!r}"


def test_filler_tiles_for_long_spans(tokenizer):
    chain, ev = _chain_and_event(tokenizer)
    ev.end = ev.onset + 200  # force a span longer than one filler sentence
    p = excision_prefixes(chain, ev, tokenizer)
    assert len(p["text_neutral_filler"]) == ev.end + 1
