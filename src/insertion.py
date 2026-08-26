"""Text-level insertion test + induced-verbalization helpers (sufficiency).

The excision test (src/excision.py) closed NECESSITY at the text level:
removing the verbalized correction from context costs ~nothing. This module
closes the mirror-image SUFFICIENCY question: at a point where the model's
intended answer is WRONG and it is not verbalizing reconsideration, does
splicing in a synthetic correction make it recover?

Two populations:

  wrong_point    for each informative event, choose_control_position picks a
                 truncation in the SAME chain where the intended answer is
                 (still) wrong and every event span is >= min_distance away —
                 the model is on a wrong track and not verbalizing. Chains
                 here DID self-correct elsewhere, so the base rescue rate is
                 high; this population ties directly to the necessity grid.
  failed_chain   chains whose FINAL answer is wrong — the model never rescued
                 itself. Insertion mid-trace tests whether prompting the
                 verbalization can save a chain that verbalization-as-emitted
                 did not. No ceiling: base rescue is low by construction.

Arms (shared backend + seeds; correction comes in three variants so the
result can't hinge on one phrasing, and so 'says a mistake exists' is
dissociated from the bare reconsideration form):

  insert_natural                    continue from tokens[:t]  (base rescue rate)
  insert_correction/v=<variant>     tokens[:t] + synthetic correction
  insert_filler                     tokens[:t] + length-matched neutral text

Insertion points are snapped to sentence starts — that is where real
correction onsets live, and it avoids a mid-expression splice that would
disrupt only the spliced arms.

Sufficient verbalization ⇒ acc(insert_correction) >> acc(insert_natural) and
>> acc(insert_filler).
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .answers import answers_equal, extract_explicit_answer
from .events import _marker_pattern
from .excision import DEFAULT_FILLER, tile_tokens
from .experiment import choose_control_position
from .generation import char_to_completion_token
from .io_utils import ChainRecord, EventRecord
from .trajectory import Trajectory

# Three synthetic corrections in R1-distill's own register. All CONTAIN
# lexicon markers (they must read as verbalized self-correction); they differ
# in how much they assert beyond the form itself:
#   assert_mistake  claims an error exists (form + information)
#   soft_recheck    prompts re-examination without asserting an error
#   minimal_marker  the bare backtracking interjection
CORRECTION_VARIANTS = {
    "assert_mistake": (
        " Wait, I think I made a mistake somewhere — that doesn't seem right. "
        "Let me re-examine the last steps carefully."
    ),
    "soft_recheck": (
        " Wait, let me double-check that last step before moving on."
    ),
    "minimal_marker": " Hmm, wait.",
}

INSERTION_REF = "insert_natural"
INSERTION_ARMS = (
    INSERTION_REF,
    *(f"insert_correction/v={v}" for v in CORRECTION_VARIANTS),
    "insert_filler",
)

_SENTENCE_END = re.compile(r"[.!?\n]")


def insertion_prefixes(
    chain: ChainRecord,
    t_insert: int,
    tokenizer,
    variants: dict[str, str] = CORRECTION_VARIANTS,
    filler_text: str = DEFAULT_FILLER,
) -> dict[str, list[int]]:
    """Token prefixes for every insertion arm. The filler is tiled/truncated
    to the LONGEST variant's length, so 'any extra tokens of thinking time'
    is controlled at least as generously as any correction arm."""
    base = list(chain.token_ids[:t_insert])
    enc = {v: tokenizer.encode(text, add_special_tokens=False)
           for v, text in variants.items()}
    max_len = max(len(ids) for ids in enc.values())
    filler = tile_tokens(tokenizer.encode(filler_text, add_special_tokens=False), max_len)
    out = {INSERTION_REF: base, "insert_filler": base + filler}
    for v, ids in enc.items():
        out[f"insert_correction/v={v}"] = base + ids
    return out


def snap_to_sentence_start(chain: ChainRecord, t: int) -> int:
    """Nearest sentence-start token at or before absolute position `t`
    (first token after the previous [.!?\\n]), clipped inside the trace.
    Real correction onsets live at sentence starts; splicing there avoids a
    mid-expression discontinuity that only the spliced arms would suffer.
    Falls back to `t` when no boundary is found nearby."""
    start, end = chain.trace_token_span
    rel = t - chain.prompt_token_len
    if rel <= 0 or rel > len(chain.token_char_offsets):
        return t
    char = chain.token_char_offsets[min(rel, len(chain.token_char_offsets) - 1)][0]
    window_start = max(0, char - 400)
    m = None
    for m in _SENTENCE_END.finditer(chain.text, window_start, char):
        pass  # keep the LAST boundary before the insertion char
    if m is None:
        return t
    tok = char_to_completion_token(chain.token_char_offsets, m.end())
    if chain.token_char_offsets[tok][0] < m.end():
        tok += 1
    snapped = chain.prompt_token_len + tok
    if not (start < snapped <= min(t, end)):
        return t
    return snapped


def _far_from_events(t: int, events: Sequence[EventRecord], min_distance: int) -> bool:
    return all(t < ev.onset - min_distance or t > ev.end + min_distance for ev in events)


def select_insertion_point(
    chain: ChainRecord,
    all_events: Sequence[EventRecord],
    event: EventRecord,
    min_distance: int,
    trajectory: Optional[Trajectory],
) -> Optional[tuple[int, Optional[str]]]:
    """wrong_point population: (t_insert, intended answer at the ORIGINAL
    probe point) for one informative event — the matched 'wrong intended
    answer, far from every event' truncation used by the position-specificity
    control, snapped to a sentence start (snap kept only if it stays clear of
    events). The pre-answer is what REVERT is scored against; None when the
    trajectory has no parse there (callers fall back to the transition's
    pre_answer)."""
    t_probe = choose_control_position(chain, list(all_events), event, min_distance, trajectory)
    if t_probe is None:
        return None
    t = t_probe
    snapped = snap_to_sentence_start(chain, t_probe)
    if snapped != t_probe and _far_from_events(snapped, all_events, min_distance):
        t = snapped
    # the intended answer was probed at t_probe; the snap moves the cut by at
    # most one sentence, so the probed answer remains the right REVERT anchor
    pre = None
    if trajectory is not None:
        pre = next(
            (p.answer for p in trajectory.points
             if p.position == t_probe and p.answer is not None),
            None,
        )
    return t, pre


def select_failed_chain_point(
    chain: ChainRecord,
    events: Sequence[EventRecord],
    min_distance: int,
    frac: float = 0.5,
) -> Optional[tuple[int, Optional[str]]]:
    """failed_chain population: a mid-trace insertion point (default 50% of
    the trace, snapped to a sentence start) at least min_distance from every
    event span, scanning outward from the midpoint when the midpoint itself
    is too close to an event. Returns (t_insert, the chain's wrong final
    answer) — REVERT then means 'still lands on the same wrong answer'."""
    if chain.trace_token_span is None:
        return None
    start, end = chain.trace_token_span
    if end - start < 4 * min_distance:
        return None
    mid = start + int((end - start) * frac)
    t = None
    for off in range(0, end - start, 8):  # scan outward from the midpoint
        for cand in (mid - off, mid + off):
            if start + min_distance < cand < end - min_distance and _far_from_events(
                cand, events, min_distance
            ):
                t = cand
                break
        if t is not None:
            break
    if t is None:
        return None
    snapped = snap_to_sentence_start(chain, t)
    if snapped != t and _far_from_events(snapped, events, min_distance):
        t = snapped
    return t, extract_explicit_answer(chain.text)


def failed_chains(chains: Sequence[ChainRecord]) -> list[ChainRecord]:
    """Chains whose final explicit answer parses and is WRONG — the model
    never rescued itself. (Unparseable finals are excluded: ambiguous.)"""
    out = []
    for c in chains:
        final = extract_explicit_answer(c.text)
        if final is not None and not answers_equal(final, c.gold_answer):
            out.append(c)
    return out


# ------------------------------------------------- induced-verbalization metric


def first_marker_offset(text: str, lexicon: list[str]) -> Optional[int]:
    """Char offset of the first correction marker in `text`, or None."""
    m = _marker_pattern(list(lexicon)).search(text)
    return m.start() if m else None


def verbalizes_marker(text: str, lexicon: list[str], within_chars: int = 240) -> bool:
    """Does the continuation verbalize a correction near its start? Used to
    measure whether clamping the feature/direction ON *induces* the
    verbalization (mechanistic sufficiency), vs the natural base rate."""
    off = first_marker_offset(text, lexicon)
    return off is not None and off < within_chars
