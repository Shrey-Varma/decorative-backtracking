"""Correction-event detection + span extraction (build order step 5, §3, §9).

Events are seeded from a marker lexicon (lexical detection) inside the
reasoning trace only. Because the lexicon is vocabulary-keyed, §11 requires a
human-validated subset: `sample_validation_events` emits events for labeling,
`detection_quality` scores the labels. Feature-based detection is provided as
the cross-check detector.
"""

from __future__ import annotations

import random
import re
from typing import Optional, Sequence

from .config import EventsConfig
from .generation import char_to_completion_token
from .io_utils import ChainRecord, EventRecord

_SENTENCE_END = re.compile(r"[.!?\n]")

# context window around an onset for human validation — one definition so the
# lexical and feature-based detectors produce comparably-sized snippets
_CTX_BEFORE, _CTX_AFTER = 120, 160


def _context_window(chain: ChainRecord, onset_char: int) -> str:
    lo = max(0, onset_char - _CTX_BEFORE)
    hi = min(len(chain.text), onset_char + _CTX_AFTER)
    return chain.text[lo:hi]


def _marker_pattern(lexicon: list[str]) -> re.Pattern:
    # Longest-first alternation so "no, wait" wins over "wait"; word-ish
    # boundaries so "wait" doesn't fire inside "waiting".
    parts = sorted((re.escape(m) for m in lexicon), key=len, reverse=True)
    return re.compile(r"(?<![a-z])(" + "|".join(parts) + r")(?![a-z])", re.IGNORECASE)


def detect_events(chain: ChainRecord, cfg: EventsConfig) -> list[EventRecord]:
    """Lexical detection. Onset t0 = first token of the marker; span extends to
    the end of the sentence containing the marker, capped at max_span_tokens.
    All indices absolute into chain.token_ids."""
    if chain.trace_token_span is None:
        return []
    trace_start, trace_end = chain.trace_token_span
    # char window of the trace within the completion text
    rel_start = trace_start - chain.prompt_token_len
    rel_end = trace_end - chain.prompt_token_len
    if rel_end <= rel_start:
        return []
    char_start = chain.token_char_offsets[rel_start][0]
    char_end = (
        chain.token_char_offsets[rel_end - 1][1]
        if rel_end - 1 < len(chain.token_char_offsets)
        else len(chain.text)
    )
    trace_text = chain.text[char_start:char_end]

    pattern = _marker_pattern(cfg.lexicon)
    raw: list[EventRecord] = []
    for m in pattern.finditer(trace_text):
        onset_char = char_start + m.start()
        onset_tok = char_to_completion_token(chain.token_char_offsets, onset_char)
        # span end: sentence boundary after the marker, capped
        sent = _SENTENCE_END.search(chain.text, char_start + m.end(), char_end)
        end_char = sent.end() if sent else char_end
        end_tok = char_to_completion_token(chain.token_char_offsets, max(end_char - 1, onset_char))
        end_tok = min(end_tok, onset_tok + cfg.max_span_tokens, rel_end - 1)
        end_tok = max(end_tok, onset_tok)

        raw.append(
            EventRecord(
                chain_id=chain.chain_id,
                event_id="",  # assigned after merging
                onset=chain.prompt_token_len + onset_tok,
                end=chain.prompt_token_len + end_tok,
                marker=m.group(1).lower(),
                marker_text=m.group(1),
                context=_context_window(chain, onset_char),
                source="lexical",
            )
        )

    merged = _merge_events(raw, cfg)
    for i, ev in enumerate(merged):
        ev.event_id = f"{chain.chain_id}__e{i}"
    return merged


def _merge_events(events: list[EventRecord], cfg: EventsConfig) -> list[EventRecord]:
    """Merge overlapping/adjacent events; drop onsets too close to the
    previous kept event (a 'wait... actually' pile-up is one event)."""
    events = sorted(events, key=lambda e: e.onset)
    out: list[EventRecord] = []
    for ev in events:
        if out:
            prev = out[-1]
            if ev.onset <= prev.end + cfg.merge_gap_tokens:
                prev.end = max(prev.end, ev.end)
                continue
            if ev.onset - prev.onset < cfg.min_onset_gap_tokens:
                continue
        out.append(ev)
    return out


# ------------------------------------------------------------ feature-based events


def detect_events_from_feature(
    chain: ChainRecord,
    feature_activations: Sequence[float],  # per completion token, one feature
    threshold: float,
    cfg: EventsConfig,
) -> list[EventRecord]:
    """Secondary detector (§11 circularity check): contiguous runs where the
    candidate feature exceeds `threshold` inside the trace."""
    if chain.trace_token_span is None:
        return []
    trace_start, trace_end = chain.trace_token_span
    rel = range(
        trace_start - chain.prompt_token_len, trace_end - chain.prompt_token_len
    )
    events: list[EventRecord] = []
    run_start: Optional[int] = None
    for i in rel:
        active = i < len(feature_activations) and feature_activations[i] > threshold
        if active and run_start is None:
            run_start = i
        elif not active and run_start is not None:
            events.append(_feature_event(chain, run_start, i - 1, cfg))
            run_start = None
    if run_start is not None:
        events.append(_feature_event(chain, run_start, rel[-1], cfg))
    merged = _merge_events(events, cfg)
    for i, ev in enumerate(merged):
        ev.event_id = f"{chain.chain_id}__f{i}"
    return merged


def _feature_event(chain: ChainRecord, rel_start: int, rel_end: int, cfg: EventsConfig) -> EventRecord:
    rel_end = min(rel_end, rel_start + cfg.max_span_tokens)
    onset_char = chain.token_char_offsets[rel_start][0]
    return EventRecord(
        chain_id=chain.chain_id,
        event_id="",
        onset=chain.prompt_token_len + rel_start,
        end=chain.prompt_token_len + rel_end,
        marker="<feature>",
        marker_text=chain.text[onset_char: onset_char + 24],
        context=_context_window(chain, onset_char),
        source="feature",
    )


# ---------------------------------------------------------------- human validation


def sample_validation_events(
    events: list[EventRecord], n: int, seed: int = 0
) -> list[dict]:
    """Sample events for human labeling. Label file format: same records with
    a filled `is_correction` field (true/false)."""
    rng = random.Random(seed)
    sample = rng.sample(events, min(n, len(events)))
    return [
        {
            "event_id": ev.event_id,
            "chain_id": ev.chain_id,
            "marker": ev.marker,
            "context": ev.context,
            "is_correction": None,  # human fills this in
        }
        for ev in sample
    ]


def detection_quality(labeled: list[dict]) -> dict:
    """Precision over the labeled sample (recall needs labeled negatives from
    a marker-free scan, reported separately when available)."""
    done = [d for d in labeled if d.get("is_correction") is not None]
    tp = sum(1 for d in done if d["is_correction"])
    return {
        "n_labeled": len(done),
        "precision": tp / len(done) if done else None,
    }
