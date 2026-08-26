"""Text-level excision test (the causality closer).

The feature/direction ablations are causal nulls *about their carriers*; the
inference to "the verbalized correction is decorative" still leans on having
localized the right carrier. This experiment removes that assumption entirely
by intervening in TEXT space:

  A  text_with_correction    continue from tokens[:t1+1]  (correction in context)
  B  text_without_correction continue from tokens[:t0]    (correction never happened)
  C  text_neutral_filler     tokens[:t0] + same-length neutral filler
                             (controls prefix length/position vs A)

If having verbalized the correction causally helps downstream, A beats B
(Δ = acc(B) − acc(A) strongly negative). If A ≈ B ≈ C, the verbalization
itself confers no downstream benefit — decorative at the text level, with no
localization assumptions. All arms run on ONE backend with shared seeds so
the contrast is never cross-backend.
"""

from __future__ import annotations

from .io_utils import ChainRecord, EventRecord

# topic-neutral, non-corrective connective; tiled/truncated to the exact span
# length. Deliberately contains no lexicon marker ("wait", "hmm", ...).
DEFAULT_FILLER = (
    " I keep working through the problem carefully, continuing with the same "
    "approach and the same steps as before."
)

TEXT_ARMS = ("text_with_correction", "text_without_correction", "text_neutral_filler")


def tile_tokens(ids: list[int], span: int) -> list[int]:
    """Tile/truncate a token sequence to exactly `span` tokens (length-matched
    filler construction, shared with the insertion test)."""
    return (ids * (span // len(ids) + 1))[:span]


def excision_prefixes(
    chain: ChainRecord, event: EventRecord, tokenizer, filler_text: str = DEFAULT_FILLER
) -> dict[str, list[int]]:
    """Token prefixes for the three arms. A and C have identical length."""
    t0, t1 = event.onset, event.end
    span = t1 + 1 - t0
    filler = tile_tokens(tokenizer.encode(filler_text, add_special_tokens=False), span)
    return {
        "text_with_correction": list(chain.token_ids[: t1 + 1]),
        "text_without_correction": list(chain.token_ids[:t0]),
        "text_neutral_filler": list(chain.token_ids[:t0]) + filler,
    }
