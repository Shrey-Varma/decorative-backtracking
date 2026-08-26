"""Intended-answer trajectory + flip localization (build order step 7, §6).

The trajectory is the force-answer probe evaluated on a grid of prefixes plus
every event onset/offset. It localizes where the intended answer flips
wrong -> right and whether that flip is aligned to a correction event or
drifts in smoothly between events — the position-confound diagnostic that
keeps "later = more context = more correct" from masquerading as a
correction effect (§1, §10 localization).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .answers import answers_equal
from .config import TrajectoryConfig
from .io_utils import ChainRecord, EventRecord

# batch prober: (chain, positions) -> {position: parsed_answer or None}
BatchProber = Callable[[ChainRecord, list[int]], dict[int, Optional[str]]]


@dataclass
class TrajectoryPoint:
    position: int
    answer: Optional[str]
    correct: bool


@dataclass
class FlipRecord:
    from_position: int          # last probe point with the old answer
    to_position: int            # first probe point with the new answer
    from_answer: Optional[str]
    to_answer: Optional[str]
    wrong_to_right: bool
    aligned_event_id: Optional[str]  # event onset inside the flip interval, if any


@dataclass
class Trajectory:
    chain_id: str
    points: list[TrajectoryPoint] = field(default_factory=list)
    flips: list[FlipRecord] = field(default_factory=list)


def probe_grid(chain: ChainRecord, events: list[EventRecord], stride: int) -> list[int]:
    """Grid positions: every `stride` tokens across the trace, plus every event
    onset and first-token-after-offset (§6)."""
    start, end = chain.trace_token_span
    positions = set(range(start + 1, end, stride))
    positions.add(end)  # intended answer at end of trace
    for ev in events:
        positions.add(ev.onset)
        positions.add(min(ev.end + 1, end))
    return sorted(p for p in positions if start < p <= end)


def compute_trajectory(
    chain: ChainRecord,
    events: list[EventRecord],
    batch_prober: BatchProber,
    cfg: TrajectoryConfig,
) -> Trajectory:
    positions = probe_grid(chain, events, cfg.stride)
    answers = batch_prober(chain, positions)
    points = [
        TrajectoryPoint(
            position=t,
            answer=answers.get(t),
            correct=answers_equal(answers.get(t), chain.gold_answer),
        )
        for t in positions
    ]
    traj = Trajectory(chain_id=chain.chain_id, points=points)
    traj.flips = localize_flips(points, events, cfg.align_window_tokens)
    return traj


def localize_flips(
    points: list[TrajectoryPoint],
    events: list[EventRecord],
    align_window: int,
) -> list[FlipRecord]:
    """A flip is a change of parsed answer between consecutive probe points.
    It is event-aligned when an event onset lies in (from_position - window,
    to_position] — i.e. the flip interval, padded backwards by the alignment
    window so a probe landing exactly on an onset still credits the event."""
    flips = []
    for a, b in zip(points, points[1:]):
        if answers_equal(a.answer, b.answer) or (a.answer is None and b.answer is None):
            continue
        aligned = None
        for ev in events:
            if a.position - align_window < ev.onset <= b.position:
                aligned = ev.event_id
                break
        flips.append(
            FlipRecord(
                from_position=a.position,
                to_position=b.position,
                from_answer=a.answer,
                to_answer=b.answer,
                # parseable-wrong -> correct only: an unparseable probe is a
                # parse artifact, not a wrong answer (same rule as
                # EventTransition.wrong_to_right), so it must not count
                # toward the §10 localization estimand
                wrong_to_right=a.answer is not None and (not a.correct) and b.correct,
                aligned_event_id=aligned,
            )
        )
    return flips


# ---------------------------------------------------------------- informative set


@dataclass
class EventTransition:
    """Probe answers straddling one event: the pre-correction answer at t0 and
    the intended answer just after the event span."""

    event_id: str
    chain_id: str
    pre_answer: Optional[str]
    post_answer: Optional[str]
    pre_correct: bool
    post_correct: bool

    @property
    def flipped(self) -> bool:
        # both ends must parse: an unparseable probe is a parse artifact, not
        # answer movement — this is the answer-predictive AUROC label (§5), so
        # None/None (== not equal) must NOT count as a flip
        return (
            self.pre_answer is not None
            and self.post_answer is not None
            and not answers_equal(self.pre_answer, self.post_answer)
        )

    @property
    def wrong_to_right(self) -> bool:
        # the informative case for the necessity test (§7.3): parseable wrong
        # answer before the event, correct after it
        return (
            self.pre_answer is not None
            and not self.pre_correct
            and self.post_correct
        )


def event_transitions(
    chain: ChainRecord,
    events: list[EventRecord],
    batch_prober: BatchProber,
) -> list[EventTransition]:
    if not events:
        return []
    end = chain.trace_token_span[1]
    positions = sorted({ev.onset for ev in events} | {min(ev.end + 1, end) for ev in events})
    answers = batch_prober(chain, positions)
    out = []
    for ev in events:
        pre = answers.get(ev.onset)
        post = answers.get(min(ev.end + 1, end))
        out.append(
            EventTransition(
                event_id=ev.event_id,
                chain_id=chain.chain_id,
                pre_answer=pre,
                post_answer=post,
                pre_correct=answers_equal(pre, chain.gold_answer),
                post_correct=answers_equal(post, chain.gold_answer),
            )
        )
    return out


def localization_summary(trajectories: list[Trajectory]) -> dict:
    """§10 localization estimand: how often wrong->right flips are
    event-aligned vs drifting in between events."""
    w2r = [f for t in trajectories for f in t.flips if f.wrong_to_right]
    aligned = [f for f in w2r if f.aligned_event_id is not None]
    return {
        "n_wrong_to_right_flips": len(w2r),
        "n_event_aligned": len(aligned),
        "event_aligned_fraction": len(aligned) / len(w2r) if w2r else None,
    }


# -------------------------------------------------------------------- persistence


def load_trajectories(path) -> dict[str, "Trajectory"]:
    """Read trajectories.jsonl (written by scripts/04 as dataclass dicts)."""
    from .io_utils import read_jsonl

    out = {}
    for d in read_jsonl(path):
        out[d["chain_id"]] = Trajectory(
            chain_id=d["chain_id"],
            points=[TrajectoryPoint(**p) for p in d["points"]],
            flips=[FlipRecord(**f) for f in d["flips"]],
        )
    return out


def load_transitions(path) -> list[EventTransition]:
    """Read transitions.jsonl. Rows carry derived flipped/wrong_to_right
    fields for human inspection; the dataclass recomputes them as properties,
    so only the underlying fields are loaded."""
    from .io_utils import read_jsonl

    fields = ("event_id", "chain_id", "pre_answer", "post_answer", "pre_correct", "post_correct")
    return [EventTransition(**{k: d[k] for k in fields}) for d in read_jsonl(path)]


# ------------------------------------------------------------ optional logit lens


def logit_lens_margin(
    tl_model,
    token_ids: list[int],
    position: int,
    target_token_id: int,
    reference_token_id: int,
) -> list[float]:
    """Optional diagnostic (§4): per-layer logit margin (target - reference)
    for the next token at `position`, reading the residual stream through the
    final layer norm + unembed."""
    import torch

    tokens = torch.tensor([token_ids[:position]], dtype=torch.long, device=tl_model.cfg.device)
    _, cache = tl_model.run_with_cache(tokens)
    margins = []
    for layer in range(tl_model.cfg.n_layers):
        resid = cache["resid_post", layer][0, -1]
        logits = tl_model.unembed(tl_model.ln_final(resid.unsqueeze(0).unsqueeze(0)))[0, 0]
        margins.append(float(logits[target_token_id] - logits[reference_token_id]))
    return margins
