from src.config import EventsConfig, TrajectoryConfig
from src.events import detect_events
from src.io_utils import EventRecord
from src.trajectory import (
    Trajectory,
    TrajectoryPoint,
    compute_trajectory,
    event_transitions,
    localization_summary,
    localize_flips,
    probe_grid,
)
from tests.conftest import DEFAULT_LEXICON, make_chain


def _points(specs):
    return [TrajectoryPoint(position=p, answer=a, correct=c) for p, a, c in specs]


def _event(chain_id, event_id, onset, end=None):
    return EventRecord(chain_id, event_id, onset, end if end is not None else onset + 3,
                       "wait", "wait", "ctx")


def test_flip_aligned_to_event():
    points = _points([(10, "7", False), (42, "7", False), (74, "42", True)])
    events = [_event("c", "c__e0", 60)]
    flips = localize_flips(points, events, align_window=8)
    assert len(flips) == 1
    f = flips[0]
    assert f.wrong_to_right and f.aligned_event_id == "c__e0"
    assert (f.from_position, f.to_position) == (42, 74)


def test_flip_not_aligned_when_no_event_in_interval():
    points = _points([(10, "7", False), (42, "42", True)])
    events = [_event("c", "c__e0", 80)]
    flips = localize_flips(points, events, align_window=8)
    assert len(flips) == 1
    assert flips[0].aligned_event_id is None


def test_alignment_window_pads_backwards():
    """Probe grid lands exactly on the onset: the flip interval ends at the
    onset probe, and the window must still credit the event."""
    points = _points([(10, "7", False), (60, "42", True)])
    events = [_event("c", "c__e0", 60)]
    flips = localize_flips(points, events, align_window=8)
    assert flips[0].aligned_event_id == "c__e0"


def test_unparseable_pre_answer_is_not_wrong_to_right():
    """A None->correct transition is a parse artifact, not a correction; it
    must not inflate the localization estimand (same rule as
    EventTransition.wrong_to_right)."""
    points = _points([(10, None, False), (42, "42", True)])
    flips = localize_flips(points, [], align_window=8)
    assert len(flips) == 1
    assert not flips[0].wrong_to_right


def test_right_to_wrong_flip_is_recorded_but_not_w2r():
    points = _points([(10, "42", True), (42, "7", False)])
    flips = localize_flips(points, [], align_window=8)
    assert len(flips) == 1 and not flips[0].wrong_to_right


def test_equivalent_answers_do_not_flip():
    points = _points([(10, "1/2", False), (42, "0.5", False)])
    assert localize_flips(points, [], 8) == []


def test_probe_grid_includes_event_bounds(tokenizer):
    chain = make_chain(
        tokenizer,
        "<think> " + "a " * 100 + " wait, recheck this. " + "b " * 100 + "</think> \\boxed{1}",
    )
    events = detect_events(chain, EventsConfig(lexicon=DEFAULT_LEXICON))
    grid = probe_grid(chain, events, stride=32)
    ev = events[0]
    assert ev.onset in grid
    assert ev.end + 1 in grid
    start, end = chain.trace_token_span
    assert all(start < p <= end for p in grid)
    assert end in grid


def test_event_transition_flipped_requires_both_parseable():
    """flipped is the answer-predictive AUROC label (§5) — a None end is a
    parse artifact, not answer movement."""
    from src.trajectory import EventTransition

    both_none = EventTransition("e", "c", None, None, False, False)
    assert not both_none.flipped
    one_none = EventTransition("e", "c", "7", None, False, False)
    assert not one_none.flipped
    real = EventTransition("e", "c", "7", "42", False, True)
    assert real.flipped
    same = EventTransition("e", "c", "42", "42", True, True)
    assert not same.flipped


def test_compute_trajectory_and_transitions(tokenizer):
    chain = make_chain(
        tokenizer,
        "<think> " + "a " * 60 + " wait, recheck this sum. " + "b " * 60 + "</think> \\boxed{42}",
        gold="42",
    )
    events = detect_events(chain, EventsConfig(lexicon=DEFAULT_LEXICON))
    ev = events[0]

    def prober(ch, positions):
        # wrong before the event, right after it
        return {t: ("7" if t <= ev.onset else "42") for t in positions}

    traj = compute_trajectory(chain, events, prober, TrajectoryConfig(stride=16))
    w2r = [f for f in traj.flips if f.wrong_to_right]
    assert len(w2r) == 1
    assert w2r[0].aligned_event_id == ev.event_id

    trans = event_transitions(chain, events, prober)
    assert len(trans) == 1
    t = trans[0]
    assert t.pre_answer == "7" and t.post_answer == "42"
    assert t.wrong_to_right and t.flipped


def test_localization_summary():
    t1 = Trajectory("c1")
    t1.flips = localize_flips(
        _points([(10, "7", False), (40, "42", True)]), [_event("c1", "c1__e0", 35)], 8
    )
    t2 = Trajectory("c2")
    t2.flips = localize_flips(
        _points([(10, "7", False), (40, "42", True)]), [], 8
    )
    s = localization_summary([t1, t2])
    assert s["n_wrong_to_right_flips"] == 2
    assert s["n_event_aligned"] == 1
    assert abs(s["event_aligned_fraction"] - 0.5) < 1e-9
