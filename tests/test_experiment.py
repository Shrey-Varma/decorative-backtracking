"""Necessity-test orchestration on a scripted (GPU-free) runner: condition
plan structure, shared seeds, outcome classification, control-position
choice, and cache resume."""

from src.config import EventsConfig, NecessityConfig
from src.events import detect_events
from src.experiment import (
    CONTINUATION_CACHE_KEY,
    NecessityInputs,
    build_condition_plan,
    choose_control_position,
    classify_outcome,
    run_necessity_test,
    select_informative,
)
from src.io_utils import JsonlCache
from src.trajectory import EventTransition, Trajectory, TrajectoryPoint
from tests.conftest import DEFAULT_LEXICON, make_chain

CORRECTION_FEATURE = 11
CONTROL_FEATURE = 5


def _inputs(tokenizer):
    filler = "step " * 60
    chain = make_chain(
        tokenizer,
        f"<think> the total is 7. {filler} wait, that's wrong, recheck. "
        f"{filler} so 42. </think> \\boxed{{42}}",
        gold="42",
    )
    events = detect_events(chain, EventsConfig(lexicon=DEFAULT_LEXICON))
    assert len(events) == 1
    tr = EventTransition(
        event_id=events[0].event_id, chain_id=chain.chain_id,
        pre_answer="7", post_answer="42", pre_correct=False, post_correct=True,
    )
    return NecessityInputs(
        chain=chain, event=events[0], transition=tr, trajectory=None,
        all_events=events,
    )


# -------------------------------------------------------------- outcome + subset


def test_classify_outcome():
    assert classify_outcome("42", "7", "42") == "NO_CHANGE"
    assert classify_outcome("7", "7", "42") == "REVERT"
    assert classify_outcome("13", "7", "42") == "DISRUPTED"
    assert classify_outcome(None, "7", "42") == "DISRUPTED"
    assert classify_outcome("0.5", "7", "1/2") == "NO_CHANGE"   # normalization
    assert classify_outcome("1/7", "0.142", "42") == "DISRUPTED"


def test_select_informative():
    mk = lambda pre, pre_c, post_c: EventTransition("e", "c", pre, "x", pre_c, post_c)
    assert select_informative([mk("7", False, True)])          # wrong -> right
    assert not select_informative([mk("42", True, True)])      # right -> right
    assert not select_informative([mk("7", False, False)])     # wrong -> wrong
    assert not select_informative([mk(None, False, True)])     # unparseable pre


# ----------------------------------------------------------------- condition plan


def test_condition_plan_structure(tokenizer):
    inputs = _inputs(tokenizer)
    cfg = NecessityConfig()  # zero / event_onward primary, mean + event_only secondary
    plan = build_condition_plan(inputs, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg)
    by_name = {p.condition: p for p in plan}

    core = {"natural_resample", "ablate_correction", "ablate_control_feature",
            "ablate_correction_at_control_pos"}
    assert core <= set(by_name)
    assert "ablate_correction/mode=mean" in by_name
    assert "ablate_correction/mask=event_only" in by_name

    t0 = inputs.event.onset
    # same truncation point for all but the position control (§7.3)
    for name in core - {"ablate_correction_at_control_pos"}:
        assert by_name[name].truncation == t0
    ctrl = by_name["ablate_correction_at_control_pos"]
    assert ctrl.truncation != t0
    assert ctrl.intervention.position_mask.t0 == ctrl.truncation

    assert by_name["natural_resample"].intervention is None
    assert by_name["ablate_correction"].intervention.feature_ids == [CORRECTION_FEATURE]
    assert by_name["ablate_control_feature"].intervention.feature_ids == [CONTROL_FEATURE]
    assert by_name["ablate_correction"].intervention.position_mask.mode == "event_onward"
    assert by_name["ablate_correction/mask=event_only"].intervention.position_mask.t1 == inputs.event.end


def test_control_position_far_from_events(tokenizer):
    inputs = _inputs(tokenizer)
    t = choose_control_position(inputs.chain, inputs.all_events, inputs.event, 20)
    assert t is not None
    ev = inputs.event
    assert t < ev.onset - 20 or t > ev.end + 20
    start, end = inputs.chain.trace_token_span
    assert start < t < end


def test_intervention_key_ignores_counterfactual_for_zero_mode(tokenizer):
    """A zero/mean arm's cache key must NOT depend on counterfactual floats it
    never uses — else keys drift across hardware, causing duplicate rows."""
    from src.experiment import build_condition_plan, intervention_key

    inputs = _inputs(tokenizer)
    cfg = NecessityConfig(secondary_ablation_modes=[], secondary_position_masks=[])
    plan_a = build_condition_plan(
        inputs, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg, counterfactual_values=[1.0, 2.0]
    )
    plan_b = build_condition_plan(
        inputs, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg, counterfactual_values=[9.9, 8.8]
    )
    key = lambda plan, cond: intervention_key(next(p for p in plan if p.condition == cond))
    # zero-mode primary arm: key stable despite different cf values
    assert key(plan_a, "ablate_correction") == key(plan_b, "ablate_correction")


def test_counterfactual_without_control_position_skips_event(tokenizer):
    """Counterfactual mode + no matched prefix must SKIP the event, not crash
    make_target_values mid-grid."""
    inputs = _inputs(tokenizer)
    cfg = NecessityConfig(
        ablation_mode="counterfactual",
        secondary_ablation_modes=[], secondary_position_masks=[],
    )
    runner = ScriptedRunner(inputs.event.onset)
    import tempfile

    cache_path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    records = run_necessity_test(
        [inputs], runner, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg,
        n_continuations=1, base_seed=0, cache=JsonlCache(cache_path, CONTINUATION_CACHE_KEY),
        counterfactual_values_fn=lambda _inp: None,  # no control prefix available
    )
    assert records == []          # event skipped, no crash
    assert runner.calls == []


def test_per_candidate_arms(tokenizer):
    """§5: extra candidates get their own causal arms — the vocab ranking
    alone must not decide which feature is 'the' mechanism."""
    inputs = _inputs(tokenizer)
    cfg = NecessityConfig(secondary_ablation_modes=[], secondary_position_masks=[])
    plan = build_condition_plan(
        inputs, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg,
        per_candidate_features=[CORRECTION_FEATURE, 12, 13],
    )
    by_name = {p.condition: p for p in plan}
    assert "ablate_correction/cand=12" in by_name
    assert "ablate_correction/cand=13" in by_name
    # the primary arm already covers the top candidate — no duplicate arm
    assert f"ablate_correction/cand={CORRECTION_FEATURE}" not in by_name
    cand = by_name["ablate_correction/cand=12"]
    assert cand.intervention.feature_ids == [12]
    assert cand.truncation == inputs.event.onset


def test_joint_arm_added_alongside_individual(tokenizer):
    """also_ablate_joint adds an extra 'joint' arm (all candidates together)
    WITHOUT replacing the single-feature primary — answers the distributed-
    mechanism objection."""
    inputs = _inputs(tokenizer)
    cfg = NecessityConfig(
        secondary_ablation_modes=[], secondary_position_masks=[], also_ablate_joint=True
    )
    cands = [CORRECTION_FEATURE, 12, 13]
    plan = build_condition_plan(
        inputs, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg, per_candidate_features=cands
    )
    by_name = {p.condition: p for p in plan}
    # primary is still the single top feature
    assert by_name["ablate_correction"].intervention.feature_ids == [CORRECTION_FEATURE]
    # individual arms for the other candidates
    assert "ablate_correction/cand=12" in by_name and "ablate_correction/cand=13" in by_name
    # one joint arm knocking out ALL candidates together
    assert by_name["ablate_correction/joint"].intervention.feature_ids == cands


def test_control_position_revert_reference_from_trajectory(tokenizer):
    """REVERT at the control position is judged against the intended answer
    AT that position (from the trajectory), not blindly against t0's."""
    inputs = _inputs(tokenizer)
    start, _ = inputs.chain.trace_token_span
    wrong_pos = start + 5
    inputs.trajectory = Trajectory(
        chain_id=inputs.chain.chain_id,
        points=[TrajectoryPoint(wrong_pos, "13", False)],
    )
    cfg = NecessityConfig(secondary_ablation_modes=[], secondary_position_masks=[])
    plan = build_condition_plan(inputs, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg)
    ctrl = next(p for p in plan if p.condition == "ablate_correction_at_control_pos")
    assert ctrl.truncation == wrong_pos
    assert ctrl.pre_answer_reference == "13"   # not the t0 pre-answer "7"


def test_control_position_prefers_wrong_trajectory_points(tokenizer):
    inputs = _inputs(tokenizer)
    start, _ = inputs.chain.trace_token_span
    wrong_pos = start + 5
    traj = Trajectory(
        chain_id=inputs.chain.chain_id,
        points=[
            TrajectoryPoint(wrong_pos, "7", False),
            TrajectoryPoint(inputs.event.onset - 2, "7", False),  # too close to event
        ],
    )
    t = choose_control_position(inputs.chain, inputs.all_events, inputs.event, 20, traj)
    assert t == wrong_pos


# --------------------------------------------------------------- full orchestration


class ScriptedRunner:
    """Simulates a FAITHFUL correction feature: ablating it at the event onset
    reverts to the pre-correction answer; everything else lands on gold.
    Records per-seed calls plus the batch groupings it was invoked with."""

    def __init__(self, event_onset: int):
        self.event_onset = event_onset
        self.calls = []
        self.batches = []

    def run_batch(self, chain, truncation, seeds, intervention):
        self.batches.append((truncation, tuple(seeds)))
        out = []
        for seed in seeds:
            self.calls.append((chain.chain_id, truncation, seed,
                               tuple(intervention.feature_ids) if intervention else None))
            if (
                intervention is not None
                and CORRECTION_FEATURE in intervention.feature_ids
                and truncation == self.event_onset
            ):
                out.append("so it stays \\boxed{7}")
            else:
                out.append("confirmed \\boxed{42}")
        return out


def test_run_necessity_test_faithful_scenario(tokenizer, tmp_path):
    inputs = _inputs(tokenizer)
    runner = ScriptedRunner(inputs.event.onset)
    cfg = NecessityConfig(secondary_ablation_modes=[], secondary_position_masks=[])
    cache = JsonlCache(tmp_path / "cont.jsonl", CONTINUATION_CACHE_KEY)

    records = run_necessity_test(
        [inputs], runner, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg,
        n_continuations=3, base_seed=100, cache=cache,
    )
    by_cond = {}
    for r in records:
        by_cond.setdefault(r.condition, []).append(r)

    assert all(r.outcome == "NO_CHANGE" for r in by_cond["natural_resample"])
    assert all(r.outcome == "REVERT" for r in by_cond["ablate_correction"])
    assert all(r.outcome == "NO_CHANGE" for r in by_cond["ablate_control_feature"])
    assert all(r.outcome == "NO_CHANGE" for r in by_cond["ablate_correction_at_control_pos"])
    # same K seeds across all conditions (§7.3)
    for recs in by_cond.values():
        assert sorted(r.seed for r in recs) == [100, 101, 102]


def test_seeds_of_a_condition_run_as_one_batch(tokenizer, tmp_path):
    """All cache-missing seeds of a condition go to the runner as ONE batch
    (they share prefix + clamp mask); partially-cached conditions batch only
    the missing seeds."""
    inputs = _inputs(tokenizer)
    cfg = NecessityConfig(secondary_ablation_modes=[], secondary_position_masks=[])
    path = tmp_path / "cont.jsonl"

    runner1 = ScriptedRunner(inputs.event.onset)
    run_necessity_test(
        [inputs], runner1, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg,
        n_continuations=3, base_seed=100, cache=JsonlCache(path, CONTINUATION_CACHE_KEY),
    )
    # 4 conditions x one batch each, every batch carrying all 3 seeds
    assert len(runner1.batches) == 4
    assert all(seeds == (100, 101, 102) for _, seeds in runner1.batches)

    # extend K to 5 on the same cache: only the 2 new seeds run, as one batch
    runner2 = ScriptedRunner(inputs.event.onset)
    run_necessity_test(
        [inputs], runner2, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg,
        n_continuations=5, base_seed=100, cache=JsonlCache(path, CONTINUATION_CACHE_KEY),
    )
    assert all(seeds == (103, 104) for _, seeds in runner2.batches)


def test_run_necessity_test_resumes_from_cache(tokenizer, tmp_path):
    inputs = _inputs(tokenizer)
    cfg = NecessityConfig(secondary_ablation_modes=[], secondary_position_masks=[])
    path = tmp_path / "cont.jsonl"

    runner1 = ScriptedRunner(inputs.event.onset)
    cache1 = JsonlCache(path, CONTINUATION_CACHE_KEY)
    first = run_necessity_test(
        [inputs], runner1, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg,
        n_continuations=2, base_seed=0, cache=cache1,
    )
    assert len(runner1.calls) == len(first)

    runner2 = ScriptedRunner(inputs.event.onset)
    cache2 = JsonlCache(path, CONTINUATION_CACHE_KEY)
    second = run_necessity_test(
        [inputs], runner2, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg,
        n_continuations=2, base_seed=0, cache=cache2,
    )
    assert runner2.calls == []  # fully served from cache
    assert {(r.condition, r.seed, r.outcome) for r in second} == {
        (r.condition, r.seed, r.outcome) for r in first
    }


def test_cache_misses_when_intervention_changes(tokenizer, tmp_path):
    """Re-running after feature re-selection must NOT reuse continuations
    generated under the old intervention — the fingerprint is in the key."""
    inputs = _inputs(tokenizer)
    cfg = NecessityConfig(secondary_ablation_modes=[], secondary_position_masks=[])
    path = tmp_path / "cont.jsonl"

    runner1 = ScriptedRunner(inputs.event.onset)
    run_necessity_test(
        [inputs], runner1, [CORRECTION_FEATURE], CONTROL_FEATURE, cfg,
        n_continuations=1, base_seed=0, cache=JsonlCache(path, CONTINUATION_CACHE_KEY),
    )

    runner2 = ScriptedRunner(inputs.event.onset)
    run_necessity_test(
        [inputs], runner2, [99], CONTROL_FEATURE, cfg,   # re-selected feature
        n_continuations=1, base_seed=0, cache=JsonlCache(path, CONTINUATION_CACHE_KEY),
    )
    # natural_resample and the (unchanged) control-feature arm stay cached;
    # ablate_correction and the position control changed feature set -> re-run
    assert len(runner2.calls) == 2
    rerun_features = {feats for _, _, _, feats in runner2.calls}
    assert rerun_features == {(99,)}
