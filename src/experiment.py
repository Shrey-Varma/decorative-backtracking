"""Necessity test orchestration (build order step 8, §7.3).

For every informative event (correction event straddling a wrong -> right
intended-answer transition), run K seeded continuations from the event onset
under four conditions:

    natural_resample                  sampling-noise baseline
    ablate_correction                 the intervention
    ablate_control_feature            generic-disruption control
    ablate_correction_at_control_pos  position-specificity control

All conditions share the same K seeds and the same truncation point. The
continuation backend is injected (`ContinuationRunner`) so the orchestration
is testable without a GPU; the real runner wraps TLGenerator + ClampHook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import hashlib

from .answers import answers_equal, extract_explicit_answer
from .config import ContinuationConfig, NecessityConfig
from .io_utils import ChainRecord, ContinuationRecord, EventRecord, JsonlCache
from .sae_interventions import PositionMask
from .trajectory import EventTransition, Trajectory

logger = logging.getLogger(__name__)

CONDITIONS = (
    "natural_resample",
    "ablate_correction",
    "ablate_control_feature",
    "ablate_correction_at_control_pos",
)


@dataclass
class InterventionSpec:
    feature_ids: list[int]
    ablation_mode: str                      # zero | mean | counterfactual
    position_mask: PositionMask
    counterfactual_values: Optional[list[float]] = None


class ContinuationRunner(Protocol):
    """Continue `chain` from absolute token `truncation` once per seed, under
    an optional (shared) intervention. Returns one continuation text per seed,
    in seed order. Implementations may batch the seeds into a single decode —
    they share the prefix and the clamp mask, so batching changes throughput,
    not semantics."""

    def run_batch(
        self,
        chain: ChainRecord,
        truncation: int,
        seeds: Sequence[int],
        intervention: Optional[InterventionSpec],
    ) -> list[str]: ...


# ------------------------------------------------------------------- real runner


class TLContinuationRunner:
    def __init__(self, tl_generator, sae, hook_info, cont_cfg: ContinuationConfig,
                 think_close: str, corpus_means=None):
        self.gen = tl_generator
        self.sae = sae
        self.hook_info = hook_info
        self.cfg = cont_cfg
        self.think_close = think_close
        self.corpus_means = corpus_means

    def run_batch(self, chain, truncation, seeds, intervention):
        import torch

        clamp = None
        if intervention is not None:
            from .sae_interventions import make_clamp_hook

            cf = intervention.counterfactual_values
            # one hook serves the whole batch: the mask is position-keyed and
            # identical across rows (they share the truncation point)
            clamp = make_clamp_hook(
                self.sae,
                intervention.feature_ids,
                intervention.ablation_mode,
                intervention.position_mask,
                corpus_means=self.corpus_means,
                counterfactual_values=torch.tensor(cf) if cf is not None else None,
            )
        outs = self.gen.generate_hooked_batch(
            chain.token_ids[:truncation],
            seeds=list(seeds),
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            hook_name=self.hook_info.hook_name,
            clamp_hook=clamp,
            early_stop_regex=self.cfg.early_stop_regex,
            early_stop_check_every=self.cfg.early_stop_check_every,
            think_close=self.think_close,
        )
        return [o.text for o in outs]


class HFContinuationRunner:
    """Fast interventional runner: HF model + nn.Module hook at the SAE layer
    (see HFHookedGenerator). Drop-in replacement for TLContinuationRunner —
    same conditions, same seeds contract, ~10-20x the decode throughput."""

    def __init__(self, hf_generator, sae, cont_cfg: ContinuationConfig,
                 think_close: str, corpus_means=None):
        self.gen = hf_generator
        self.sae = sae
        self.cfg = cont_cfg
        self.think_close = think_close
        self.corpus_means = corpus_means

    def run_batch(self, chain, truncation, seeds, intervention):
        import torch

        clamp = None
        if intervention is not None:
            from .sae_interventions import make_clamp_hook

            cf = intervention.counterfactual_values
            clamp = make_clamp_hook(
                self.sae,
                intervention.feature_ids,
                intervention.ablation_mode,
                intervention.position_mask,
                corpus_means=self.corpus_means,
                counterfactual_values=torch.tensor(cf) if cf is not None else None,
            )
        outs = self.gen.generate_hooked_batch(
            chain.token_ids[:truncation],
            seeds=list(seeds),
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            clamp_hook=clamp,
            early_stop_regex=self.cfg.early_stop_regex,
            early_stop_check_every=self.cfg.early_stop_check_every,
            think_close=self.think_close,
        )
        return [o.text for o in outs]


class DirectionContinuationRunner:
    """Runs continuations with diff-in-means direction ablation at a given
    layer (SAE-free localization sweep). Same seeds/outcome contract as the SAE
    runners; the position mask is passed through unchanged."""

    def __init__(self, hf_generator, direction, hook_layer: int,
                 cont_cfg: ContinuationConfig, think_close: str):
        self.gen = hf_generator
        self.direction = direction
        self.hook_layer = hook_layer
        self.cfg = cont_cfg
        self.think_close = think_close

    def run_batch(self, chain, truncation, seeds, position_mask):
        from .directions import DirectionAblationHook

        hook = DirectionAblationHook(self.direction, position_mask)
        outs = self.gen.generate_hooked_batch(
            chain.token_ids[:truncation],
            seeds=list(seeds),
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            clamp_hook=hook,
            hook_layer=self.hook_layer,
            early_stop_regex=self.cfg.early_stop_regex,
            early_stop_check_every=self.cfg.early_stop_check_every,
            think_close=self.think_close,
        )
        return [o.text for o in outs]


class DirectionAdditionContinuationRunner:
    """Sufficiency mirror of DirectionContinuationRunner (stage 12): ADDS
    α·d̂ at masked positions instead of projecting it out — tests whether
    pushing the state toward the correction direction induces a verbalized
    correction / rescues a wrong intended answer."""

    def __init__(self, hf_generator, direction, alpha: float, hook_layer: int,
                 cont_cfg: ContinuationConfig, think_close: str):
        self.gen = hf_generator
        self.direction = direction
        self.alpha = float(alpha)
        self.hook_layer = hook_layer
        self.cfg = cont_cfg
        self.think_close = think_close

    def run_batch(self, chain, truncation, seeds, position_mask):
        from .directions import DirectionAdditionHook

        hook = DirectionAdditionHook(self.direction, self.alpha, position_mask)
        outs = self.gen.generate_hooked_batch(
            chain.token_ids[:truncation],
            seeds=list(seeds),
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            clamp_hook=hook,
            hook_layer=self.hook_layer,
            early_stop_regex=self.cfg.early_stop_regex,
            early_stop_check_every=self.cfg.early_stop_check_every,
            think_close=self.think_close,
        )
        return [o.text for o in outs]


# ------------------------------------------------------------ outcome + controls


def classify_outcome(
    final_answer: Optional[str], pre_answer: Optional[str], gold: str
) -> str:
    """3-way outcome (§7.3). DISRUPTED is tracked separately so generic
    breakage is never miscounted as faithfulness."""
    if final_answer is not None and answers_equal(final_answer, gold):
        return "NO_CHANGE"
    if final_answer is not None and pre_answer is not None and answers_equal(
        final_answer, pre_answer
    ):
        return "REVERT"
    return "DISRUPTED"


def select_informative(
    transitions: list[EventTransition],
) -> list[EventTransition]:
    """The core subset (§7.3): events whose intended answer goes wrong -> right."""
    return [t for t in transitions if t.wrong_to_right]


def choose_control_position(
    chain: ChainRecord,
    events: list[EventRecord],
    event: EventRecord,
    min_distance: int,
    trajectory: Optional[Trajectory] = None,
) -> Optional[int]:
    """Matched non-event prefix for the position-specificity control: a
    truncation point at least `min_distance` tokens from every event span,
    preferring points where the intended answer is (still) wrong — matching
    the pre-correction state — and closest in depth to the real onset."""
    start, end = chain.trace_token_span

    def far_from_events(t: int) -> bool:
        return all(
            t < ev.onset - min_distance or t > ev.end + min_distance for ev in events
        )

    candidates: list[int] = []
    if trajectory is not None:
        wrong_points = [
            p.position
            for p in trajectory.points
            if not p.correct and p.answer is not None and far_from_events(p.position)
        ]
        candidates = wrong_points
    if not candidates:
        candidates = [t for t in range(start + 1, end) if far_from_events(t)]
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t - event.onset))


# ----------------------------------------------------------------- orchestration


@dataclass
class NecessityInputs:
    chain: ChainRecord
    event: EventRecord
    transition: EventTransition
    trajectory: Optional[Trajectory]
    all_events: list[EventRecord]


@dataclass
class NecessityPlanItem:
    condition: str
    truncation: int
    intervention: Optional[InterventionSpec]
    pre_answer_reference: Optional[str]      # what REVERT is measured against


def intervention_key(item: NecessityPlanItem) -> str:
    """Fingerprint of everything that defines a continuation besides
    (chain, event, condition, seed): truncation point, feature set, ablation
    mode, mask geometry, and counterfactual targets. Part of the cache key —
    re-running after feature re-selection or a config change must MISS, never
    silently reuse continuations generated under a different intervention."""
    iv = item.intervention
    if iv is None:
        return f"natural|t={item.truncation}"
    parts = [
        f"t={item.truncation}",
        f"mode={iv.ablation_mode}",
        f"mask={iv.position_mask.mode}:{iv.position_mask.t0}-{iv.position_mask.t1}",
        f"f={','.join(map(str, iv.feature_ids))}",
    ]
    # only counterfactual mode consumes the target values; hashing them for
    # zero/mean arms would make their key depend on floats that don't affect
    # the intervention, causing spurious cache misses + duplicate rows
    if iv.ablation_mode == "counterfactual" and iv.counterfactual_values is not None:
        blob = ",".join(f"{v:.6g}" for v in iv.counterfactual_values)
        parts.append(f"cf={hashlib.sha256(blob.encode()).hexdigest()[:8]}")
    return "|".join(parts)


def build_condition_plan(
    inputs: NecessityInputs,
    correction_features: Sequence[int],
    control_feature: int,
    cfg: NecessityConfig,
    counterfactual_values: Optional[list[float]] = None,
    per_candidate_features: Optional[Sequence[int]] = None,
) -> list[NecessityPlanItem]:
    ev = inputs.event
    t0, t1 = ev.onset, ev.end
    pre = inputs.transition.pre_answer

    def mask(mode: str) -> PositionMask:
        return PositionMask(mode=mode, t0=t0, t1=t1)

    # a counterfactual arm needs matched-prefix targets; if they couldn't be
    # computed for this event (no control position), that arm is unbuildable —
    # skip it rather than construct a spec that crashes make_target_values
    def usable(mode: str) -> bool:
        return mode != "counterfactual" or counterfactual_values is not None

    plan = [
        NecessityPlanItem("natural_resample", t0, None, pre),
        NecessityPlanItem(
            "ablate_correction",
            t0,
            InterventionSpec(
                list(correction_features), cfg.ablation_mode, mask(cfg.position_mask),
                counterfactual_values,
            ),
            pre,
        ),
        NecessityPlanItem(
            "ablate_control_feature",
            t0,
            InterventionSpec(
                [control_feature],
                # counterfactual targets are matched per correction feature and
                # have no analogue for the control feature; fall back to zero
                "zero" if cfg.ablation_mode == "counterfactual" else cfg.ablation_mode,
                mask(cfg.position_mask),
            ),
            pre,
        ),
    ]

    # per-candidate arms (§5: the causal test decides which candidate is the
    # mechanism — testing only the top vocab feature would presuppose the
    # ranking the spec explicitly distrusts). Directly answers the reviewer
    # objection "you just picked the wrong feature."
    non_cf_mode = "zero" if cfg.ablation_mode == "counterfactual" else cfg.ablation_mode
    for fid in per_candidate_features or []:
        if list(correction_features) == [fid]:
            continue  # already the primary arm
        plan.append(
            NecessityPlanItem(
                f"ablate_correction/cand={fid}",
                t0,
                InterventionSpec([fid], non_cf_mode, mask(cfg.position_mask)),
                pre,
            )
        )

    # joint arm: ablate ALL candidate features together, as an ADDITIONAL
    # condition (unlike ablate_top_k_jointly, which replaces the primary). If
    # the mechanism is distributed across the shortlist, knocking out any one
    # feature would miss it but knocking out all k should surface it.
    if cfg.also_ablate_joint and per_candidate_features and len(per_candidate_features) > 1:
        plan.append(
            NecessityPlanItem(
                "ablate_correction/joint",
                t0,
                InterventionSpec(list(per_candidate_features), non_cf_mode, mask(cfg.position_mask)),
                pre,
            )
        )

    # secondary reporting conditions on the correction feature (§7.2, §11)
    for mode in cfg.secondary_ablation_modes:
        if mode == cfg.ablation_mode or not usable(mode):
            continue
        plan.append(
            NecessityPlanItem(
                f"ablate_correction/mode={mode}",
                t0,
                InterventionSpec(
                    list(correction_features), mode, mask(cfg.position_mask),
                    counterfactual_values if mode == "counterfactual" else None,
                ),
                pre,
            )
        )
    for pmask in cfg.secondary_position_masks:
        if pmask == cfg.position_mask:
            continue
        plan.append(
            NecessityPlanItem(
                f"ablate_correction/mask={pmask}",
                t0,
                InterventionSpec(
                    list(correction_features), cfg.ablation_mode, mask(pmask),
                    counterfactual_values,
                ),
                pre,
            )
        )

    t_ctrl = choose_control_position(
        inputs.chain, inputs.all_events, ev, cfg.min_control_pos_distance,
        inputs.trajectory,
    )
    if t_ctrl is not None:
        span = t1 - t0
        ctrl_mask = (
            PositionMask("event_onward", t0=t_ctrl)
            if cfg.position_mask == "event_onward"
            else PositionMask("event_only", t0=t_ctrl, t1=t_ctrl + span)
        )
        # REVERT at the control position is measured against the intended
        # answer AT that position, read from the trajectory probe when
        # available — t0's pre-answer is only a fallback (the two coincide in
        # the common case, but assuming it would mislabel outcomes whenever
        # the intended answer differs at t_ctrl).
        ctrl_pre = pre
        if inputs.trajectory is not None:
            ctrl_pre = next(
                (p.answer for p in inputs.trajectory.points
                 if p.position == t_ctrl and p.answer is not None),
                pre,
            )
        plan.append(
            NecessityPlanItem(
                "ablate_correction_at_control_pos",
                t_ctrl,
                InterventionSpec(
                    list(correction_features), cfg.ablation_mode, ctrl_mask,
                    counterfactual_values,
                ),
                ctrl_pre,
            )
        )
    else:
        logger.warning(
            "no control position for %s (all positions near events)", ev.event_id
        )
    return plan


CONTINUATION_CACHE_KEY = ["chain_id", "event_id", "condition", "seed", "intervention_key"]


def run_necessity_test(
    inputs_list: list[NecessityInputs],
    runner: ContinuationRunner,
    correction_features: Sequence[int],
    control_feature: int,
    cfg: NecessityConfig,
    n_continuations: int,
    base_seed: int,
    cache: JsonlCache,
    counterfactual_values_fn=None,   # (inputs) -> list[float] | None
    per_candidate_features: Optional[Sequence[int]] = None,
) -> list[ContinuationRecord]:
    """The full grid: informative events x conditions x K shared seeds.
    Cached by (chain, event, condition, seed, intervention_key) so interrupted
    runs resume, while feature-selection or config changes re-run instead of
    reusing continuations from a different intervention."""
    records: list[ContinuationRecord] = []
    for inputs in inputs_list:
        cf_values = counterfactual_values_fn(inputs) if counterfactual_values_fn else None
        # the primary arm and the position control both use the primary mode;
        # if it's counterfactual and no matched prefix exists for this event,
        # the event is unusable — skip it rather than crash the whole grid
        if cfg.ablation_mode == "counterfactual" and cf_values is None:
            logger.warning(
                "skipping %s: counterfactual mode has no matched control prefix",
                inputs.event.event_id,
            )
            continue
        plan = build_condition_plan(
            inputs, correction_features, control_feature, cfg, cf_values,
            per_candidate_features=per_candidate_features,
        )
        gold = inputs.chain.gold_answer
        for item in plan:
            ikey = intervention_key(item)
            pending: list[int] = []
            for k in range(n_continuations):
                seed = base_seed + k  # same K seeds across all conditions (§7.3)
                hit = cache.get(
                    chain_id=inputs.chain.chain_id,
                    event_id=inputs.event.event_id,
                    condition=item.condition,
                    seed=seed,
                    intervention_key=ikey,
                )
                if hit is not None:
                    records.append(ContinuationRecord(**hit))
                else:
                    pending.append(seed)
            if not pending:
                continue
            # all cache-missing seeds of this condition run as ONE batched
            # decode (shared prefix + shared clamp mask)
            texts = runner.run_batch(
                inputs.chain, item.truncation, pending, item.intervention
            )
            for seed, text in zip(pending, texts):
                # explicit answers only: a bare-number fallback would score
                # truncated continuations as spurious REVERTs
                final = extract_explicit_answer(text)
                rec = ContinuationRecord(
                    chain_id=inputs.chain.chain_id,
                    event_id=inputs.event.event_id,
                    condition=item.condition,
                    seed=seed,
                    truncation=item.truncation,
                    ablation_mode=item.intervention.ablation_mode if item.intervention else None,
                    position_mask=item.intervention.position_mask.mode if item.intervention else None,
                    feature_ids=item.intervention.feature_ids if item.intervention else [],
                    text=text,
                    final_answer=final,
                    outcome=classify_outcome(final, item.pre_answer_reference, gold),
                    intervention_key=ikey,
                    meta={
                        "gold_answer": gold,
                        "pre_answer": item.pre_answer_reference,
                        "problem_id": inputs.chain.problem_id,
                    },
                )
                cache.put(rec)
                records.append(rec)
    return records
