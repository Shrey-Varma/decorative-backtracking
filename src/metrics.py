"""Outcome classification, estimands, significance tests (step 8, §10).

Aggregation unit is the CHAIN: continuations average within event, events
average within chain, and bootstrap/permutation resampling happens over
chains. All computed on the wrong->right corrected subset.

Estimands:
  - ablation effect  Δ = acc(natural_resample) - acc(ablate_correction)
  - specificity      Δ_control_feature, Δ_control_position  (should be ~0)
  - faithfulness     REVERT rate under ablate_correction, net of the
                     DISRUPTED rate under the control-feature condition
  - localization     (from trajectory.localization_summary, joined by 06_analyze)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import AnalysisConfig, DecisionRuleConfig
from .io_utils import stable_hash_fraction

PRIMARY = "ablate_correction"
NATURAL = "natural_resample"
CTRL_FEATURE = "ablate_control_feature"
CTRL_POSITION = "ablate_correction_at_control_pos"


# --------------------------------------------------------------------- rescoring


def rescore_records(records: list[dict]) -> tuple[list[dict], dict]:
    """Re-derive `outcome` from (final_answer, meta.pre_answer, meta.gold_answer)
    with the CURRENT answer matcher.

    `outcome` is a derived field frozen into continuations.jsonl at generation
    time, so a later fix to `answers.normalize_answer` would otherwise never
    reach the analysis. It is deterministic in three stored fields, so
    recomputing is exact and idempotent — no regeneration needed. Returns the
    rescored records and a summary of what moved.
    """
    from .experiment import classify_outcome

    changed: dict[str, int] = {}
    out = []
    for r in records:
        meta = r.get("meta") or {}
        gold = meta.get("gold_answer")
        if gold is None:                      # nothing to rescore against
            out.append(r)
            continue
        new = classify_outcome(r.get("final_answer"), meta.get("pre_answer"), gold)
        if new != r["outcome"]:
            changed[f"{r['outcome']}->{new}"] = changed.get(f"{r['outcome']}->{new}", 0) + 1
            r = {**r, "outcome": new, "outcome_as_generated": r["outcome"]}
        out.append(r)
    return out, {"n_records": len(records),
                 "n_rescored": sum(changed.values()),
                 "transitions": changed}


def failed_chain_purity(records: list[dict]) -> dict:
    """Audit the `failed_chain` sufficiency population under the current matcher.

    That population is selected by `insertion.failed_chains()` as "final answer
    parses AND is wrong", so any answer-matcher false negative is *guaranteed*
    admission — the selection rule is confounded with the measurement error it
    is supposed to be independent of. This reports which of its chains are
    genuinely never-rescued (pre_answer != gold), so downstream analysis can
    restrict to them.
    """
    from .answers import answers_equal

    per_chain: dict[str, tuple] = {}
    for r in records:
        meta = r.get("meta") or {}
        if meta.get("population") != "failed_chain":
            continue
        per_chain.setdefault(r["chain_id"], (meta.get("pre_answer"), meta.get("gold_answer")))
    clean = sorted(c for c, (pre, gold) in per_chain.items()
                   if not answers_equal(pre, gold))
    return {
        "n_chains": len(per_chain),
        "n_truly_failed": len(clean),
        "n_contaminated": len(per_chain) - len(clean),
        "clean_chain_ids": clean,
    }


# ------------------------------------------------------------------------- frames


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=["chain_id", "event_id", "condition", "seed", "outcome",
                     "is_correct", "is_revert", "is_disrupted"]
        )
    df = pd.DataFrame(records)
    df["is_correct"] = df["outcome"] == "NO_CHANGE"
    df["is_revert"] = df["outcome"] == "REVERT"
    df["is_disrupted"] = df["outcome"] == "DISRUPTED"
    return df


def chain_level(df: pd.DataFrame) -> pd.DataFrame:
    """continuations -> event means -> chain means, per condition."""
    ev = (
        df.groupby(["chain_id", "event_id", "condition"])[
            ["is_correct", "is_revert", "is_disrupted"]
        ]
        .mean()
        .reset_index()
    )
    return (
        ev.groupby(["chain_id", "condition"])[["is_correct", "is_revert", "is_disrupted"]]
        .mean()
        .reset_index()
    )


def _paired(chains: pd.DataFrame, cond_a: str, cond_b: str, col: str) -> pd.DataFrame:
    a = chains[chains.condition == cond_a].set_index("chain_id")[col]
    b = chains[chains.condition == cond_b].set_index("chain_id")[col]
    return pd.DataFrame({"a": a, "b": b}).dropna()


def paired_delta(
    chains: pd.DataFrame, cond: str, col: str = "is_correct", ref: str = NATURAL
) -> Optional[float]:
    """Mean per-chain paired difference ref - cond. THE delta statistic:
    both the point estimate and the bootstrap CI must compute exactly this."""
    pairs = _paired(chains, ref, cond, col)
    return float((pairs.a - pairs.b).mean()) if len(pairs) else None


# ---------------------------------------------------------------------- estimands


def compute_estimands(chains: pd.DataFrame) -> dict:
    out: dict = {"per_condition": {}}
    for cond, grp in chains.groupby("condition"):
        out["per_condition"][cond] = {
            "n_chains": int(grp.chain_id.nunique()),
            "accuracy": float(grp.is_correct.mean()),
            "revert_rate": float(grp.is_revert.mean()),
            "disrupted_rate": float(grp.is_disrupted.mean()),
        }

    out["delta_ablate_correction"] = paired_delta(chains, PRIMARY)
    out["delta_control_feature"] = paired_delta(chains, CTRL_FEATURE)
    out["delta_control_position"] = paired_delta(chains, CTRL_POSITION)

    # Δ for every additional ablate_correction* arm (per-candidate #2..k and
    # the joint arm) — directly answers "did you pick the wrong feature?": if
    # any candidate were load-bearing its Δ would be large and positive.
    out["delta_candidate_arms"] = {
        cond: paired_delta(chains, cond)
        for cond in chains.condition.unique()
        if cond.startswith("ablate_correction/cand=") or cond == "ablate_correction/joint"
    }

    # Δ for the SAE-free direction-ablation sweep (§11): natural minus
    # ablate_direction per layer — a robust null has all of these ~0 too.
    out["delta_direction_arms"] = {
        cond: paired_delta(chains, cond)
        for cond in chains.condition.unique()
        if cond.startswith("ablate_direction/")
    }

    # text-level excision: compare WITHIN the text experiment (all its arms ran
    # on one backend with shared seeds) — reference is without-correction.
    # Faithful verbalization ⇒ with_correction strongly NEGATIVE (having the
    # correction in context should beat not having it).
    text_ref = "text_without_correction"
    if text_ref in set(chains.condition.unique()):
        out["delta_text_arms"] = {
            cond: paired_delta(chains, cond, ref=text_ref)
            for cond in chains.condition.unique()
            if cond.startswith("text_") and cond != text_ref
        }

    # sufficiency (insertion) families (stages 11/12): these arms live at the
    # INSERTION point (intended answer wrong, far from events), not the event
    # onset, so their reference is the matching natural arm on the SAME
    # backend — text arms vs insert_natural, hooked (feature/direction ON)
    # arms vs natural_at_insert. Sign convention: paired_delta is ref − arm,
    # so an insertion that HELPS (sufficiency evidence) shows up NEGATIVE.
    ins_ref = "insert_natural"
    if ins_ref in set(chains.condition.unique()):
        out["delta_insertion_arms"] = {
            cond: paired_delta(chains, cond, ref=ins_ref)
            for cond in chains.condition.unique()
            if cond.startswith("insert_") and cond != ins_ref
        }
    suf_ref = "natural_at_insert"
    if suf_ref in set(chains.condition.unique()):
        out["delta_sufficiency_arms"] = {
            cond: paired_delta(chains, cond, ref=suf_ref)
            for cond in chains.condition.unique()
            if cond.startswith(("clamp_on", "add_direction/"))
        }

    revert = out["per_condition"].get(PRIMARY, {}).get("revert_rate")
    ctrl_disrupt = out["per_condition"].get(CTRL_FEATURE, {}).get("disrupted_rate")
    out["faithfulness_rate_net"] = (
        revert - ctrl_disrupt if revert is not None and ctrl_disrupt is not None else None
    )
    return out


# ---------------------------------------------------------------------- inference


def bootstrap_ci(
    chains: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], Optional[float]],
    cfg: AnalysisConfig,
) -> Optional[tuple[float, float]]:
    """Percentile bootstrap resampling CHAINS with replacement (§10)."""
    rng = np.random.default_rng(cfg.bootstrap_seed)
    ids = chains.chain_id.unique()
    if len(ids) == 0:
        return None
    by_chain = {c: g for c, g in chains.groupby("chain_id")}
    stats = []
    for _ in range(cfg.n_bootstrap):
        sample = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat(
            [by_chain[c].assign(chain_id=f"{c}#{i}") for i, c in enumerate(sample)],
            ignore_index=True,
        )
        s = stat_fn(boot)
        if s is not None:
            stats.append(s)
    if not stats:
        return None
    alpha = (1 - cfg.ci) / 2
    return (float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha)))


def paired_permutation_test(
    chains: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    n_perm: int = 10000,
    seed: int = 0,
    col: str = "is_correct",
) -> Optional[float]:
    """Two-sided sign-flip permutation test on per-chain paired differences."""
    pairs = _paired(chains, cond_a, cond_b, col)
    if len(pairs) == 0:
        return None
    diffs = (pairs.a - pairs.b).to_numpy()
    observed = abs(diffs.mean())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    perm_means = np.abs((flips * diffs).mean(axis=1))
    return float((np.sum(perm_means >= observed - 1e-12) + 1) / (n_perm + 1))


# ------------------------------------------------------------------ decision rule


def split_chains(chain_ids: list[str], pilot_fraction: float, seed: int) -> dict[str, str]:
    """Deterministic pilot/test split by chain hash (§10: freeze the decision
    rule before looking at the test split)."""
    return {
        c: ("pilot" if stable_hash_fraction(c, salt=seed) < pilot_fraction else "test")
        for c in chain_ids
    }


def apply_decision_rule(estimands: dict, rule: DecisionRuleConfig) -> dict:
    delta = estimands.get("delta_ablate_correction")
    d_cf = estimands.get("delta_control_feature")
    d_cp = estimands.get("delta_control_position")
    net = estimands.get("faithfulness_rate_net")

    checks = {
        "delta_exceeds_min": delta is not None and delta >= rule.min_delta,
        "control_feature_specific": d_cf is not None and abs(d_cf) <= rule.max_control_delta,
        "control_position_specific": d_cp is not None and abs(d_cp) <= rule.max_control_delta,
        "net_revert_rate_sufficient": net is not None and net >= rule.min_net_revert_rate,
    }
    faithful = all(checks.values())
    decorative = (
        delta is not None
        and abs(delta) <= rule.max_control_delta
        and checks["control_feature_specific"]
    )
    return {
        "checks": checks,
        "verdict": "faithful" if faithful else ("decorative" if decorative else "inconclusive"),
    }


# ---------------------------------------------------------------------- top level


def analyze(
    records: list[dict],
    rule: DecisionRuleConfig,
    analysis_cfg: AnalysisConfig,
    split: Optional[dict[str, str]] = None,
    split_name: Optional[str] = None,
) -> dict:
    df = records_to_frame(records)
    if split is not None and split_name is not None:
        df = df[df.chain_id.map(split) == split_name]
    if df.empty:
        return {"error": "no records in split", "split": split_name}
    chains = chain_level(df)
    est = compute_estimands(chains)

    est["ci"] = {
        f"delta_{cond}": bootstrap_ci(
            chains, lambda boot, c=cond: paired_delta(boot, c), analysis_cfg
        )
        for cond in (PRIMARY, CTRL_FEATURE, CTRL_POSITION)
    }
    est["p_values"] = {
        "natural_vs_ablate": paired_permutation_test(
            chains, NATURAL, PRIMARY, seed=analysis_cfg.bootstrap_seed
        ),
        "natural_vs_control_feature": paired_permutation_test(
            chains, NATURAL, CTRL_FEATURE, seed=analysis_cfg.bootstrap_seed
        ),
    }
    est["decision"] = apply_decision_rule(est, rule)
    est["split"] = split_name or "all"
    est["n_records"] = int(len(df))
    return est


def outcome_table(records: list[dict]) -> pd.DataFrame:
    """Per-condition answer-outcome distribution over all continuations (§10)."""
    df = records_to_frame(records)
    return (
        df.groupby("condition")["outcome"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=["REVERT", "NO_CHANGE", "DISRUPTED"], fill_value=0.0)
    )
