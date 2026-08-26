import numpy as np

from src.config import AnalysisConfig, DecisionRuleConfig
from src.metrics import (
    analyze,
    chain_level,
    compute_estimands,
    outcome_table,
    paired_permutation_test,
    records_to_frame,
    split_chains,
)

CONDITIONS = (
    "natural_resample",
    "ablate_correction",
    "ablate_control_feature",
    "ablate_correction_at_control_pos",
)


def _records(n_chains: int, ablate_outcome: str, n_seeds: int = 4) -> list[dict]:
    """Synthetic continuation records: natural + controls always NO_CHANGE;
    ablate_correction gets `ablate_outcome`."""
    recs = []
    for c in range(n_chains):
        for cond in CONDITIONS:
            for s in range(n_seeds):
                outcome = ablate_outcome if cond == "ablate_correction" else "NO_CHANGE"
                recs.append(
                    {
                        "chain_id": f"chain{c}",
                        "event_id": f"chain{c}__e0",
                        "condition": cond,
                        "seed": s,
                        "outcome": outcome,
                    }
                )
    return recs


def test_estimands_faithful():
    chains = chain_level(records_to_frame(_records(8, "REVERT")))
    est = compute_estimands(chains)
    assert est["delta_ablate_correction"] == 1.0
    assert est["delta_control_feature"] == 0.0
    assert est["delta_control_position"] == 0.0
    assert est["faithfulness_rate_net"] == 1.0
    assert est["per_condition"]["ablate_correction"]["revert_rate"] == 1.0


def test_estimands_decorative():
    chains = chain_level(records_to_frame(_records(8, "NO_CHANGE")))
    est = compute_estimands(chains)
    assert est["delta_ablate_correction"] == 0.0
    assert est["faithfulness_rate_net"] == 0.0


def test_disruption_not_counted_as_faithful():
    chains = chain_level(records_to_frame(_records(8, "DISRUPTED")))
    est = compute_estimands(chains)
    assert est["delta_ablate_correction"] == 1.0          # accuracy drops...
    assert est["per_condition"]["ablate_correction"]["revert_rate"] == 0.0
    assert est["faithfulness_rate_net"] == 0.0            # ...but no REVERT credit


def test_analyze_verdicts():
    cfg = AnalysisConfig(n_bootstrap=200)
    rule = DecisionRuleConfig()
    faithful = analyze(_records(10, "REVERT"), rule, cfg)
    assert faithful["decision"]["verdict"] == "faithful"
    lo, hi = faithful["ci"]["delta_ablate_correction"]
    assert lo <= faithful["delta_ablate_correction"] <= hi
    assert faithful["p_values"]["natural_vs_ablate"] < 0.05

    decorative = analyze(_records(10, "NO_CHANGE"), rule, cfg)
    assert decorative["decision"]["verdict"] == "decorative"


def test_analyze_inconclusive_when_controls_also_break():
    recs = []
    for c in range(10):
        for cond in CONDITIONS:
            for s in range(4):
                outcome = (
                    "REVERT"
                    if cond in ("ablate_correction", "ablate_control_feature")
                    else "NO_CHANGE"
                )
                recs.append(
                    {"chain_id": f"c{c}", "event_id": f"c{c}__e0",
                     "condition": cond, "seed": s, "outcome": outcome}
                )
    res = analyze(recs, DecisionRuleConfig(), AnalysisConfig(n_bootstrap=100))
    assert res["decision"]["verdict"] == "inconclusive"
    assert res["decision"]["checks"]["control_feature_specific"] is False


def test_permutation_test_null():
    chains = chain_level(records_to_frame(_records(12, "NO_CHANGE")))
    p = paired_permutation_test(chains, "natural_resample", "ablate_correction", seed=1)
    assert p > 0.5  # no effect -> p near 1


def test_outcome_table_rows_sum_to_one():
    table = outcome_table(_records(4, "REVERT"))
    assert np.allclose(table.sum(axis=1).to_numpy(), 1.0)
    assert table.loc["ablate_correction", "REVERT"] == 1.0


def test_empty_records_do_not_crash():
    df = records_to_frame([])
    assert df.empty and "outcome" in df.columns
    res = analyze([], DecisionRuleConfig(), AnalysisConfig(n_bootstrap=10))
    assert "error" in res


def test_text_arm_deltas_use_within_experiment_reference():
    """Text arms compare against text_without_correction (same backend/seeds),
    NOT against the HF natural baseline."""
    recs = []
    for c in range(6):
        for cond, outcome in [
            ("natural_resample", "NO_CHANGE"),
            ("ablate_correction", "NO_CHANGE"),
            ("ablate_control_feature", "NO_CHANGE"),
            ("ablate_correction_at_control_pos", "NO_CHANGE"),
            ("text_without_correction", "DISRUPTED"),   # ref: 0% accuracy
            ("text_with_correction", "NO_CHANGE"),      # 100% accuracy
        ]:
            for s in range(2):
                recs.append({"chain_id": f"c{c}", "event_id": f"c{c}__e0",
                             "condition": cond, "seed": s, "outcome": outcome})
    chains = chain_level(records_to_frame(recs))
    est = compute_estimands(chains)
    # ref acc 0, arm acc 1 -> delta = 0 - 1 = -1 (with-correction HELPS)
    assert est["delta_text_arms"]["text_with_correction"] == -1.0
    # and natural-based deltas unaffected by text arms
    assert est["delta_ablate_correction"] == 0.0


def test_insertion_and_sufficiency_deltas_use_family_references():
    """Sufficiency arms (stages 11/12) compare within their own family —
    text arms vs insert_natural, hooked arms vs natural_at_insert — and an
    arm that HELPS shows up NEGATIVE (delta = ref - arm)."""
    recs = []
    for c in range(6):
        for cond, outcome in [
            ("natural_resample", "NO_CHANGE"),
            ("insert_natural", "DISRUPTED"),                     # base rescue 0
            ("insert_correction/v=assert_mistake", "NO_CHANGE"), # rescues -> helps
            ("insert_filler", "DISRUPTED"),                      # extra tokens: 0
            ("natural_at_insert", "DISRUPTED"),
            ("clamp_on_feature/scale=1", "NO_CHANGE"),           # feature ON rescues
            ("add_direction/layer=19/scale=1", "DISRUPTED"),
        ]:
            for s in range(2):
                recs.append({"chain_id": f"c{c}", "event_id": f"c{c}__e0",
                             "condition": cond, "seed": s, "outcome": outcome})
    est = compute_estimands(chain_level(records_to_frame(recs)))
    assert est["delta_insertion_arms"]["insert_correction/v=assert_mistake"] == -1.0
    assert est["delta_insertion_arms"]["insert_filler"] == 0.0
    assert "insert_natural" not in est["delta_insertion_arms"]
    assert est["delta_sufficiency_arms"]["clamp_on_feature/scale=1"] == -1.0
    assert est["delta_sufficiency_arms"]["add_direction/layer=19/scale=1"] == 0.0
    # hooked arms are NOT in the text-insertion family and vice versa
    assert "clamp_on_feature/scale=1" not in est["delta_insertion_arms"]
    assert "insert_correction/v=assert_mistake" not in est["delta_sufficiency_arms"]


def test_split_chains_deterministic_and_proportional():
    ids = [f"chain{i}" for i in range(500)]
    s1 = split_chains(ids, 0.3, seed=17)
    s2 = split_chains(ids, 0.3, seed=17)
    assert s1 == s2
    pilot_frac = sum(1 for v in s1.values() if v == "pilot") / len(ids)
    assert 0.2 < pilot_frac < 0.4
    assert split_chains(ids, 0.3, seed=18) != s1


def test_analyze_respects_split():
    recs = _records(10, "REVERT")
    split = {f"chain{i}": ("pilot" if i < 5 else "test") for i in range(10)}
    res = analyze(recs, DecisionRuleConfig(), AnalysisConfig(n_bootstrap=50),
                  split=split, split_name="pilot")
    assert res["per_condition"]["ablate_correction"]["n_chains"] == 5
