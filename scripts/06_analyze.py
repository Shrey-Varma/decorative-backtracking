#!/usr/bin/env python
"""Stage 06 (§10): estimands, CIs, significance, decision rule, report.

Analyzes pilot and test splits separately (the decision rule is pre-registered
in config before the test split is examined), plus the pooled view, and joins
the localization diagnostic and the event-detection validation labels.

  python scripts/06_analyze.py --run-dir runs/exp1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import script_setup
from src.events import detection_quality
from src.io_utils import read_json, read_jsonl, write_json
from src.metrics import analyze, outcome_table, records_to_frame, split_chains, rescore_records


def main() -> None:
    cfg, cfg_dict, run_dir = script_setup("06_analyze")
    records = list(read_jsonl(run_dir / "continuations.jsonl"))
    if not records:
        sys.exit("no continuation records; run 05 first")

    # continuations.jsonl is stage 05's append-only cache; a prior run under a
    # different feature set / config leaves superseded rows behind. Keep only
    # the intervention keys the latest stage-05 run declared active, so a
    # re-selected feature never gets pooled with the feature it replaced.
    active_path = run_dir / "active_intervention_keys.json"
    if active_path.exists():
        active = set(read_json(active_path))
        kept = [r for r in records if r.get("intervention_key", "") in active]
        if len(kept) != len(records):
            print(f"filtered {len(records) - len(kept)} superseded continuation rows "
                  f"({len(kept)} active of {len(records)})")
        records = kept
    else:
        print("WARNING: active_intervention_keys.json missing — analyzing ALL rows; "
              "re-run stage 05 to regenerate it if you have re-selected features")
    if not records:
        sys.exit("no active continuation records after filtering; re-run stage 05")

    # `outcome` is derived from (final_answer, pre_answer, gold) and was frozen
    # at generation time; re-derive it so a later answer-matcher fix reaches the
    # analysis without regenerating continuations (metrics.rescore_records).
    records, rescore_summary = rescore_records(records)
    if rescore_summary["n_rescored"]:
        print(f"rescored {rescore_summary['n_rescored']} outcomes under the current "
              f"answer matcher: {rescore_summary['transitions']}")

    chain_ids = sorted({r["chain_id"] for r in records})
    split = split_chains(
        chain_ids, cfg.experiment.splits.pilot_fraction, cfg.experiment.splits.split_seed
    )

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pre-registration guard (§10): the test split stays sealed until the
    # decision rule is frozen and include_test_split is flipped on — otherwise
    # every analysis run would leak test-split estimands into threshold tuning
    split_names = ["pilot"]
    if cfg.experiment.analysis.include_test_split:
        split_names += ["test", "all"]
    results = {
        name: analyze(
            records, cfg.experiment.decision_rule, cfg.experiment.analysis,
            split=split if name != "all" else None,
            split_name=name if name != "all" else None,
        )
        for name in split_names
    }
    if not cfg.experiment.analysis.include_test_split:
        results["test"] = {
            "error": "withheld — freeze decision_rule, then re-run with "
                     "--set experiment.analysis.include_test_split=true"
        }

    table = outcome_table(records)
    table.to_csv(out_dir / "outcome_distribution.csv")
    records_to_frame(records).to_csv(out_dir / "continuations_flat.csv", index=False)

    traj_summary = {}
    ts_path = run_dir / "trajectory_summary.json"
    if ts_path.exists():
        traj_summary = read_json(ts_path)

    validation = {}
    val_path = run_dir / "events_for_validation.json"
    if val_path.exists():
        validation = detection_quality(read_json(val_path))

    report = {
        "results_by_split": results,
        "localization": traj_summary,
        "event_detection_validation": validation,
    }
    write_json(out_dir / "analysis.json", report)
    _plot(table, out_dir)
    _write_report_md(report, table, out_dir)
    print(f"analysis -> {out_dir}")
    for name in ("test", "pilot"):
        res = results.get(name, {})
        if "decision" in res:
            print(f"{name.upper()}-SPLIT VERDICT: {res['decision']['verdict']}")
            print(f"  delta={res.get('delta_ablate_correction')}, checks={res['decision']['checks']}")
            break
    else:
        print("test split withheld (pre-registration guard)")


def _plot(table, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = table.plot.bar(stacked=True, figsize=(9, 4.5), rot=20)
        ax.set_ylabel("fraction of continuations")
        ax.set_title("Outcome distribution per condition")
        plt.tight_layout()
        plt.savefig(out_dir / "outcomes_per_condition.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"plot skipped: {e}")


def _write_report_md(report: dict, table, out_dir: Path) -> None:
    lines = ["# Necessity-test analysis\n"]
    for name, res in report["results_by_split"].items():
        lines.append(f"## Split: {name}\n")
        if "error" in res:
            lines.append(f"_{res['error']}_\n")
            continue
        lines.append(f"- Δ ablate_correction: **{_fmt(res['delta_ablate_correction'])}** "
                     f"(CI {res['ci'].get('delta_ablate_correction')})")
        lines.append(f"- Δ control feature: {_fmt(res['delta_control_feature'])} "
                     f"(CI {res['ci'].get('delta_ablate_control_feature')})")
        lines.append(f"- Δ control position: {_fmt(res['delta_control_position'])} "
                     f"(CI {res['ci'].get('delta_ablate_correction_at_control_pos')})")
        lines.append(f"- faithfulness rate (net of control DISRUPTED): "
                     f"{_fmt(res['faithfulness_rate_net'])}")
        lines.append(f"- p(natural vs ablate): {_fmt(res['p_values']['natural_vs_ablate'])}")
        cand = res.get("delta_candidate_arms") or {}
        if cand:
            lines.append("- Δ per candidate / joint (all should be ~0 for a robust null):")
            for c in sorted(cand):
                label = c.replace("ablate_correction/", "")
                lines.append(f"    - {label}: {_fmt(cand[c])}")
        dirs = res.get("delta_direction_arms") or {}
        if dirs:
            lines.append("- Δ SAE-free direction ablation per layer (robustness sweep):")
            for c in sorted(dirs):
                label = c.replace("ablate_direction/", "")
                lines.append(f"    - {label}: {_fmt(dirs[c])}")
        text = res.get("delta_text_arms") or {}
        if text:
            lines.append("- Δ text-level excision vs without-correction "
                         "(faithful verbalization ⇒ with_correction strongly negative):")
            for c in sorted(text):
                lines.append(f"    - {c}: {_fmt(text[c])}")
        ins = res.get("delta_insertion_arms") or {}
        if ins:
            lines.append("- Δ text insertion vs insert_natural "
                         "(sufficient verbalization ⇒ insert_correction strongly negative):")
            for c in sorted(ins):
                lines.append(f"    - {c}: {_fmt(ins[c])}")
        suf = res.get("delta_sufficiency_arms") or {}
        if suf:
            lines.append("- Δ hooked sufficiency vs natural_at_insert "
                         "(carrier ON; negative = induces improvement):")
            for c in sorted(suf):
                lines.append(f"    - {c}: {_fmt(suf[c])}")
        lines.append(f"- **verdict: {res['decision']['verdict']}**\n")
    if report["localization"]:
        lines.append("## Localization (position-confound diagnostic)\n")
        lines.append(f"```\n{report['localization']}\n```\n")
    if report["event_detection_validation"]:
        lines.append("## Event-detection validation\n")
        lines.append(f"```\n{report['event_detection_validation']}\n```\n")
    lines.append("## Outcome distribution\n")
    try:
        lines.append(table.to_markdown())
    except ImportError:  # tabulate not installed
        lines.append(f"```\n{table.to_string()}\n```")
    lines.append(
        "\n\n*A null here is single-layer-bounded: it says this feature at this "
        "layer is not load-bearing, not that no correction mechanism exists (§11).*"
    )
    (out_dir / "report.md").write_text("\n".join(lines))


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


if __name__ == "__main__":
    main()
