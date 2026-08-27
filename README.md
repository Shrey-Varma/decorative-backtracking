# SAE Correction Faithfulness

Causal-mediation harness testing whether the verbalized self-corrections a
reasoning model produces in its chain of thought ("wait, that's wrong…") are
**causally load-bearing** for the final answer, or **decorative** text that
co-occurs with a conclusion the model was already reaching.

Core intervention: **knockout-and-continue**. At the point where the model
verbalizes a correction, ablate the SAE "correction feature"
(error-preserving), let the model finish, and check whether the final answer
reverts to the pre-correction answer.

- **Faithful** ⇒ ablation restores the pre-correction (wrong) answer (`REVERT`)
- **Decorative** ⇒ ablation changes nothing (`NO_CHANGE`)
- Generic breakage is tracked separately (`DISRUPTED`) and never counted as
  faithfulness.

Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.
SAE: `andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts` (JumpReLU,
residual stream layer 19 — the hook point is **read from the SAE config at
load time**, never hardcoded).

## Reproducing the paper's figures without a GPU

The raw generation output (`chains.jsonl`, `continuations.jsonl`,
`continuations_flat.csv`, ~390 MB) is not in this repository. It is not needed:
`results/exp1/release/` carries the two slices the figure scripts actually
read, and the scripts fall back to them automatically when the full files are
absent. All five data figures rebuild byte-identically from what is tracked
here.

```bash
python scripts/14_make_figures.py        # -> results/figures/
python scripts/16_make_figures_extra.py  # -> results/figures/
```

`scripts/17_make_release_subset.py` regenerates `release/` from a full run.
Everything upstream of stage 14 needs the raw output and a GPU.

## Layout

```
configs/            model.yaml + generation.yaml + experiment.yaml, composed by default.yaml
src/
  config.py         pydantic schema + YAML include/merge + CLI overrides
  io_utils.py       records, JSONL caches, run manifests, determinism
  model_loading.py  TL model + SAE loading, hook discovery, startup checks
  sae_interventions.py  error-preserving ablation, position masks, clamp hook
  generation.py     HF/vLLM/TL backends, seeded continuation, force-answer probe
  answers.py        boxed/final-answer parsing, normalization, optional LLM verifier
  events.py         lexical + feature-based correction-event detection
  features.py       vocab-association + answer-predictive scores, selection, dashboards
  trajectory.py     intended-answer trajectory, flip localization
  experiment.py     necessity test: conditions, controls, orchestration
  metrics.py        estimands, bootstrap CIs, permutation tests, decision rule
scripts/00..06      thin CLI wrappers: preflight + one per pipeline stage
tests/              CPU-only, no downloads (toy SAE + fake tokenizer/backends)
```

## Setup (any GPU box)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[model,vllm,dev]"     # add ,verifier for LLM answer grading
pytest                                  # 136 CPU-only tests, no model downloads
```

The fast backend defaults to **vLLM** (`model.fast_backend: vllm`) for all
non-interventional generation — initial chains (stage 01) and force-answer
probes (stage 04). Interventional continuations (stage 05) always run on
TransformerLens: vLLM cannot execute residual-stream hooks.
`--set model.fast_backend=hf` falls back to HF `generate`.

**Pinning (do this before a real run).** The interventional backend needs a
TransformerLens build that supports R1 distills. Validate one version/fork
commit, then pin it (`pip install git+https://github.com/TransformerLensOrg/TransformerLens@<commit>`)
and record it — the run manifest captures installed versions automatically.
Same for `sae_lens` (must match the released SAE format) and the HF model/SAE
revisions (`model.model_revision`, `model.sae.revision` in `configs/model.yaml`).
Startup checks in `model_loading.load_all` verify the SAE round-trip, the
error-preserving identity, and tokenizer consistency across backends, so a bad
pin fails loudly at load time.

## Pipeline

All stages share one `--run-dir`; every stage writes a manifest (config hash,
git SHA, seeds, package versions). Caches (`probe_cache.jsonl`,
`continuations.jsonl`, `answer_cache.jsonl`) make every stage resumable —
re-running skips completed work.

```bash
R=runs/exp1
python scripts/01_generate_chains.py    --run-dir $R   # fast backend: chains + token offsets
python scripts/02_detect_events.py      --run-dir $R   # lexical events + validation sample
python scripts/04_run_trajectory.py     --run-dir $R   # force-answer trajectories + wrong→right subset
python scripts/03_select_features.py    --run-dir $R   # feature scores (pilot split only)
python scripts/05_run_necessity_test.py --run-dir $R   # knockout-and-continue (TL backend)
python scripts/06_analyze.py            --run-dir $R   # estimands, CIs, verdict, report.md
```

Execution order is **04 before 03** (script numbers follow build order, not
run order): stage 03's answer-predictive labels come from 04's
`transitions.jsonl`, which keeps every force-answer probe on the single fast
backend — mixing TL and vLLM probes within one trajectory would let backend
numerics masquerade as answer flips at event boundaries.

Any config key is overridable per run, e.g.
`--set generation.continuation.n_continuations=4 --set experiment.necessity.ablation_mode=mean`.

The test split is sealed until you freeze the decision rule: stage 06 reports
the pilot split only, and prints test-split estimands only after
`--set experiment.analysis.include_test_split=true`.

| Stage | Backend | Key outputs |
|---|---|---|
| 01 | fast (HF/vLLM) | `chains.jsonl` (raw token ids + char offsets, parseable-answer filter) |
| 02 | — | `events.jsonl`, `events_for_validation.json` |
| 03 | TL + SAE | `features/candidate_features.json`, `features/corpus_stats.pt` |
| 04 | fast | `trajectories.jsonl`, `transitions.jsonl`, localization summary |
| 05 | TL + SAE | `continuations.jsonl` (events × 4+ conditions × K shared seeds) |
| 06 | — | `analysis/analysis.json`, `report.md`, outcome CSVs + plot |
