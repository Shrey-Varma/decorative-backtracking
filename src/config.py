"""Config schema + loader.

YAML files compose via a top-level `includes:` list (paths relative to the
including file, deep-merged in order, later wins; the including file wins
over all includes). The merged dict validates into the pydantic tree below.
Dotted CLI overrides (`--set a.b.c=v`) are applied after merging.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- model


class SAEConfig(BaseModel):
    repo_id: str
    sae_id: str
    revision: Optional[str] = None
    dtype: str = "float32"


class ModelConfig(BaseModel):
    model_name: str
    model_revision: Optional[str] = None
    tl_model_name: Optional[str] = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    fast_backend: Literal["hf", "vllm"] = "hf"
    # TL rotary/context window: must exceed the longest sequence TL ever sees
    # (full chain up to chain.max_new_tokens, plus a stage-05 continuation up to
    # continuation.max_new_tokens). TL otherwise defaults n_ctx to 2048 and the
    # rotary table is too small for our multi-thousand-token reasoning chains.
    tl_n_ctx: int = 16384
    sae: SAEConfig


# ---------------------------------------------------------------------- generation


class ChainGenConfig(BaseModel):
    n_chains_per_problem: int = 4
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 8192
    base_seed: int = 1234


class ContinuationConfig(BaseModel):
    n_continuations: int = 8
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 4096
    early_stop_regex: Optional[str] = None
    early_stop_check_every: int = 32
    base_seed: int = 91


class ProbeConfig(BaseModel):
    template: str
    max_new_tokens: int = 24
    stop_string: Optional[str] = "}"


class GenerationConfig(BaseModel):
    think_open: str = "<think>"
    think_close: str = "</think>"
    chain: ChainGenConfig = ChainGenConfig()
    continuation: ContinuationConfig = ContinuationConfig()
    probe: ProbeConfig


# ---------------------------------------------------------------------- experiment


class DatasetConfig(BaseModel):
    name: str
    hf_path: str
    split: str = "test"
    problem_field: str = "problem"
    answer_field: str = "answer"
    id_field: Optional[str] = None
    limit: Optional[int] = None


class EventsConfig(BaseModel):
    lexicon: list[str]
    max_span_tokens: int = 24
    merge_gap_tokens: int = 4
    min_onset_gap_tokens: int = 16
    validation_sample: int = 50

    @field_validator("lexicon")
    @classmethod
    def _lowercase(cls, v: list[str]) -> list[str]:
        return [m.lower() for m in v]


class FeaturesConfig(BaseModel):
    n_candidate_features: int = 5
    corpus_max_chains: int = 200
    nonmarker_per_marker: int = 1
    position_match_bins: int = 10
    prewindow_tokens: int = 4
    activation_batch_tokens: int = 4096
    top_examples_per_feature: int = 20


class TrajectoryConfig(BaseModel):
    stride: int = 32
    align_window_tokens: int = 8


class NecessityConfig(BaseModel):
    ablation_mode: Literal["zero", "mean", "counterfactual"] = "zero"
    secondary_ablation_modes: list[Literal["zero", "mean", "counterfactual"]] = Field(
        default_factory=lambda: ["mean"]
    )
    position_mask: Literal["event_onward", "event_only"] = "event_onward"
    secondary_position_masks: list[Literal["event_onward", "event_only"]] = Field(
        default_factory=lambda: ["event_only"]
    )
    ablate_top_k_jointly: bool = False
    # test this many of the top candidates as SEPARATE ablation conditions
    # (§5: the causal test, not the vocab ranking, decides the mechanism);
    # 1 = only the top candidate, k = adds ablate_correction/cand=<fid> arms
    n_candidates_to_test: int = 1
    # add an ablate_correction/joint arm knocking out all tested candidates
    # together (catches a mechanism distributed across the shortlist)
    also_ablate_joint: bool = False
    control_feature: Literal["matched", "random"] = "matched"
    control_feature_seed: int = 0
    min_control_pos_distance: int = 32
    max_informative_events: Optional[int] = None  # cap for pilots / cost control


class SplitsConfig(BaseModel):
    pilot_fraction: float = 0.3
    split_seed: int = 17


class DecisionRuleConfig(BaseModel):
    min_delta: float = 0.30
    max_control_delta: float = 0.10
    min_net_revert_rate: float = 0.50


class AnalysisConfig(BaseModel):
    n_bootstrap: int = 2000
    ci: float = 0.95
    bootstrap_seed: int = 7
    # pre-registration guard (§10): the test split is only analyzed/printed
    # once this is flipped on — i.e. AFTER decision_rule is frozen on pilot
    include_test_split: bool = False


class VerifierConfig(BaseModel):
    enabled: bool = False
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"


class ExperimentConfig(BaseModel):
    output_dir: str = "runs"
    datasets: list[DatasetConfig]
    events: EventsConfig
    features: FeaturesConfig = FeaturesConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig()
    necessity: NecessityConfig = NecessityConfig()
    splits: SplitsConfig = SplitsConfig()
    decision_rule: DecisionRuleConfig = DecisionRuleConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    verifier: VerifierConfig = VerifierConfig()


class Config(BaseModel):
    run_name: Optional[str] = None
    model: ModelConfig
    generation: GenerationConfig
    experiment: ExperimentConfig


# ------------------------------------------------------------------------- loading


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_includes(path: Path, _stack: Optional[set[Path]] = None) -> dict:
    path = path.resolve()
    _stack = _stack or set()
    if path in _stack:
        raise ValueError(f"circular config include: {path}")
    # _stack tracks only the current ANCESTOR chain — a diamond (two siblings
    # including the same base) is legal and must not be reported as a cycle
    _stack.add(path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        includes = raw.pop("includes", [])
        merged: dict = {}
        for inc in includes:
            merged = _deep_merge(merged, _load_yaml_with_includes(path.parent / inc, _stack))
        return _deep_merge(merged, raw)
    finally:
        _stack.discard(path)


def _coerce(value: str) -> Any:
    """Parse a CLI override value with YAML semantics ('true' -> bool, etc.)."""
    return yaml.safe_load(value)


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    out = copy.deepcopy(d)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key.path=value, got: {ov}")
        key, value = ov.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            # numeric segments index into lists: experiment.datasets.0.limit=10
            node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = _coerce(value)
        else:
            node[last] = _coerce(value)
    return out


def load_config(path: str | Path, overrides: Optional[list[str]] = None) -> Config:
    merged = _load_yaml_with_includes(Path(path))
    if overrides:
        merged = apply_overrides(merged, overrides)
    return Config.model_validate(merged)
