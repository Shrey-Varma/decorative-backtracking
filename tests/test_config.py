from pathlib import Path

import pytest

from src.config import load_config
from src.io_utils import config_hash

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_repo_default_config_validates():
    cfg = load_config(REPO_CONFIG)
    assert cfg.model.sae.repo_id.startswith("andreuka18/")
    assert cfg.generation.probe.template.startswith("\n</think>")
    assert cfg.experiment.necessity.position_mask == "event_onward"
    assert "wait" in cfg.experiment.events.lexicon
    assert cfg.experiment.features.n_candidate_features == 5


def test_includes_merge_and_override_precedence(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\nb: keep\n")
    (tmp_path / "top.yaml").write_text("includes: [base.yaml]\na: {y: 3}\n")
    from src.config import _load_yaml_with_includes

    merged = _load_yaml_with_includes(tmp_path / "top.yaml")
    assert merged == {"a": {"x": 1, "y": 3}, "b": "keep"}


def test_diamond_include_is_not_circular(tmp_path):
    (tmp_path / "base.yaml").write_text("x: 1\n")
    (tmp_path / "a.yaml").write_text("includes: [base.yaml]\na: 1\n")
    (tmp_path / "b.yaml").write_text("includes: [base.yaml]\nb: 2\n")
    (tmp_path / "top.yaml").write_text("includes: [a.yaml, b.yaml]\n")
    from src.config import _load_yaml_with_includes

    merged = _load_yaml_with_includes(tmp_path / "top.yaml")
    assert merged == {"x": 1, "a": 1, "b": 2}


def test_circular_include_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("includes: [b.yaml]\n")
    (tmp_path / "b.yaml").write_text("includes: [a.yaml]\n")
    from src.config import _load_yaml_with_includes

    with pytest.raises(ValueError, match="circular"):
        _load_yaml_with_includes(tmp_path / "a.yaml")


def test_cli_overrides_typed():
    cfg = load_config(
        REPO_CONFIG,
        overrides=[
            "generation.continuation.n_continuations=16",
            "experiment.necessity.ablation_mode=mean",
            "model.device=cpu",
            "experiment.splits.pilot_fraction=0.5",
        ],
    )
    assert cfg.generation.continuation.n_continuations == 16
    assert cfg.experiment.necessity.ablation_mode == "mean"
    assert cfg.experiment.splits.pilot_fraction == 0.5


def test_cli_override_into_list_element():
    cfg = load_config(
        REPO_CONFIG,
        overrides=[
            "experiment.datasets.0.limit=10",
            "experiment.datasets.0.split=train",
            "experiment.necessity.max_informative_events=5",
        ],
    )
    assert cfg.experiment.datasets[0].limit == 10
    assert cfg.experiment.datasets[0].split == "train"
    assert cfg.experiment.necessity.max_informative_events == 5


def test_invalid_override_rejected():
    with pytest.raises(Exception):
        load_config(REPO_CONFIG, overrides=["experiment.necessity.ablation_mode=bogus"])


def test_config_hash_stable_and_sensitive():
    d1 = load_config(REPO_CONFIG).model_dump()
    d2 = load_config(REPO_CONFIG).model_dump()
    assert config_hash(d1) == config_hash(d2)
    d3 = load_config(REPO_CONFIG, overrides=["generation.chain.base_seed=999"]).model_dump()
    assert config_hash(d3) != config_hash(d1)


def test_lexicon_lowercased():
    cfg = load_config(REPO_CONFIG, overrides=[])
    assert all(m == m.lower() for m in cfg.experiment.events.lexicon)
