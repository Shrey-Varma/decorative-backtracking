import json

from src.io_utils import (
    ChainRecord,
    JsonlCache,
    append_jsonl,
    load_chains,
    read_jsonl,
    stable_hash_fraction,
    write_manifest,
)


def test_jsonl_cache_roundtrip_and_reload(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = JsonlCache(path, ["a", "b"])
    assert cache.get(a=1, b="x") is None
    cache.put({"a": 1, "b": "x", "value": 42})
    assert cache.get(a=1, b="x")["value"] == 42
    # reload from disk
    cache2 = JsonlCache(path, ["a", "b"])
    assert len(cache2) == 1
    assert cache2.get(a=1, b="x")["value"] == 42


def test_jsonl_cache_last_write_wins(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = JsonlCache(path, ["k"])
    cache.put({"k": "a", "v": 1})
    cache.put({"k": "a", "v": 2})
    assert cache.get(k="a")["v"] == 2
    assert JsonlCache(path, ["k"]).get(k="a")["v"] == 2


def test_chain_record_roundtrip(tmp_path):
    chain = ChainRecord(
        chain_id="c1", problem_id="p1", dataset="d", prompt="q", gold_answer="42",
        seed=7, token_ids=[1, 2, 3, 4], prompt_token_len=2, text="ab",
        token_char_offsets=[(0, 1), (1, 2)], trace_token_span=(2, 4),
        final_answer="42", final_correct=True,
    )
    path = tmp_path / "chains.jsonl"
    append_jsonl(path, chain)
    loaded = load_chains(path)[0]
    assert loaded == chain
    assert loaded.abs_to_completion_token(loaded.completion_token_to_abs(1)) == 1


def test_manifest_contents(tmp_path):
    cfg = {"model": {"model_name": "m"}, "seed": 1}
    path = write_manifest(tmp_path / "run", cfg, extra={"stage": "test"})
    m = json.loads(path.read_text())
    assert m["config"] == cfg
    assert m["stage"] == "test"
    assert len(m["config_hash"]) == 12
    assert "python" in m["package_versions"]


def test_stable_hash_fraction_bounds_and_determinism():
    vals = [stable_hash_fraction(f"chain{i}", salt=3) for i in range(200)]
    assert all(0 <= v < 1 for v in vals)
    assert vals == [stable_hash_fraction(f"chain{i}", salt=3) for i in range(200)]
    assert vals != [stable_hash_fraction(f"chain{i}", salt=4) for i in range(200)]


def test_read_jsonl_missing_file(tmp_path):
    assert list(read_jsonl(tmp_path / "nope.jsonl")) == []
