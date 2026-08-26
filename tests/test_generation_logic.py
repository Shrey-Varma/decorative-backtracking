"""Pure-logic generation tests: offset maps, trace spans, probe assembly,
probe caching. No model, no downloads."""

from src.config import ProbeConfig
from src.generation import (
    GenOut,
    char_to_completion_token,
    compute_token_char_offsets,
    find_trace_span,
    make_cached_batch_prober,
)
from src.io_utils import JsonlCache
from tests.conftest import make_chain


def test_offsets_roundtrip(tokenizer):
    text = "<think> a b c </think> answer is \\boxed{42}"
    ids = tokenizer.encode(text)
    decoded, offsets = compute_token_char_offsets(tokenizer, ids)
    assert decoded == text
    assert len(offsets) == len(ids)
    # spans tile the text exactly
    assert offsets[0][0] == 0 and offsets[-1][1] == len(text)
    for (a, b), (c, d) in zip(offsets, offsets[1:]):
        assert b == c


def test_offsets_fallback_when_reencode_mismatches(tokenizer):
    """Force the fallback path by handing ids whose decode re-encodes
    differently (simulated via a tokenizer whose __call__ lies)."""

    class LyingTokenizer:
        eos_token_id = 0

        def decode(self, ids, skip_special_tokens=False):
            return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        def __call__(self, *a, **k):
            return {"input_ids": [999], "offset_mapping": [(0, 1)]}

    text = "one two three"
    ids = tokenizer.encode(text)
    decoded, offsets = compute_token_char_offsets(LyingTokenizer(), ids)
    assert decoded == text
    assert len(offsets) == len(ids)
    assert offsets[-1][1] == len(text)


def test_char_to_completion_token():
    offsets = [(0, 4), (4, 9), (9, 15)]
    assert char_to_completion_token(offsets, 0) == 0
    assert char_to_completion_token(offsets, 4) == 1
    assert char_to_completion_token(offsets, 8) == 1
    assert char_to_completion_token(offsets, 14) == 2


def test_find_trace_span(tokenizer):
    text = "<think> step one step two </think> final \\boxed{5}"
    ids = tokenizer.encode(text)
    decoded, offsets = compute_token_char_offsets(tokenizer, ids)
    start, end = find_trace_span(decoded, offsets, prompt_token_len=10,
                                 think_open="<think>", think_close="</think>")
    assert start >= 10  # absolute
    span_text = "".join(
        decoded[offsets[i - 10][0]: offsets[i - 10][1]] for i in range(start, end)
    )
    assert "step one step two" in span_text
    assert "</think>" not in span_text
    assert "boxed" not in span_text


def test_find_trace_span_unclosed_think(tokenizer):
    text = "<think> never closes the block 42"
    ids = tokenizer.encode(text)
    decoded, offsets = compute_token_char_offsets(tokenizer, ids)
    start, end = find_trace_span(decoded, offsets, 0, "<think>", "</think>")
    assert end == len(ids)  # whole completion counts as trace


class ScriptedBackend:
    """Returns scripted probe completions keyed by prefix length; records the
    prefixes it was asked to generate from."""

    def __init__(self, answer_by_prefix_len: dict[int, str], default: str = "0}"):
        self.answers = answer_by_prefix_len
        self.default = default
        self.calls: list[list[int]] = []

    def batch_generate(self, prefixes, max_new_tokens, temperature=0.0,
                       top_p=1.0, seed=None, greedy=False, seeds=None):
        self.calls.append([len(p) for p in prefixes])
        return [GenOut([1], self.answers.get(len(p), self.default)) for p in prefixes]


def test_probe_cache_key_includes_stop_string(tokenizer, tmp_path):
    """Changing stop_string must invalidate the probe cache — the parsed value
    depends on it via extract_probe_answer."""
    from src.config import ProbeConfig
    from src.generation import make_cached_batch_prober

    chain = make_chain(tokenizer, "<think> a b c d </think> \\boxed{42}")
    cache = JsonlCache(tmp_path / "p.jsonl", ["chain_id", "position", "template"])
    t = chain.prompt_token_len + 2

    backend1 = ScriptedBackend({}, default="42}")
    make_cached_batch_prober(
        backend1, tokenizer, ProbeConfig(template="X", stop_string="}"), cache
    )(chain, [t])
    assert len(backend1.calls) == 1

    # same template, different stop_string -> must MISS (re-probe), not reuse
    backend2 = ScriptedBackend({}, default="42}")
    cache2 = JsonlCache(tmp_path / "p.jsonl", ["chain_id", "position", "template"])
    make_cached_batch_prober(
        backend2, tokenizer, ProbeConfig(template="X", stop_string=None), cache2
    )(chain, [t])
    assert len(backend2.calls) == 1  # did not hit the stop_string='}' row


def test_cached_batch_prober_probes_once(tokenizer, tmp_path):
    chain = make_chain(tokenizer, "<think> a b c d e f g h </think> \\boxed{42}")
    probe_cfg = ProbeConfig(template="\n</think>\n\nFinal Answer: \\boxed{")
    template_len = len(tokenizer.encode(probe_cfg.template))
    positions = [chain.prompt_token_len + 2, chain.prompt_token_len + 4]
    backend = ScriptedBackend(
        {p + template_len: f"{i}}}" for i, p in enumerate(positions)}
    )
    cache = JsonlCache(tmp_path / "probes.jsonl", ["chain_id", "position", "template"])
    prober = make_cached_batch_prober(backend, tokenizer, probe_cfg, cache)

    first = prober(chain, positions)
    assert first == {positions[0]: "0", positions[1]: "1"}
    assert len(backend.calls) == 1

    # second call: all hits, no new backend calls, survives cache reload
    cache2 = JsonlCache(tmp_path / "probes.jsonl", ["chain_id", "position", "template"])
    prober2 = make_cached_batch_prober(backend, tokenizer, probe_cfg, cache2)
    second = prober2(chain, positions)
    assert second == first
    assert len(backend.calls) == 1


class SeedRecordingBackend:
    """Fake seed-capable backend: one completion per request, text derived
    from the request's seed so pairing is verifiable."""

    supports_per_request_seeds = True

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls: list[dict] = []

    def batch_generate(self, prefixes, max_new_tokens, temperature=0.0,
                       top_p=1.0, seed=None, greedy=False, seeds=None):
        self.calls.append({"n": len(prefixes), "seeds": list(seeds or [])})
        outs = []
        for s in seeds:
            text = f"<think> attempt {s} </think> \\boxed{{{s}}}"
            outs.append(GenOut(self.tokenizer.encode(text), text))
        return outs


class PerProblemBackend:
    """Fake backend WITHOUT per-request seeds (the HF shape)."""

    supports_per_request_seeds = False

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls: list[dict] = []

    def batch_generate(self, prefixes, max_new_tokens, temperature=0.0,
                       top_p=1.0, seed=None, greedy=False, seeds=None):
        assert seeds is None
        self.calls.append({"n": len(prefixes), "seed": seed})
        text = "<think> some work </think> \\boxed{7}"
        return [GenOut(self.tokenizer.encode(text), text) for _ in prefixes]


def _problems(n):
    return [
        {"problem_id": f"p{i}", "dataset": "test",
         "prompt": f"Problem number {i}?", "gold_answer": "42"}
        for i in range(n)
    ]


def test_generate_chains_cross_problem_single_submission(tokenizer, gen_cfg):
    from src.config import ChainGenConfig
    from src.generation import generate_chains

    gen_cfg.chain = ChainGenConfig(n_chains_per_problem=2, base_seed=100)
    backend = SeedRecordingBackend(tokenizer)
    chains = generate_chains(backend, tokenizer, _problems(3), gen_cfg)

    # one submission covering all (problem, chain) requests
    assert len(backend.calls) == 1
    assert backend.calls[0]["n"] == 6
    # seeds keyed to (problem_index, chain_index), independent of batching
    assert backend.calls[0]["seeds"] == [100, 101, 102, 103, 104, 105]

    assert [c.chain_id for c in chains] == [
        "p0__c0", "p0__c1", "p1__c0", "p1__c1", "p2__c0", "p2__c1"
    ]
    for c in chains:
        assert c.generation_config["seed_scheme"] == "per_request"
        assert f"attempt {c.seed}" in c.text          # output paired to its request
        assert c.trace_token_span is not None
        s, e = c.trace_token_span
        assert c.prompt_token_len <= s < e            # trace parsed, absolute indices


def test_generate_chains_hf_fallback_batches_per_problem(tokenizer, gen_cfg):
    from src.config import ChainGenConfig
    from src.generation import generate_chains

    gen_cfg.chain = ChainGenConfig(n_chains_per_problem=4, base_seed=50)
    backend = PerProblemBackend(tokenizer)
    chains = generate_chains(backend, tokenizer, _problems(3), gen_cfg)

    assert [c["n"] for c in backend.calls] == [4, 4, 4]
    assert [c["seed"] for c in backend.calls] == [50, 51, 52]
    assert len(chains) == 12
    assert all(c.generation_config["seed_scheme"] == "hf_batch" for c in chains)


def test_prober_includes_template_in_prefix(tokenizer, tmp_path):
    """The probe prefix must be chain[:t] + template ids (§6: template fixed
    and identical across all probes)."""
    chain = make_chain(tokenizer, "<think> a b c </think> \\boxed{1}")
    probe_cfg = ProbeConfig(template="\n</think>\n\nFinal Answer: \\boxed{")
    backend = ScriptedBackend({})
    cache = JsonlCache(tmp_path / "p.jsonl", ["chain_id", "position", "template"])
    prober = make_cached_batch_prober(backend, tokenizer, probe_cfg, cache)
    t = chain.prompt_token_len + 1
    prober(chain, [t])
    expected_len = t + len(tokenizer.encode(probe_cfg.template))
    assert backend.calls[0][0] == expected_len
