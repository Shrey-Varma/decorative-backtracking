import torch

from src.config import FeaturesConfig
from src.features import (
    CandidateFeature,
    CorpusStats,
    answer_predictive_score,
    compute_corpus_stats,
    event_window_activations,
    select_candidates,
    select_control_feature,
)
from src.io_utils import EventRecord
from tests.conftest import make_chain

D_SAE = 12
MARKER_FEATURE = 3


def _synthetic_encode_fn(chain, marker_rel_positions, noise_seed=0):
    """Feature MARKER_FEATURE fires exactly on marker tokens; feature 7 is
    position-independent noise; everything else silent."""
    g = torch.Generator().manual_seed(noise_seed)
    n = len(chain.token_ids)
    acts = torch.zeros(n, D_SAE)
    acts[:, 7] = torch.rand(n, generator=g) * 0.5
    for p in marker_rel_positions:
        acts[chain.prompt_token_len + p, MARKER_FEATURE] = 2.0
    table = {tuple(chain.token_ids): acts}

    def encode_fn(token_ids):
        return table[tuple(token_ids)][: len(token_ids)]

    return encode_fn


def _chain_with_event(tokenizer):
    filler_a = "step " * 30
    filler_b = "more " * 30
    chain = make_chain(
        tokenizer,
        f"<think> {filler_a} wait, recheck the sum now. {filler_b} </think> \\boxed{{42}}",
    )
    from src.config import EventsConfig
    from src.events import detect_events
    from tests.conftest import DEFAULT_LEXICON

    events = detect_events(chain, EventsConfig(lexicon=DEFAULT_LEXICON))
    assert len(events) == 1
    return chain, events


def test_vocab_score_ranks_marker_feature_first(tokenizer):
    chain, events = _chain_with_event(tokenizer)
    ev = events[0]
    marker_rel = list(
        range(ev.onset - chain.prompt_token_len, ev.end + 1 - chain.prompt_token_len)
    )
    encode_fn = _synthetic_encode_fn(chain, marker_rel)
    cfg = FeaturesConfig(n_candidate_features=2)
    stats = compute_corpus_stats([chain], {chain.chain_id: events}, encode_fn, cfg)

    assert int(stats.vocab_score.argmax()) == MARKER_FEATURE
    assert stats.n_marker_tokens == len(marker_rel)
    assert stats.n_nonmarker_tokens > 0
    # corpus mean over trace tokens only
    s, e = chain.trace_token_span
    assert stats.corpus_mean[MARKER_FEATURE] > 0
    assert float(stats.corpus_mean[5]) == 0.0

    cands = select_candidates(stats, None, cfg)
    assert cands[0].feature_id == MARKER_FEATURE
    assert cands[0].vocab_score >= cands[1].vocab_score


def test_combined_pass_matches_separate_calls(tokenizer):
    """compute_corpus_stats_and_windows shares one encode per chain; its
    windows must equal the standalone event_window_activations result."""
    from src.features import compute_corpus_stats_and_windows, event_window_activations

    chain, events = _chain_with_event(tokenizer)
    ev = events[0]
    marker_rel = [ev.onset - chain.prompt_token_len]
    encode_fn = _synthetic_encode_fn(chain, marker_rel)
    cfg = FeaturesConfig()
    stats, windows = compute_corpus_stats_and_windows(
        [chain], {chain.chain_id: events}, encode_fn, cfg
    )
    separate = event_window_activations(chain, events, encode_fn, cfg)
    assert set(windows) == set(separate) == {ev.event_id}
    assert torch.equal(windows[ev.event_id], separate[ev.event_id])
    assert int(stats.vocab_score.argmax()) == MARKER_FEATURE


def test_vocab_score_raises_without_marker_tokens(tokenizer):
    """No marker tokens => all-zero vocab_score would make select_candidates
    return arbitrary features 0..k-1; fail loud instead."""
    import pytest

    from src.features import compute_corpus_stats

    chain = make_chain(tokenizer, "<think> just plain reasoning no markers 42 </think> \\boxed{42}")
    encode_fn = _synthetic_encode_fn(chain, [])
    with pytest.raises(ValueError, match="vocab score undefined"):
        compute_corpus_stats([chain], {chain.chain_id: []}, encode_fn, FeaturesConfig())


def test_collect_top_examples_trace_only(tokenizer):
    """Dashboard examples must be trace-internal — a feature firing hardest in
    the post-</think> answer region must not dominate."""
    from src.config import EventsConfig
    from src.events import detect_events
    from src.features import collect_top_examples
    from tests.conftest import DEFAULT_LEXICON

    chain = make_chain(
        tokenizer,
        "<think> " + "step " * 20 + " wait, recheck. </think> the answer is \\boxed{42}",
    )
    # feature 3 fires hardest AFTER </think> (in the answer region)
    n = len(chain.token_ids)
    acts = torch.zeros(n, D_SAE)
    trace_end_rel = chain.trace_token_span[1] - chain.prompt_token_len
    acts[chain.prompt_token_len + trace_end_rel + 1:, 3] = 9.0  # answer region
    s, e = chain.trace_token_span
    acts[s + 2, 3] = 2.0  # a smaller in-trace activation
    table = {tuple(chain.token_ids): acts}
    encode_fn = lambda ids: table[tuple(ids)][: len(ids)]

    examples = collect_top_examples([chain], encode_fn, [3], FeaturesConfig())
    assert examples[3]  # found the in-trace activation
    # every example is inside the trace, none from the answer region (act 9.0)
    assert all(ex["activation"] <= 2.0 for ex in examples[3])


def test_event_window_activations(tokenizer):
    chain, events = _chain_with_event(tokenizer)
    ev = events[0]
    marker_rel = [ev.onset - chain.prompt_token_len]
    encode_fn = _synthetic_encode_fn(chain, marker_rel)
    acts = event_window_activations(chain, events, encode_fn, FeaturesConfig())
    assert set(acts) == {ev.event_id}
    assert float(acts[ev.event_id][MARKER_FEATURE]) == 2.0


def test_answer_predictive_auroc_perfect_and_inverse():
    window_acts = {
        "e1": torch.tensor([5.0, 0.0]),
        "e2": torch.tensor([4.0, 1.0]),
        "e3": torch.tensor([1.0, 4.0]),
        "e4": torch.tensor([0.0, 5.0]),
    }
    flips = {"e1": True, "e2": True, "e3": False, "e4": False}
    auc = answer_predictive_score(window_acts, flips)
    assert float(auc[0]) == 1.0   # perfectly predictive
    assert float(auc[1]) == 0.0   # perfectly anti-predictive


def test_answer_predictive_auroc_ties_give_half():
    window_acts = {f"e{i}": torch.tensor([1.0]) for i in range(4)}
    flips = {"e0": True, "e1": True, "e2": False, "e3": False}
    auc = answer_predictive_score(window_acts, flips)
    assert abs(float(auc[0]) - 0.5) < 1e-9


def test_midranks_match_scipy_with_ties():
    import numpy as np
    from scipy.stats import rankdata

    from src.features import _midranks

    g = torch.Generator().manual_seed(5)
    # sparse-ish data: lots of exact zeros, like SAE activations
    X = torch.randn(40, 7, generator=g)
    X[torch.rand(40, 7, generator=g) < 0.6] = 0.0
    ours = _midranks(X.double()).numpy()
    ref = np.stack(
        [rankdata(X[:, j].numpy(), method="average") for j in range(X.shape[1])], axis=1
    )
    assert np.allclose(ours, ref)


def test_auroc_matches_sklearn_style_reference():
    """Cross-check the vectorized Mann-Whitney AUROC against a direct
    pairwise-comparison computation on tied, sparse data."""
    g = torch.Generator().manual_seed(9)
    n = 30
    X = torch.randn(n, 5, generator=g)
    X[torch.rand(n, 5, generator=g) < 0.5] = 0.0
    y = torch.rand(n, generator=g) < 0.4
    if not y.any() or y.all():
        y[0], y[1] = True, False
    window_acts = {f"e{i}": X[i] for i in range(n)}
    flips = {f"e{i}": bool(y[i]) for i in range(n)}
    auc = answer_predictive_score(window_acts, flips)
    pos, neg = X[y], X[~y]
    for j in range(5):
        gt = (pos[:, j].unsqueeze(1) > neg[:, j].unsqueeze(0)).double().sum()
        eq = (pos[:, j].unsqueeze(1) == neg[:, j].unsqueeze(0)).double().sum()
        ref = (gt + 0.5 * eq) / (pos.shape[0] * neg.shape[0])
        assert abs(float(auc[j]) - float(ref)) < 1e-6  # score is fp32


def test_answer_predictive_nan_when_one_class():
    window_acts = {"e1": torch.tensor([1.0]), "e2": torch.tensor([2.0])}
    auc = answer_predictive_score(window_acts, {"e1": True, "e2": True})
    assert torch.isnan(auc[0])


def test_control_feature_excludes_candidates_and_matches_stats():
    stats = CorpusStats(
        corpus_mean=torch.tensor([0.5, 0.49, 0.01, 3.0, 0.0, 0.51]),
        activation_freq=torch.tensor([0.1, 0.099, 0.9, 0.5, 0.0, 0.101]),
        vocab_score=torch.tensor([9.0, 0.1, 0.2, 0.3, 0.0, 0.15]),
        marker_mean=torch.zeros(6),
        nonmarker_mean=torch.zeros(6),
        n_marker_tokens=10,
        n_nonmarker_tokens=10,
        n_chains=1,
    )
    primary = CandidateFeature(0, 9.0, None, corpus_mean=0.5, activation_freq=0.1)
    ctrl = select_control_feature(stats, [primary], "matched", exclude_top_vocab=1)
    # nearest in (freq, mean) to feature 0 among allowed: features 1 or 5;
    # feature 4 (dead) and feature 0 (candidate/top-vocab) excluded
    assert ctrl in (1, 5)

    ctrl_rand = select_control_feature(stats, [primary], "random", seed=0, exclude_top_vocab=1)
    assert ctrl_rand not in (0, 4)
