"""Build-order step 2 unit tests (§12): clamping with target=f[i] is a no-op,
zero removes ONLY that feature's contribution, error term always preserved."""

import torch

from src.sae_interventions import (
    ClampHook,
    PositionMask,
    error_preserving_ablate,
    make_clamp_hook,
    make_target_values,
)


def _acts(toy_sae, n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n, toy_sae.W_enc.shape[0], generator=g)


def test_noop_when_target_equals_current_activation(toy_sae):
    x = _acts(toy_sae)
    f = toy_sae.encode(x)
    fid = int(f.abs().sum(dim=(0, 1)).argmax())  # a feature that actually fires
    out = error_preserving_ablate(x, toy_sae, [fid], f[..., [fid]])
    assert torch.allclose(out, x, atol=1e-5)


def test_zero_ablation_removes_only_that_features_contribution(toy_sae):
    x = _acts(toy_sae)
    f = toy_sae.encode(x)
    fid = int(f.abs().sum(dim=(0, 1)).argmax())
    out = error_preserving_ablate(x, toy_sae, [fid], 0.0)
    # linear decoder: removing feature i subtracts exactly f_i * W_dec[i]
    expected = x - f[..., fid: fid + 1] * toy_sae.W_dec[fid]
    assert torch.allclose(out, expected, atol=1e-5)
    assert not torch.allclose(out, x, atol=1e-4)  # it did something


def test_error_term_is_preserved_not_reconstruction(toy_sae):
    """The naive intervention (decode(encode(x)) with feature zeroed) differs
    from ours by exactly the reconstruction error."""
    x = _acts(toy_sae)
    f = toy_sae.encode(x)
    fid = 3
    f_mod = f.clone()
    f_mod[..., fid] = 0
    naive = toy_sae.decode(f_mod)
    ours = error_preserving_ablate(x, toy_sae, [fid], 0.0)
    err = x - toy_sae.decode(f)
    assert torch.allclose(ours - naive, err, atol=1e-5)


def test_multi_feature_ablation(toy_sae):
    x = _acts(toy_sae)
    f = toy_sae.encode(x)
    fids = [1, 4, 7]
    out = error_preserving_ablate(x, toy_sae, fids, 0.0)
    expected = x.clone()
    for fid in fids:
        expected = expected - f[..., fid: fid + 1] * toy_sae.W_dec[fid]
    assert torch.allclose(out, expected, atol=1e-5)


def test_target_values_modes():
    means = torch.arange(16, dtype=torch.float32)
    assert make_target_values("zero", [3, 5]) == 0.0
    mean_t = make_target_values("mean", [3, 5], corpus_means=means)
    assert torch.equal(mean_t, torch.tensor([3.0, 5.0]))
    cf = make_target_values("counterfactual", [3, 5], counterfactual_values=torch.tensor([1.0, 2.0]))
    assert torch.equal(cf, torch.tensor([1.0, 2.0]))


def test_mean_ablation_sets_feature_to_mean(toy_sae):
    x = _acts(toy_sae)
    means = torch.full((16,), 0.5)
    out = error_preserving_ablate(x, toy_sae, [2], make_target_values("mean", [2], means))
    f = toy_sae.encode(x)
    expected = x + (0.5 - f[..., 2: 2 + 1]) * toy_sae.W_dec[2]
    assert torch.allclose(out, expected, atol=1e-5)


# ------------------------------------------------------------------ position masks


def test_position_mask_event_only():
    m = PositionMask("event_only", t0=10, t1=12)
    active = m.active(torch.arange(8, 16))
    assert active.tolist() == [False, False, True, True, True, False, False, False]


def test_position_mask_event_onward():
    m = PositionMask("event_onward", t0=10)
    active = m.active(torch.arange(8, 14))
    assert active.tolist() == [False, False, True, True, True, True]


def test_position_mask_requires_t1_for_event_only():
    import pytest

    with pytest.raises(ValueError):
        PositionMask("event_only", t0=5)


# ----------------------------------------------------------------- clamp hook


def _simulate_generation(hook, d_model, prefill_len, n_steps, seed=2):
    """Prefill [1, prefill_len, d] then n single-token steps, like the TL loop."""
    g = torch.Generator().manual_seed(seed)
    hook.tracker.reset()
    outs = []
    x = torch.randn(1, prefill_len, d_model, generator=g)
    outs.append((x, hook(x)))
    for _ in range(n_steps):
        x = torch.randn(1, 1, d_model, generator=g)
        outs.append((x, hook(x)))
    return outs


def test_clamp_hook_event_onward_only_touches_generated_positions(toy_sae):
    t0 = 5
    hook = make_clamp_hook(
        toy_sae, [3], "zero", PositionMask("event_onward", t0=t0)
    )
    outs = _simulate_generation(hook, 8, prefill_len=t0, n_steps=4)
    prefill_in, prefill_out = outs[0]
    assert torch.equal(prefill_in, prefill_out)  # positions < t0 untouched
    for x_in, x_out in outs[1:]:
        f = toy_sae.encode(x_in)
        if float(f[..., 3].abs().max()) > 1e-6:
            assert not torch.allclose(x_in, x_out, atol=1e-6)
    assert hook.positions_clamped == 4


def test_clamp_hook_event_only_window(toy_sae):
    t0, t1 = 5, 6
    hook = make_clamp_hook(toy_sae, [3], "zero", PositionMask("event_only", t0=t0, t1=t1))
    outs = _simulate_generation(hook, 8, prefill_len=t0, n_steps=5)
    # steps at absolute positions 5,6 clamped; 7,8,9 not
    assert hook.positions_clamped == 2
    for i, (x_in, x_out) in enumerate(outs[1:], start=t0):
        if i > t1:
            assert torch.equal(x_in, x_out)


def test_clamp_hook_prefill_covering_event_gets_clamped(toy_sae):
    """Control-position runs prefill past their own t0? No — but event_only
    masks inside a longer prefill must clamp mid-prefill positions."""
    hook = make_clamp_hook(toy_sae, [3], "zero", PositionMask("event_only", t0=2, t1=3))
    x = torch.randn(1, 6, 8, generator=torch.Generator().manual_seed(3))
    out = hook(x)
    assert torch.equal(x[:, :2], out[:, :2])
    assert torch.equal(x[:, 4:], out[:, 4:])
    assert hook.positions_clamped == 2


def test_clamp_hook_noop_mask_returns_input_object(toy_sae):
    hook = make_clamp_hook(toy_sae, [3], "zero", PositionMask("event_onward", t0=100))
    x = torch.randn(1, 5, 8)
    assert hook(x) is x  # zero-copy when inactive
    assert hook.calls == 1 and hook.positions_clamped == 0


def test_clamp_hook_dtype_roundtrip(toy_sae):
    """Model activations may be bf16 while the SAE is fp32; output must come
    back in the input dtype."""
    hook = make_clamp_hook(toy_sae, [3], "zero", PositionMask("event_onward", t0=0))
    x = torch.randn(1, 4, 8).to(torch.bfloat16)
    out = hook(x)
    assert out.dtype == torch.bfloat16
