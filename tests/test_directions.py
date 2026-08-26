import torch

from src.directions import DirectionAblationHook, DirectionAdditionHook
from src.sae_interventions import PositionMask


def test_projection_ablation_removes_component_along_direction():
    torch.manual_seed(0)
    d = torch.randn(8)
    hook = DirectionAblationHook(d, PositionMask("event_onward", t0=0))
    x = torch.randn(2, 4, 8)
    out = hook(x)
    dhat = d / d.norm()
    # component along d̂ must be ~0 after ablation, everything else preserved
    comp = (out * dhat).sum(-1)
    assert torch.allclose(comp, torch.zeros_like(comp), atol=1e-5)
    # orthogonal complement unchanged: out == x - (x·d̂)d̂
    expected = x - (x * dhat).sum(-1, keepdim=True) * dhat
    assert torch.allclose(out, expected, atol=1e-5)


def test_direction_hook_respects_position_mask():
    d = torch.randn(8)
    hook = DirectionAblationHook(d, PositionMask("event_onward", t0=3))
    hook.tracker.reset()
    # prefill positions 0..2 (< t0) untouched; the hook advances the tracker
    prefill = torch.randn(1, 3, 8)
    assert torch.equal(hook(prefill), prefill)
    assert hook.positions_clamped == 0
    # next single-token step is position 3 -> clamped
    step = torch.randn(1, 1, 8)
    out = hook(step)
    assert not torch.equal(out, step)
    assert hook.positions_clamped == 1


def test_zero_direction_is_safe():
    hook = DirectionAblationHook(torch.zeros(8), PositionMask("event_onward", t0=0))
    x = torch.randn(1, 2, 8)
    out = hook(x)  # normalization guards against div-by-zero
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_addition_hook_adds_alpha_along_unit_direction():
    torch.manual_seed(1)
    d = torch.randn(8) * 3.0  # non-unit input; hook must normalize
    alpha = 2.5
    hook = DirectionAdditionHook(d, alpha, PositionMask("event_onward", t0=0))
    x = torch.randn(2, 4, 8)
    out = hook(x)
    dhat = d / d.norm()
    assert torch.allclose(out - x, alpha * dhat.expand_as(x), atol=1e-5)
    # projection along d̂ increased by exactly alpha
    assert torch.allclose(
        (out * dhat).sum(-1), (x * dhat).sum(-1) + alpha, atol=1e-5
    )


def test_addition_hook_respects_position_mask():
    d = torch.randn(8)
    hook = DirectionAdditionHook(d, 1.0, PositionMask("event_only", t0=3, t1=4))
    hook.tracker.reset()
    prefill = torch.randn(1, 3, 8)  # positions 0..2: untouched
    assert torch.equal(hook(prefill), prefill)
    step = torch.randn(1, 1, 8)     # position 3: modified
    assert not torch.equal(hook(step), step)
    step2 = torch.randn(1, 1, 8)    # position 4: modified
    assert not torch.equal(hook(step2), step2)
    step3 = torch.randn(1, 1, 8)    # position 5: past t1 -> untouched
    assert torch.equal(hook(step3), step3)
    assert hook.positions_clamped == 2
