"""Error-preserving SAE feature ablation + clamp hooks (build order step 2, §7).

The invariant everything rests on (§7.1): we never replace the activation with
the SAE reconstruction. We move *only* the ablated features' contribution:

    f     = encode(x)
    x_hat = decode(f)
    err   = x - x_hat            # preserved untouched
    f_mod[S] = target
    x'    = decode(f_mod) + err

With S empty (or target == f[S]) this is exactly the identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import torch


class SAELike(Protocol):
    """What we require of an SAE: sae_lens SAEs satisfy this, and tests use a
    toy implementation."""

    def encode(self, x: torch.Tensor) -> torch.Tensor: ...
    def decode(self, f: torch.Tensor) -> torch.Tensor: ...


# ------------------------------------------------------------------------ ablation


def error_preserving_ablate(
    x: torch.Tensor,
    sae: SAELike,
    feature_ids: Sequence[int],
    target_values: torch.Tensor | float,
) -> torch.Tensor:
    """Ablate feature set S at every position of `x` ([..., d_model]).

    target_values: scalar, or tensor broadcastable to f[..., S]
    (per-feature vector of shape [len(S)] is the common case).
    """
    sae_dtype = _sae_dtype(sae, x)
    x_in = x.to(sae_dtype)

    f = sae.encode(x_in)
    x_hat = sae.decode(f)
    err = x_in - x_hat

    f_mod = f.clone()
    idx = torch.as_tensor(list(feature_ids), dtype=torch.long, device=f.device)
    if isinstance(target_values, torch.Tensor):
        f_mod[..., idx] = target_values.to(dtype=f.dtype, device=f.device)
    else:
        f_mod[..., idx] = float(target_values)

    return (sae.decode(f_mod) + err).to(x.dtype)


def _sae_dtype(sae: Any, fallback_like: torch.Tensor) -> torch.dtype:
    try:
        return next(sae.parameters()).dtype
    except (AttributeError, StopIteration, TypeError):
        return fallback_like.dtype


def make_target_values(
    mode: str,
    feature_ids: Sequence[int],
    corpus_means: Optional[torch.Tensor] = None,       # [d_sae]
    counterfactual_values: Optional[torch.Tensor] = None,  # [len(feature_ids)]
) -> torch.Tensor | float:
    """Resolve `ablation_mode` (§7.1) into concrete per-feature targets."""
    if mode == "zero":
        return 0.0
    if mode == "mean":
        if corpus_means is None:
            raise ValueError("mean ablation requires corpus_means")
        idx = torch.as_tensor(list(feature_ids), dtype=torch.long)
        return corpus_means[idx]
    if mode == "counterfactual":
        if counterfactual_values is None:
            raise ValueError("counterfactual ablation requires matched-prefix values")
        return torch.as_tensor(counterfactual_values)
    raise ValueError(f"unknown ablation mode: {mode}")


# ------------------------------------------------------------------ position masks


@dataclass
class PositionMask:
    """Where the clamp applies, in *absolute* token positions (§7.2).

    - event_only:   [t0, t1]      (t1 inclusive; original-span positions)
    - event_onward: [t0, inf)     (primary mode — prevents downstream re-firing;
                                   t0=0 clamps everywhere)
    """

    mode: str
    t0: int
    t1: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in ("event_only", "event_onward"):
            raise ValueError(f"unknown position mask mode: {self.mode}")
        if self.mode == "event_only" and self.t1 is None:
            raise ValueError("event_only mask requires t1")

    def active(self, positions: torch.Tensor) -> torch.Tensor:
        if self.mode == "event_onward":
            return positions >= self.t0
        return (positions >= self.t0) & (positions <= self.t1)


@dataclass
class PositionTracker:
    """Tracks the absolute position of the next token entering the model.

    The generation loop resets it to 0 before the prefill forward; the clamp
    hook advances it on every forward, so KV-cached single-token steps see
    their true absolute position.
    """

    pos: int = 0

    def take(self, seq_len: int) -> torch.Tensor:
        out = torch.arange(self.pos, self.pos + seq_len)
        self.pos += seq_len
        return out

    def reset(self) -> None:
        self.pos = 0


# ------------------------------------------------------------------------ the hook


@dataclass
class ClampHook:
    """Forward hook for the SAE's hook_name: applies error-preserving ablation
    at every decoded position selected by the position mask, for the duration
    of a continuation (§7.2). Register as (hook_name, hook) in TL fwd_hooks.
    """

    sae: SAELike
    feature_ids: Sequence[int]
    target_values: torch.Tensor | float
    position_mask: PositionMask
    tracker: PositionTracker = field(default_factory=PositionTracker)
    calls: int = 0                # instrumentation for tests/sanity
    positions_clamped: int = 0

    def __call__(self, resid: torch.Tensor, hook: Any = None) -> torch.Tensor:
        # resid: [batch, seq, d_model]
        seq_len = resid.shape[1]
        positions = self.tracker.take(seq_len).to(resid.device)
        mask = self.position_mask.active(positions)
        self.calls += 1
        if not bool(mask.any()):
            return resid
        self.positions_clamped += int(mask.sum())
        out = resid.clone()
        out[:, mask, :] = error_preserving_ablate(
            resid[:, mask, :], self.sae, self.feature_ids, self.target_values
        )
        return out


def make_clamp_hook(
    sae: SAELike,
    feature_ids: Sequence[int],
    ablation_mode: str,
    position_mask: PositionMask,
    corpus_means: Optional[torch.Tensor] = None,
    counterfactual_values: Optional[torch.Tensor] = None,
) -> ClampHook:
    targets = make_target_values(
        ablation_mode, feature_ids,
        corpus_means=corpus_means,
        counterfactual_values=counterfactual_values,
    )
    return ClampHook(
        sae=sae,
        feature_ids=feature_ids,
        target_values=targets,
        position_mask=position_mask,
    )
