"""SAE-free "correction direction" localization + ablation (§11 robustness).

Answers the reviewer objection that the null is specific to one
vocabulary-selected SAE feature at one layer. We localize the correction
signal with a completely different method — a difference-in-means direction —

    d_L = mean(resid_L | correction-marker tokens)
        - mean(resid_L | position-matched non-marker tokens)

and knock it out during generation by projection ablation:

    x' = x - (x · d̂) d̂          (removes the component along d̂)

This is SAE-independent and sweepable across layers, so a null here across
{15, 19, 23} turns "not this SAE feature at L19" into "not the correction
direction at any of three layers, by a second localization method".
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .config import FeaturesConfig
from .features import _marker_rel_positions, _trace_rel_range
from .io_utils import ChainRecord, EventRecord
from .sae_interventions import PositionMask, PositionTracker


# ----------------------------------------------------------- direction fitting


def _resid_capture_hooks(model, layers: Sequence[int], store: dict):
    """Register forward hooks that stash each target layer's residual output
    (== resid_post_L) for the current forward. Returns removable handles."""
    handles = []
    for L in layers:
        def _mk(LL):
            def _hook(module, args, output):
                store[LL] = output[0] if isinstance(output, tuple) else output
            return _hook
        handles.append(model.model.layers[L].register_forward_hook(_mk(L)))
    return handles


def compute_correction_directions(
    hf_model,
    chains: list[ChainRecord],
    events_by_chain: dict[str, list[EventRecord]],
    layers: Sequence[int],
    cfg: FeaturesConfig,
    max_chains: int = 200,
    seed: int = 0,
    return_raw: bool = False,
) -> dict[int, torch.Tensor] | tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Diff-in-means correction direction per layer, over trace tokens.

    Marker tokens are the correction-event spans; the baseline is
    position-matched non-marker trace tokens (same within-trace decile), so the
    direction captures 'correction' rather than 'deep in the chain'. Returns a
    unit vector per layer on CPU (float32); with return_raw=True also returns
    the raw (pre-normalization) diff-in-means norm per layer — the natural α
    for the sufficiency ADDITION hook (x' = x + α d̂ moves the mean non-marker
    state onto the mean marker state)."""
    import random

    rng = random.Random(seed)
    device = next(hf_model.parameters()).device
    marker_sum = {L: None for L in layers}
    base_sum = {L: None for L in layers}
    n_marker = 0
    n_base = 0

    for chain in chains[:max_chains]:
        if chain.trace_token_span is None:
            continue
        evs = events_by_chain.get(chain.chain_id, [])
        if not evs:
            continue
        rel = _trace_rel_range(chain)
        marker_rel = {
            p for p in _marker_rel_positions(chain, evs) if rel.start <= p < rel.stop
        }
        if not marker_rel:
            continue
        # position-matched baseline sampling (deciles over the trace)
        span = max(len(rel), 1)
        bins: dict[int, list[int]] = {}
        for p in rel:
            if p in marker_rel:
                continue
            b = min(cfg.position_match_bins - 1, (p - rel.start) * cfg.position_match_bins // span)
            bins.setdefault(b, []).append(p)
        base_rel: list[int] = []
        for p in marker_rel:
            b = min(cfg.position_match_bins - 1, (p - rel.start) * cfg.position_match_bins // span)
            pool = bins.get(b) or [q for qs in bins.values() for q in qs]
            if pool:
                base_rel.append(rng.choice(pool))

        store: dict[int, torch.Tensor] = {}
        handles = _resid_capture_hooks(hf_model, layers, store)
        try:
            with torch.no_grad():
                ids = torch.tensor([chain.token_ids], dtype=torch.long, device=device)
                # we only want the captured residuals; skip the LM head entirely
                try:
                    hf_model(ids, use_cache=False, logits_to_keep=1)
                except TypeError:
                    hf_model(ids, use_cache=False, num_logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()

        m_idx = torch.tensor(
            [chain.prompt_token_len + p for p in sorted(marker_rel)], dtype=torch.long
        )
        b_idx = torch.tensor(
            [chain.prompt_token_len + p for p in base_rel], dtype=torch.long
        )
        for L in layers:
            resid = store[L][0].float().cpu()  # [seq, d]
            ms = resid[m_idx].sum(0)
            bs = resid[b_idx].sum(0) if len(b_idx) else torch.zeros_like(ms)
            marker_sum[L] = ms if marker_sum[L] is None else marker_sum[L] + ms
            base_sum[L] = bs if base_sum[L] is None else base_sum[L] + bs
        n_marker += len(m_idx)
        n_base += len(b_idx)

    if n_marker == 0 or n_base == 0:
        raise ValueError(
            f"no direction: {n_marker} marker / {n_base} baseline tokens"
        )
    dirs = {}
    raw_norms = {}
    for L in layers:
        d = marker_sum[L] / n_marker - base_sum[L] / max(n_base, 1)
        raw_norms[L] = float(d.norm())
        dirs[L] = (d / (d.norm() + 1e-8)).float()
    if return_raw:
        return dirs, raw_norms
    return dirs


# ------------------------------------------------------------- ablation hook


class DirectionAblationHook:
    """Projection-ablate a unit direction at masked positions: x' = x - (x·d̂)d̂.

    Same position-keyed, batch-agnostic contract as sae_interventions.ClampHook
    (has a .tracker, callable on [B, seq, d]) so it drops into
    HFHookedGenerator.generate_hooked_batch unchanged."""

    def __init__(self, direction: torch.Tensor, position_mask: PositionMask):
        self.d = (direction / (direction.norm() + 1e-8)).float()
        self.position_mask = position_mask
        self.tracker = PositionTracker()
        self.calls = 0
        self.positions_clamped = 0

    def __call__(self, resid: torch.Tensor, hook=None) -> torch.Tensor:
        seq_len = resid.shape[1]
        positions = self.tracker.take(seq_len).to(resid.device)
        mask = self.position_mask.active(positions)
        self.calls += 1
        if not bool(mask.any()):
            return resid
        self.positions_clamped += int(mask.sum())
        out = resid.clone()
        sel = out[:, mask, :]
        d = self.d.to(dtype=sel.dtype, device=sel.device)
        coef = (sel * d).sum(-1, keepdim=True)  # x·d̂
        out[:, mask, :] = sel - coef * d
        return out


class DirectionAdditionHook:
    """Sufficiency mirror of DirectionAblationHook: ADD α·d̂ at masked
    positions (x' = x + α d̂) — push the residual state TOWARD the correction
    direction instead of removing it, to test whether that induces a
    verbalized correction / rescues a wrong intended answer.

    α defaults to the raw diff-in-means norm (compute_correction_directions
    return_raw=True): at that scale the addition moves the mean non-marker
    state onto the mean marker state. Same position-keyed, batch-agnostic
    contract as ClampHook/DirectionAblationHook."""

    def __init__(self, direction: torch.Tensor, alpha: float, position_mask: PositionMask):
        self.d = (direction / (direction.norm() + 1e-8)).float()
        self.alpha = float(alpha)
        self.position_mask = position_mask
        self.tracker = PositionTracker()
        self.calls = 0
        self.positions_clamped = 0

    def __call__(self, resid: torch.Tensor, hook=None) -> torch.Tensor:
        seq_len = resid.shape[1]
        positions = self.tracker.take(seq_len).to(resid.device)
        mask = self.position_mask.active(positions)
        self.calls += 1
        if not bool(mask.any()):
            return resid
        self.positions_clamped += int(mask.sum())
        out = resid.clone()
        sel = out[:, mask, :]
        d = self.d.to(dtype=sel.dtype, device=sel.device)
        out[:, mask, :] = sel + self.alpha * d
        return out
