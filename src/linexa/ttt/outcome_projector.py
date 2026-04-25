"""Outcome-target projection ``Φ`` (placeholder for Phase 1+).

``Φ`` maps an outcome signal (proprio delta, contact, progress, ...) into the
key space of the wrapped feed-forward (post-GELU activation, dim 6144 for the
default Psi0 action expert). Phase 0 never instantiates this; it is here so
that Phase 1 has a clear, file-local home for the design choice.

See ``docs/idea.md`` §6 and ``docs/plan.md`` §7 for context.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IdentityPhi(nn.Module):
    """Zero-pad / truncate a proprio vector into the action-expert hidden dim.

    Useful as a confound-removed sanity baseline: the write target is the raw
    proprio signal, modulo dimension. Should be considered a plumbing test, not
    a learned method (docs/plan.md §7 caveat).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.out_dim:
            return x
        if x.shape[-1] > self.out_dim:
            return x[..., : self.out_dim]
        pad = torch.zeros(*x.shape[:-1], self.out_dim - x.shape[-1], dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)


class LinearPhi(nn.Module):
    """Single linear ``R^in_dim -> R^out_dim``. Frozen-random by default."""

    def __init__(self, in_dim: int, out_dim: int, freeze: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        if freeze:
            for p in self.proj.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
