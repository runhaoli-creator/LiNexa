"""Fast-weight wrapper for a Diffusers ``FeedForward`` module.

Verified runtime structure of ``ff_act`` in
``extern/psi0/src/psi/models/psi0.py:900``::

    ff_act.net = nn.ModuleList([
        net[0]: diffusers...GELU      # Linear(1536, 6144) + GELU
        net[1]: nn.Dropout
        net[2]: nn.Linear(6144, 1536) # final projection
    ])

The wrapper rewrites ``net[2]`` as ``F.linear(h, base_w + ΔW, bias)`` only when
the cache has a delta for this layer. Otherwise it delegates to the wrapped
module unchanged — Phase 0's bit-exact baseline path.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_weight import FastWeightCache

logger = logging.getLogger(__name__)


class FastFFWrapper(nn.Module):
    def __init__(
        self,
        wrapped: nn.Module,
        cache: FastWeightCache,
        layer_id: int,
        log_stats: bool = False,
    ) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.cache = cache
        self.layer_id = layer_id
        self.log_stats = log_stats

        # Resolve net[2] once so the hot path doesn't re-traverse net.
        # The Mode B path (see forward()) hard-codes the recomposition
        # net[0] -> net[1] -> linear(net[2]+ΔW). Pinning len == 3 makes any
        # future Diffusers layout change fail loudly here rather than silently
        # drop modules at index >= 3.
        if not hasattr(wrapped, "net") or len(wrapped.net) != 3:
            raise TypeError(
                f"FastFFWrapper expected a Diffusers FeedForward with .net of "
                f"length 3 (act, dropout, final_proj); got "
                f"{type(wrapped).__name__} with len(net)="
                f"{len(getattr(wrapped, 'net', []))}."
            )
        self._final_proj: nn.Linear = wrapped.net[2]
        if not isinstance(self._final_proj, nn.Linear):
            raise TypeError(
                f"FastFFWrapper expected net[2] to be nn.Linear, got "
                f"{type(self._final_proj).__name__}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.cache.get(self.layer_id)
        if delta is None:
            # Phase 0 fast path: defer to the wrapped module unchanged.
            return self.wrapped(x)

        # Mode B path (Phase 1+): manual recomposition of the FF stack so we
        # can substitute (base_w + ΔW) into the final projection.
        h = self.wrapped.net[0](x)
        h = self.wrapped.net[1](h)
        if self.log_stats:
            logger.debug(
                "linexa.ff[%d] key shape=%s delta_norm=%.4e",
                self.layer_id, tuple(h.shape), float(delta.norm().item()),
            )
        return F.linear(h, self._final_proj.weight + delta, self._final_proj.bias)
