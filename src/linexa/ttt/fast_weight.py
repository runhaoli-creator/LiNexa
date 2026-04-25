"""Episode-local fast-weight cache.

Holds per-layer ``ΔW`` deltas added to the base ``net[2].weight`` of a Diffusers
``FeedForward`` (see docs/plan.md §5). Base parameters are never mutated. Reset
clears the cache and restores baseline behavior.

Phase 0 never invokes ``commit`` — the cache stays empty for the whole run, so
the wrapper's fast path is bit-exact with the original module.
"""
from __future__ import annotations

from typing import Optional

import torch


class FastWeightCache:
    def __init__(self) -> None:
        self._deltas: dict[int, torch.Tensor] = {}
        self._reset_count: int = 0

    def reset(self) -> None:
        self._deltas.clear()
        self._reset_count += 1

    @property
    def reset_count(self) -> int:
        return self._reset_count

    def get(self, layer_id: int) -> Optional[torch.Tensor]:
        """Return ΔW for ``layer_id`` or None if no write has happened."""
        return self._deltas.get(layer_id)

    def commit(
        self,
        layer_id: int,
        h: torch.Tensor,
        t: torch.Tensor,
        *,
        base_w: torch.Tensor,
        lr: float,
        decay: float = 0.0,
        clip: float = 0.0,
    ) -> None:
        """Apply an outcome-aligned write. Phase 0 never calls this.

        The intended Mode B rule (analogue of In-Place TTT
        ``modeling_qwen3.py:117-152``) is::

            ΔW_new = (1 - decay) * ΔW_old + lr * contract(h, t)
            ΔW_new = clip_by_norm(ΔW_new, clip)

        Implementation deferred until the outcome target ``Φ(...)`` and the
        write rule are pinned (docs/plan.md §7 TODO).
        """
        raise NotImplementedError(
            "FastWeightCache.commit is not implemented in Phase 0. "
            "Phase 0 must run with ΔW = 0 (no commit calls)."
        )

    def stats(self) -> dict[int, dict[str, float]]:
        out = {}
        for layer_id, dw in self._deltas.items():
            out[layer_id] = {
                "norm": float(dw.norm().item()),
                "max_abs": float(dw.abs().max().item()),
                "shape": tuple(dw.shape),
            }
        return out
