"""Install / reset / commit hooks for a loaded Psi0 model.

The adapter discovers all ``VLATransformerBlock`` instances inside the given
model via ``isinstance`` (no path hardcoding), wraps each block's ``ff_act``
with :class:`FastFFWrapper`, and attaches the shared :class:`FastWeightCache`
to the model object as ``model._linexa_cache``.

Safety guarantees of ``install``:

- base parameter tensors are not mutated;
- parameter count is unchanged (the original ``ff_act`` modules live inside
  the wrappers, no parameters are added);
- clearing the cache via :func:`reset` restores baseline computation
  (``FastFFWrapper`` delegates directly to the wrapped module on the empty-cache
  fast path).

Not guaranteed:

- ``state_dict()`` *key names* shift while wrappers are installed
  (``block.ff_act.net.0.proj.weight`` becomes
  ``block.ff_act.wrapped.net.0.proj.weight``). Tensor content matches
  one-for-one. Call :func:`uninstall` before any ``state_dict()``-based
  comparison or save.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch.nn as nn

from linexa.ttt.config import LinexaConfig
from linexa.ttt.fast_ff_wrapper import FastFFWrapper
from linexa.ttt.fast_weight import FastWeightCache

logger = logging.getLogger(__name__)


def _find_blocks(model: nn.Module) -> list[nn.Module]:
    # Late import: psi is only available inside the psi0 server container.
    from psi.models.psi0 import VLATransformerBlock

    return [m for m in model.modules() if isinstance(m, VLATransformerBlock)]


def _check_phase0_safe(cfg: LinexaConfig) -> None:
    """Phase 0 supports only the no-op wrapper; writes are not implemented.

    Fail loudly if the user sets a write knob, otherwise silently-zero writes
    can be misread as broken experiments rather than the intentional Phase 0
    scaffold.
    """
    nonzero = [
        (k, v) for k, v in (
            ("LINEXA_TTT_WRITE_SCALE", cfg.write_scale),
            ("LINEXA_TTT_DECAY", cfg.decay),
            ("LINEXA_TTT_CLIP", cfg.clip),
        ) if float(v) != 0.0
    ]
    if nonzero:
        details = ", ".join(f"{k}={v}" for k, v in nonzero)
        raise RuntimeError(
            f"linexa: nonzero write knobs set ({details}) but Mode B is not "
            f"implemented. Phase 0 supports only LINEXA_TTT_ENABLED=1 with all "
            f"write knobs at 0. See plan.md §7."
        )


def install(model: nn.Module, cfg: LinexaConfig) -> Optional[FastWeightCache]:
    """Install fast-weight wrappers on selected action-expert blocks.

    Returns the attached cache, or ``None`` if disabled.
    """
    if not cfg.enabled:
        logger.info("linexa: TTT disabled, no wrappers installed")
        return None

    _check_phase0_safe(cfg)

    if getattr(model, "_linexa_cache", None) is not None:
        logger.warning("linexa: install() called twice; ignoring second call")
        return model._linexa_cache

    blocks = _find_blocks(model)
    if not blocks:
        raise RuntimeError(
            "linexa: no VLATransformerBlock found on model; check the model "
            "type or enable late-binding install."
        )

    indices = cfg.layer_indices or list(range(len(blocks)))
    cache = FastWeightCache()
    wrapped: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= len(blocks):
            raise IndexError(
                f"linexa: layer index {idx} out of range (have {len(blocks)} blocks)"
            )
        block = blocks[idx]
        original_ff = block.ff_act
        if isinstance(original_ff, FastFFWrapper):
            logger.warning("linexa: block[%d].ff_act already wrapped; skipping", idx)
            continue
        block.ff_act = FastFFWrapper(
            wrapped=original_ff,
            cache=cache,
            layer_id=idx,
            log_stats=cfg.log_stats,
        )
        wrapped.append(idx)

    model._linexa_cache = cache  # type: ignore[attr-defined]
    model._linexa_blocks = blocks  # type: ignore[attr-defined]
    model._linexa_cfg = cfg  # type: ignore[attr-defined]

    logger.info(
        "linexa: installed FastFFWrapper on %d / %d action-expert blocks (idx=%s)",
        len(wrapped), len(blocks), wrapped,
    )
    return cache


def uninstall(model: nn.Module) -> None:
    """Restore the original ``ff_act`` modules on every wrapped block."""
    blocks = getattr(model, "_linexa_blocks", None)
    if blocks is None:
        return
    for block in blocks:
        ff = getattr(block, "ff_act", None)
        if isinstance(ff, FastFFWrapper):
            block.ff_act = ff.wrapped
    for attr in ("_linexa_cache", "_linexa_blocks", "_linexa_cfg"):
        if hasattr(model, attr):
            delattr(model, attr)
    logger.info("linexa: uninstalled wrappers")


def reset(model: nn.Module) -> None:
    """Clear the per-episode fast-weight cache."""
    cache: Optional[FastWeightCache] = getattr(model, "_linexa_cache", None)
    if cache is None:
        return
    cache.reset()
    logger.info("linexa: reset cache (count=%d)", cache.reset_count)


def commit_outcome(model: nn.Module, *args, **kwargs) -> None:
    """Outcome write — explicit external call (Phase 1+).

    Phase 0 never invokes this. The signature is intentionally generic so the
    Phase 1 design (proprio source, ``Φ`` choice, write rule) is not pre-empted
    here. See plan.md §7 TODO.
    """
    raise NotImplementedError(
        "linexa.adapters.psi0.commit_outcome is not implemented. "
        "Phase 0 must run with no commit calls."
    )
