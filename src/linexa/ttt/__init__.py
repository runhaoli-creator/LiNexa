"""LiNexa TTT package.

Only torch-free symbols are eagerly re-exported here. ``FastFFWrapper`` and
``FastWeightCache`` import torch and should be imported from their submodules
directly (``from linexa.ttt.fast_ff_wrapper import FastFFWrapper``) so that
config-only tooling can ``import linexa.ttt.config`` without pulling in torch.
"""
from .config import LinexaConfig

__all__ = ["LinexaConfig"]
