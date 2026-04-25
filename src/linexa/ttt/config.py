"""LiNexa TTT configuration.

Phase 0 reads configuration from environment variables so the docker entrypoint
can toggle behavior without touching application code. YAML/Hydra integration
is deferred until the API stabilises (see plan.md §4).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else default


def _env_int_list(name: str) -> list[int]:
    """Parse a comma-separated list of layer indices. Empty -> [] (= all blocks)."""
    val = os.environ.get(name, "")
    if not val.strip():
        return []
    return [int(x) for x in val.split(",") if x.strip()]


@dataclass
class LinexaConfig:
    enabled: bool = False
    layer_indices: list[int] = field(default_factory=list)
    write_scale: float = 0.0
    decay: float = 0.0
    clip: float = 0.0
    log_stats: bool = False

    @classmethod
    def from_env(cls) -> "LinexaConfig":
        return cls(
            enabled=_env_bool("LINEXA_TTT_ENABLED", False),
            layer_indices=_env_int_list("LINEXA_TTT_LAYERS"),
            write_scale=_env_float("LINEXA_TTT_WRITE_SCALE", 0.0),
            decay=_env_float("LINEXA_TTT_DECAY", 0.0),
            clip=_env_float("LINEXA_TTT_CLIP", 0.0),
            log_stats=_env_bool("LINEXA_TTT_LOG_STATS", False),
        )
