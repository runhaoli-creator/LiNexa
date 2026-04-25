"""Config parsing tests — torch-free so they run in any environment."""
from __future__ import annotations

import os
from contextlib import contextmanager

from linexa.ttt.config import LinexaConfig


@contextmanager
def _env(**overrides: str):
    """Temporarily set env vars; restore on exit."""
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_defaults_are_safe():
    # Clear any inherited values so the test is hermetic.
    keys = [
        "LINEXA_TTT_ENABLED", "LINEXA_TTT_LAYERS", "LINEXA_TTT_WRITE_SCALE",
        "LINEXA_TTT_DECAY", "LINEXA_TTT_CLIP", "LINEXA_TTT_LOG_STATS",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        cfg = LinexaConfig.from_env()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    assert cfg.enabled is False
    assert cfg.layer_indices == []
    assert cfg.write_scale == 0.0
    assert cfg.decay == 0.0
    assert cfg.clip == 0.0
    assert cfg.log_stats is False


def test_enabled_truthy_variants():
    for val in ("1", "true", "TRUE", "yes", "On"):
        with _env(LINEXA_TTT_ENABLED=val):
            assert LinexaConfig.from_env().enabled is True


def test_enabled_falsy_variants():
    for val in ("0", "false", "no", "off", ""):
        with _env(LINEXA_TTT_ENABLED=val):
            assert LinexaConfig.from_env().enabled is False


def test_layer_indices_parses_csv():
    with _env(LINEXA_TTT_LAYERS="0,2,4"):
        assert LinexaConfig.from_env().layer_indices == [0, 2, 4]
    with _env(LINEXA_TTT_LAYERS=""):
        assert LinexaConfig.from_env().layer_indices == []


def test_write_knobs_parse_floats():
    with _env(
        LINEXA_TTT_WRITE_SCALE="0.5",
        LINEXA_TTT_DECAY="0.01",
        LINEXA_TTT_CLIP="1.0",
    ):
        cfg = LinexaConfig.from_env()
    assert cfg.write_scale == 0.5
    assert cfg.decay == 0.01
    assert cfg.clip == 1.0
