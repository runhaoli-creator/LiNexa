"""Adapter Phase 0 safety checks (do not require psi0 to be importable)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from linexa.adapters.psi0 import _check_phase0_safe  # noqa: E402
from linexa.ttt.config import LinexaConfig  # noqa: E402


def test_zero_knobs_pass():
    _check_phase0_safe(LinexaConfig(enabled=True))  # all zeros, no raise


@pytest.mark.parametrize(
    "knob, value",
    [
        ("write_scale", 0.5),
        ("decay", 0.01),
        ("clip", 1.0),
    ],
)
def test_nonzero_knob_raises(knob, value):
    cfg = LinexaConfig(enabled=True, **{knob: value})
    with pytest.raises(RuntimeError, match="Mode B is not implemented"):
        _check_phase0_safe(cfg)
