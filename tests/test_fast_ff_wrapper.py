"""Phase 0 wrapper-equivalence unit tests.

Verifies the safety contract from docs/plan.md §9 / §10:

- empty-cache fast path delegates to the wrapped module (bit-exact);
- with ``ΔW = 0`` injected, the manual recomposition path matches the wrapped
  module within the equivalence tolerance (``max |Δ| < 1e-5`` per docs/plan.md §10);
- ``reset()`` clears the cache and restores fast-path behavior;
- the wrapper rejects unsupported FeedForward layouts at install time.

Skips automatically if torch or diffusers is not available, so the rest of the
test suite (config-only) still runs in lightweight environments.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
ff_module = pytest.importorskip("diffusers.models.attention")

from linexa.ttt.fast_ff_wrapper import FastFFWrapper  # noqa: E402
from linexa.ttt.fast_weight import FastWeightCache  # noqa: E402

FeedForward = ff_module.FeedForward
NOOP_TOL = 1e-5


def _make_ff(dim: int = 1536) -> torch.nn.Module:
    torch.manual_seed(0)
    return FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate").eval()


def _input(dim: int = 1536) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 24, dim)


def test_empty_cache_passthrough_is_bit_exact():
    ff = _make_ff()
    wrapper = FastFFWrapper(wrapped=ff, cache=FastWeightCache(), layer_id=0).eval()
    x = _input()
    with torch.no_grad():
        ref = ff(x)
        got = wrapper(x)
    assert torch.equal(ref, got), f"max|Δ|={float((ref-got).abs().max()):.2e}"


def test_zero_delta_slow_path_within_tolerance():
    ff = _make_ff()
    cache = FastWeightCache()
    wrapper = FastFFWrapper(wrapped=ff, cache=cache, layer_id=0).eval()
    cache._deltas[0] = torch.zeros_like(ff.net[2].weight)
    x = _input()
    with torch.no_grad():
        ref = ff(x)
        got = wrapper(x)
    assert (ref - got).abs().max().item() < NOOP_TOL


def test_reset_restores_fast_path():
    ff = _make_ff()
    cache = FastWeightCache()
    wrapper = FastFFWrapper(wrapped=ff, cache=cache, layer_id=0).eval()
    cache._deltas[0] = torch.zeros_like(ff.net[2].weight)
    cache.reset()
    assert cache.reset_count == 1
    assert cache.get(0) is None
    x = _input()
    with torch.no_grad():
        assert torch.equal(ff(x), wrapper(x))


def test_rejects_unsupported_net_length():
    fake = torch.nn.Module()
    fake.net = torch.nn.ModuleList([
        torch.nn.Linear(4, 4),
        torch.nn.Dropout(),
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 4),  # extra trailing module
    ])
    with pytest.raises(TypeError, match="length 3"):
        FastFFWrapper(wrapped=fake, cache=FastWeightCache(), layer_id=0)
