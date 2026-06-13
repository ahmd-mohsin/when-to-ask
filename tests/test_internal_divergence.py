"""Cross-trajectory internal divergence metrics (OURS + migrated)."""

import numpy as np
import pytest

from xtid.signals import internal_divergence as idiv


def _tight(n=5, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    return np.ones((n, dim)) + 0.01 * rng.standard_normal((n, dim))


def _spread(n=5, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    return 5.0 * rng.standard_normal((n, dim))


@pytest.mark.parametrize("metric", idiv.METRICS)
def test_spread_more_divergent_than_tight(metric):
    assert idiv.divergence(_spread(), metric) > idiv.divergence(_tight(), metric)


def test_degenerate_single_vector_is_zero():
    one = np.ones((1, 16))
    assert all(idiv.divergence(one, m) == 0.0 for m in idiv.METRICS)
    assert idiv.compute_all(one) == {m: 0.0 for m in idiv.METRICS}


def test_mean_pairwise_cosine_is_scale_invariant():
    H = _spread(seed=3)
    assert idiv.mean_pairwise_cosine(H) == pytest.approx(idiv.mean_pairwise_cosine(100.0 * H), rel=1e-9)


def test_compute_all_returns_every_metric():
    assert set(idiv.compute_all(_spread())) == set(idiv.METRICS)


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        idiv.divergence(_spread(), "nope")
