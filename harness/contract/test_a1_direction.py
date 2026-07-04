"""Contract: A1 recovers a planted ambiguity direction and separates held-out
should-ask vs proceed states (spec A1, checks 1-4). These are engineering
checks on synthetic data -- the real-data numbers are research output, reported
not tuned."""

import numpy as np
import pytest

from fixtures.synthetic import FixtureConfig, generate
from wta.a1_direction import ambiguity_signal, auroc, build_direction


@pytest.fixture(scope="module")
def split():
    """Build `d` from the MATCHED contrast (should-ask vs settled states of the
    same decisions -- topic cancels; spec A1 'Training contrast'); evaluate
    separation on held-out should-ask vs ALL specified-looking states."""
    fx = generate(FixtureConfig(seed=11, n_runs=8, reads=24))
    rng = np.random.default_rng(0)

    def cut(x, frac=0.7):
        x = x[rng.permutation(len(x))]
        c = int(frac * len(x))
        return x[:c], x[c:]

    pos_tr, pos_he = cut(fx.should_ask_states())
    settled_tr, settled_he = cut(fx.settled_states())
    return {
        "fx": fx,
        "train": (pos_tr, settled_tr),
        "held": (pos_he, np.concatenate([settled_he, fx.clear_states()])),
    }


def test_recovers_planted_direction(split):
    d = build_direction(*split["train"])
    cos = float(d @ split["fx"].d_star)
    assert cos >= 0.95, f"cos(d, d*) = {cos:.3f}"
    assert abs(np.linalg.norm(d) - 1.0) < 1e-6  # spec A1 bound


def test_separates_held_out_states(split):
    d = build_direction(*split["train"])
    s_pos = ambiguity_signal(split["held"][0], d)
    s_neg = ambiguity_signal(split["held"][1], d)
    score = auroc(s_pos, s_neg)
    assert score >= 0.9, f"held-out AUROC = {score:.3f}"


def test_s_drops_at_commitment(split):
    """Timing shape: on ambiguous decisions s falls once a run settles; clear
    decisions sit near 0 throughout (spec A1 check 3)."""
    fx, d = split["fx"], build_direction(*split["train"])
    s = ambiguity_signal(fx.h, d)
    amb = fx.ambiguous[None, :, None]
    assert s[amb & ~fx.committed].mean() > s[amb & fx.committed].mean() + 0.3
    assert abs(s[~amb & np.ones_like(fx.committed)].mean()) < 0.2


def test_determinism(split):
    d1 = build_direction(*split["train"])
    d2 = build_direction(*split["train"])
    assert np.array_equal(d1, d2)


def test_error_cases():
    h = np.zeros((4, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        build_direction(np.zeros((0, 8)), h)  # empty class
    with pytest.raises(ValueError):
        build_direction(np.zeros((4, 6)), h)  # dim mismatch
    bad = h.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        build_direction(bad, h)  # non-finite
    with pytest.raises(ValueError):
        build_direction(h, h)  # coinciding means -> no direction
    with pytest.raises(ValueError):
        ambiguity_signal(np.zeros(7), np.zeros(8))  # dim mismatch
    with pytest.raises(ValueError):
        auroc(np.array([]), np.array([1.0]))  # empty class
