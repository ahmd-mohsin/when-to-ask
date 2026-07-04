"""Contract: the synthetic fixture has exactly the planted structure it claims.

If these fail, no downstream validation means anything -- fix the generator,
never the assertions (the structure IS the spec, fixtures/synthetic.py).
"""

import numpy as np
import pytest

from fixtures.synthetic import FixtureConfig, generate


@pytest.fixture(scope="module")
def fx():
    return generate(FixtureConfig(seed=7))


def test_shapes_and_planted_basis(fx):
    cfg = fx.cfg
    assert fx.h.shape == (cfg.n_runs, cfg.n_topics, cfg.reads, cfg.hidden_dim)
    assert fx.d_star.shape == (cfg.hidden_dim,)
    assert fx.topic_dirs.shape == (cfg.n_topics, cfg.hidden_dim)
    assert fx.lean_dirs.shape == (cfg.n_ambiguous, cfg.n_classes, cfg.hidden_dim)
    # Planted directions are exactly orthonormal: no accidental overlap between
    # ambiguity, topic, and lean axes.
    dirs = np.vstack([fx.d_star[None], fx.topic_dirs, fx.lean_dirs.reshape(-1, cfg.hidden_dim)])
    gram = dirs @ dirs.T
    assert np.allclose(gram, np.eye(len(dirs)), atol=1e-8)


def test_ambiguity_signal_structure(fx):
    """s* = h . d_star: high while deliberating an ambiguous decision, ~0 after
    commitment and on clear decisions -- the confident-fork geometry."""
    s = fx.h @ fx.d_star
    amb = fx.ambiguous[None, :, None]
    pre = s[amb & ~fx.committed]
    post = s[amb & fx.committed]
    clear = s[~fx.ambiguous[None, :, None] & np.ones_like(fx.committed)]
    assert pre.mean() > 0.5 * fx.cfg.amb_scale
    assert abs(post.mean()) < 0.15
    assert abs(clear.mean()) < 0.1
    assert pre.mean() > 4 * abs(post.mean() - clear.mean())


def test_fork_guarantee(fx):
    """Every ambiguous decision has >= 2 interpretations among the N runs."""
    for t in range(fx.cfg.n_ambiguous):
        assert len(set(fx.class_id[:, t].tolist())) >= 2


def test_committed_lean_is_stable(fx):
    """Commitment is defined on r (the lean projection) having stopped moving --
    NOT on raw h, whose consecutive diffs are dominated by isotropic noise
    (method doc A3: "r has stopped moving over the last few reads"). Post-settle
    lean movement must sit well below deliberation swings."""
    post, pre = [], []
    for t in range(fx.cfg.n_ambiguous):
        r = fx.h[:, t] @ fx.lean_dirs[t].T  # (N, R, C): true lean coordinates
        dr = np.linalg.norm(np.diff(r, axis=1), axis=-1)  # (N, R-1)
        cm = fx.committed[:, t]
        post.append(dr[cm[:, 1:] & cm[:, :-1]])
        pre.append(dr[~cm[:, 1:]])
    post, pre = np.concatenate(post), np.concatenate(pre)
    assert post.mean() < 0.5 * pre.mean(), f"post {post.mean():.3f} vs pre {pre.mean():.3f}"


def test_blips_and_loops():
    cfg = FixtureConfig(seed=3, blip_topics=(0,), loop_runs=(1,), reads=24)
    fx = generate(cfg)
    # Blips exist, only on the flagged ambiguous topic, only post-settle.
    assert fx.is_blip.any()
    assert fx.is_blip[:, 1:, :].sum() == 0
    blip_reads = np.argwhere(fx.is_blip)
    for i, t, k in blip_reads:
        assert k >= fx.settle_idx[i, t]
    # Loop run: never commits on ambiguous decisions, and its states repeat.
    assert fx.settle_idx[1, 0] == -1
    assert not fx.committed[1, 0].any()
    loop_states = fx.h[1, 0]
    spread = np.linalg.norm(loop_states - loop_states.mean(axis=0), axis=-1)
    non_loop_movement = np.linalg.norm(np.diff(fx.h[0, 0], axis=0), axis=-1)
    assert spread.mean() < non_loop_movement.mean()


def test_config_validation():
    with pytest.raises(ValueError):
        generate(FixtureConfig(hidden_dim=8))  # fewer dims than planted directions
    with pytest.raises(ValueError):
        generate(FixtureConfig(n_classes=1))  # no fork possible
    with pytest.raises(ValueError):
        generate(FixtureConfig(blip_topics=(5,)))  # blip on a clear topic
