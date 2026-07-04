"""Contract: A2 disentangling autoencoder (spec A2, checks 1-5) on fixtures.

Engineering checks on planted-structure data. The real-data versions of the
invariance/recovery numbers are A4 gates -- reported, never tuned.
"""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from fixtures.synthetic import FixtureConfig, generate
from wta.a2_autoencoder import (
    A2Config, A2Model, DisentangleAE, compute_losses, train_a2,
)


def _flatten_fixture(fx):
    """Reads -> (X, topic_y, class_y).

    class_y is -1 (unlabeled) except on COMMITTED reads of ambiguous
    decisions: a pre-settle read's lean content is a deliberation mixture, so
    labeling it with the run's eventual class would be wrong supervision --
    the registry class characterizes the committed interpretation (spec A2).
    """
    n, t, r, hdim = fx.h.shape
    X = fx.h.reshape(-1, hdim)
    topic_y = np.tile(np.repeat(np.arange(t), r), n)
    class_read = np.where(
        fx.ambiguous[None, :, None] & fx.committed, fx.class_id[:, :, None], -1
    )  # (N, T, R)
    return X, topic_y.astype(np.int64), class_read.reshape(-1).astype(np.int64)


@pytest.fixture(scope="module")
def trained():
    fx = generate(FixtureConfig(seed=5, n_runs=8, reads=20))
    X, ty, cy = _flatten_fixture(fx)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    tr, he = idx[:cut], idx[cut:]
    cfg = A2Config(in_dim=fx.cfg.hidden_dim, n_topics=fx.cfg.n_topics,
                   n_classes=fx.cfg.n_classes, epochs=120, seed=0)
    model = train_a2(X[tr], ty[tr], cy[tr], cfg)
    return {"fx": fx, "model": model, "cfg": cfg,
            "held": (X[he], ty[he], cy[he]), "train": (X[tr], ty[tr], cy[tr])}


def test_all_losses_wired_and_nonzero():
    """Spec check 1: at init every pull is finite and non-zero, and one step
    sends gradient into every component."""
    fx = generate(FixtureConfig(seed=1, n_runs=4, reads=8))
    X, ty, cy = _flatten_fixture(fx)
    cfg = A2Config(in_dim=fx.cfg.hidden_dim, n_topics=fx.cfg.n_topics,
                   n_classes=fx.cfg.n_classes, seed=0)
    net = DisentangleAE(cfg)
    losses = compute_losses(net, torch.from_numpy(X), torch.from_numpy(ty),
                            torch.from_numpy(cy), grl_lam=1.0)
    for name in ("topic", "supcon", "lean", "adv", "condmean", "ortho", "recon", "total"):
        val = float(losses[name].detach())
        # non-zero = not disconnected; ortho/condmean legitimately start tiny
        # at random init (decorrelation of random features), so the threshold
        # only rejects literal zeros -- gradient flow is asserted per
        # component below.
        assert np.isfinite(val) and abs(val) > 1e-15, f"{name} = {val}"
    losses["total"].backward()
    for comp in ("body", "head_topic", "head_lean", "decoder",
                 "topic_cls", "lean_cls", "adversary"):
        grads = [p.grad for p in getattr(net, comp).parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), comp


def test_topic_recovers_decision(trained):
    X, ty, _ = trained["held"]
    Xt, tyt, _ = trained["train"]
    t_he = trained["model"].encode_topic(X)
    probe = LogisticRegression(max_iter=1000).fit(trained["model"].encode_topic(Xt), tyt)
    acc = probe.score(t_he, ty)
    assert acc >= 0.9, f"topic probe acc = {acc:.3f}"


def test_lean_recovers_class(trained):
    X, _, cy = trained["held"]
    Xt, _, cyt = trained["train"]
    he, tr = cy >= 0, cyt >= 0
    probe = LogisticRegression(max_iter=1000).fit(
        trained["model"].encode_lean(Xt[tr]), cyt[tr])
    acc = probe.score(trained["model"].encode_lean(X[he]), cy[he])
    assert acc >= 0.9, f"lean probe acc = {acc:.3f}"


def test_topic_blind_to_lean(trained):
    """Fixture-level invariance (planted structure is orthogonal, so this is
    an engineering check; the real-data version is gate 1)."""
    X, _, cy = trained["held"]
    Xt, _, cyt = trained["train"]
    he, tr = cy >= 0, cyt >= 0
    probe = LogisticRegression(max_iter=1000).fit(
        trained["model"].encode_topic(Xt[tr]), cyt[tr])
    acc = probe.score(trained["model"].encode_topic(X[he]), cy[he])
    chance = 1.0 / trained["cfg"].n_classes
    assert acc <= chance + 0.15, f"class-from-T acc = {acc:.3f} (chance {chance:.2f})"


def test_reconstruction_floor(trained):
    X = trained["held"][0]
    h_hat = trained["model"].reconstruct(X)
    r2 = 1.0 - np.mean((h_hat - X) ** 2) / np.var(X)
    assert r2 >= 0.5, f"held-out recon R^2 = {r2:.3f}"


def test_determinism_and_roundtrip(tmp_path):
    fx = generate(FixtureConfig(seed=2, n_runs=4, reads=8))
    X, ty, cy = _flatten_fixture(fx)
    cfg = A2Config(in_dim=fx.cfg.hidden_dim, n_topics=fx.cfg.n_topics,
                   n_classes=fx.cfg.n_classes, epochs=3, seed=42)
    m1 = train_a2(X, ty, cy, cfg)
    m2 = train_a2(X, ty, cy, cfg)
    assert np.array_equal(m1.encode_topic(X), m2.encode_topic(X))
    assert np.array_equal(m1.encode_lean(X), m2.encode_lean(X))
    m1.save(tmp_path / "a2.pt")
    m3 = A2Model.load(tmp_path / "a2.pt")
    assert np.array_equal(m1.encode_topic(X), m3.encode_topic(X))


def test_input_validation():
    cfg = A2Config(in_dim=8, n_topics=2, n_classes=2, epochs=1)
    with pytest.raises(ValueError):
        train_a2(np.zeros((4, 7), dtype=np.float32), np.zeros(4), np.zeros(4), cfg)
    with pytest.raises(ValueError):
        train_a2(np.zeros((4, 8), dtype=np.float32), np.zeros(3), np.zeros(4), cfg)
