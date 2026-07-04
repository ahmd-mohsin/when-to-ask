"""Contract: A3 commitment + conformal tau (spec A3, checks 1-4).

Uses the fixtures' PLANTED lean projections as r -- A3 is agnostic to where r
comes from, and the planted structure gives exact ground truth for settle
points."""

import numpy as np
import pytest

from fixtures.synthetic import FixtureConfig, generate
from wta.a3_commitment import (
    CommitmentDetector, calibrate_tau, conformal_quantile, instability,
    s_reference, stabilization_point,
)

WINDOW = 4


def _lean_seqs(fx):
    """r sequences (planted lean coordinates) for every (run, ambiguous topic)."""
    seqs, meta = [], []
    for t in range(fx.cfg.n_ambiguous):
        for i in range(fx.cfg.n_runs):
            seqs.append(fx.h[i, t] @ fx.lean_dirs[t].T)
            meta.append((i, t))
    return seqs, meta


def test_conformal_quantile_finite_sample_formula():
    scores = np.arange(1.0, 11.0)  # 1..10
    # n=10, delta=0.1 -> k = ceil(11*0.9) = 10 -> 10th smallest
    assert conformal_quantile(scores, 0.1) == 10.0
    # delta=0.5 -> k = ceil(5.5) = 6
    assert conformal_quantile(scores, 0.5) == 6.0
    assert conformal_quantile(np.array([]), 0.1) == float("inf")


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    calib = rng.gamma(2.0, 1.0, size=300)
    tau = conformal_quantile(calib, delta=0.1)
    fresh = rng.gamma(2.0, 1.0, size=20000)
    coverage = float((fresh <= tau).mean())
    assert coverage >= 0.88, f"empirical coverage {coverage:.3f} < ~0.9"


def test_instability_shape_and_edges():
    r = np.zeros((6, 3))
    inst = instability(r, window=3)
    assert np.isinf(inst[:2]).all() and (inst[2:] == 0).all()
    r[3] += 5.0  # a jump makes every window containing read 3 unstable
    inst = instability(r, window=3)
    assert inst[3] > 1 and inst[4] > 1 and inst[5] > 1
    with pytest.raises(ValueError):
        instability(np.zeros(5), window=3)


def test_stabilization_point_matches_planted_settle():
    fx = generate(FixtureConfig(seed=9, n_runs=8, reads=24))
    seqs, meta = _lean_seqs(fx)
    hits = []
    for r, (i, t) in zip(seqs, meta):
        k = stabilization_point(r, eps_settle=0.35)
        assert k is not None
        hits.append(abs(k - fx.settle_idx[i, t]) <= 1)
    assert np.mean(hits) >= 0.9, f"only {np.mean(hits):.2f} within 1 read of planted settle"


def test_never_settling_run_yields_no_calibration_point():
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.standard_normal((30, 3)), axis=0)  # never settles
    assert stabilization_point(walk, eps_settle=0.3, min_tail=WINDOW) is None


def test_detector_matches_planted_commitment_and_ignores_loops():
    fx = generate(FixtureConfig(seed=13, n_runs=8, reads=24, loop_runs=(2,)))
    seqs, meta = _lean_seqs(fx)
    # eps_settle is in l_scale units (spec A3 amendment): 0.45 sits between the
    # scaled post-settle noise floor and the scaled deliberation wobble here;
    # tighter values starve the calibration set and admit late commits.
    calib = calibrate_tau(seqs, window=WINDOW, eps_settle=0.45, delta=0.1)
    assert calib.n_points > 0 and np.isfinite(calib.tau)

    s_ref = s_reference((fx.h[fx.ambiguous[None, :, None] & ~fx.committed] @ fx.d_star),
                        (fx.h[fx.ambiguous[None, :, None] & fx.committed] @ fx.d_star))

    ok, n_eval = [], 0
    for r_seq, (i, t) in zip(seqs, meta):
        s_seq = fx.h[i, t] @ fx.d_star
        det = CommitmentDetector(tau=calib.tau, s_ref=s_ref, window=WINDOW,
                                 l_scale=calib.l_scale)
        first = None
        for k in range(len(r_seq)):
            committed, w = det.step(r_seq[k], float(s_seq[k]))
            if committed and first is None:
                first = k
            if i == 2:  # loop run: must never commit, weight must stay 0
                assert not committed and w == 0.0
        if i == 2:
            continue
        n_eval += 1
        settle = fx.settle_idx[i, t]
        ok.append(first is not None and settle <= first <= settle + WINDOW + 1)
    assert np.mean(ok) >= 0.9, f"{np.mean(ok):.2f} of {n_eval} committed in [settle, settle+w+1]"


def test_streaming_equals_batch():
    fx = generate(FixtureConfig(seed=4, n_runs=4, reads=20))
    seqs, _ = _lean_seqs(fx)
    r = seqs[0]
    inst_batch = instability(r, WINDOW)
    det = CommitmentDetector(tau=0.4, s_ref=1e9, window=WINDOW)  # s-gate open
    for k in range(len(r)):
        committed, _ = det.step(r[k], s=0.0)
        expected = np.isfinite(inst_batch[k]) and inst_batch[k] <= 0.4
        assert committed == expected, f"read {k}: stream {committed} vs batch {expected}"


def test_decommit_on_blip():
    """A settled run that blips must flip committed off (vote retraction path)."""
    fx = generate(FixtureConfig(seed=6, n_runs=3, reads=30, blip_topics=(0,),
                                blip_at=0.8, blip_len=3))
    i, t = 0, 0
    r_seq = fx.h[i, t] @ fx.lean_dirs[t].T
    det = CommitmentDetector(tau=0.5, s_ref=1e9, window=3)
    states = [det.step(r_seq[k], s=0.0)[0] for k in range(len(r_seq))]
    blip_ks = np.where(fx.is_blip[i, t])[0]
    assert states[blip_ks[0] - 1]  # committed just before the blip
    assert not states[blip_ks[0]]  # de-committed the moment r jumps
