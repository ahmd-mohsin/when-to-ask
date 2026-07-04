"""A4 gate MACHINERY checks on planted-structure fixtures.

These assert that the gate code measures what it claims on data where the
truth is known -- the pipeline SHOULD pass here. The real-data gate run
(scripts/run_gates.py on AWS activations) is the science; its numbers are
reported to the owner unfiltered and never tuned (spec A4, decisions/011).
"""

import numpy as np
import pytest

from fixtures.synthetic import FixtureConfig, generate
from wta.a1_direction import build_direction
from wta.a2_autoencoder import A2Config, train_a2
from wta.a3_commitment import CommitmentDetector, calibrate_tau, s_reference
from wta.a4_gates import (
    gate1_topic_leakage, gate2_decision_recovery, gate3_fork_collocation,
    gate4_conflation, gate5_lean_separation, gate6_ood_transfer,
    gate7_aggregate, gate7_lead_time,
)

TRAIN_TOPICS = (0, 1, 3, 4)  # topics 2 (ambiguous) and 5 (clear) are unseen


def _flatten(fx, topics):
    n, t, r, hdim = fx.h.shape
    rows = []
    for tt in topics:
        for i in range(n):
            for k in range(r):
                cls = int(fx.class_id[i, tt]) if (fx.ambiguous[tt] and fx.committed[i, tt, k]) else -1
                rows.append((fx.h[i, tt, k], tt, cls))
    X = np.stack([x for x, _, _ in rows])
    return X, np.array([t for _, t, _ in rows]), np.array([c for _, _, c in rows])


@pytest.fixture(scope="module")
def pipe():
    fx = generate(FixtureConfig(seed=21, n_runs=8, reads=24))
    X_tr_all, top_tr_all, cls_tr_all = _flatten(fx, TRAIN_TOPICS)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X_tr_all))
    cut = int(0.7 * len(idx))
    tr, he = idx[:cut], idx[cut:]

    cfg = A2Config(in_dim=fx.cfg.hidden_dim, n_topics=fx.cfg.n_topics,
                   n_classes=fx.cfg.n_classes, epochs=120, seed=0)
    model = train_a2(X_tr_all[tr], top_tr_all[tr], cls_tr_all[tr], cfg)

    X_he, top_he, cls_he = X_tr_all[he], top_tr_all[he], cls_tr_all[he]
    return {
        "fx": fx, "model": model,
        "t_tr": model.encode_topic(X_tr_all[tr]), "top_tr": top_tr_all[tr],
        "cls_tr": cls_tr_all[tr],
        "t_he": model.encode_topic(X_he), "l_he": model.encode_lean(X_he),
        "top_he": top_he, "cls_he": cls_he,
    }


def test_gate1_topic_blind_to_lean(pipe):
    g = gate1_topic_leakage(pipe["t_tr"], pipe["cls_tr"], pipe["t_he"], pipe["cls_he"])
    assert g.numbers["class_from_T_acc"] <= g.numbers["chance"] + 0.15, str(g)
    assert g.numbers["eta2"] < 0.1, str(g)


def test_gate2_decision_recovery(pipe):
    g = gate2_decision_recovery(pipe["t_tr"], pipe["top_tr"], pipe["t_he"], pipe["top_he"])
    assert g.numbers["topic_from_T_acc"] >= 0.9, str(g)


def test_gate3_collocation_sets_theta(pipe):
    g = gate3_fork_collocation(pipe["t_he"], pipe["top_he"], pipe["cls_he"])
    assert g.numbers["mean_same_decision_cos"] > g.numbers["mean_diff_decision_cos"] + 0.2, str(g)
    assert g.numbers["mean_diff_decision_cos"] < g.numbers["theta"] < g.numbers["mean_same_decision_cos"], str(g)
    pipe["theta"] = g.numbers["theta"]


def test_gate4_conflation_machinery(pipe):
    """Adversarial pairing: different decisions labelled 'same file' must NOT
    collocate on planted-orthogonal fixtures."""
    g3 = gate3_fork_collocation(pipe["t_he"], pipe["top_he"], pipe["cls_he"])
    rng = np.random.default_rng(1)
    tops = pipe["top_he"]
    pairs = []
    for _ in range(500):
        a, b = rng.integers(0, len(tops), 2)
        if tops[a] != tops[b]:
            pairs.append((a, b))
    g = gate4_conflation(pipe["t_he"], np.array(pairs), g3.numbers["theta"])
    assert g.numbers["frac_collocated"] <= 0.2, str(g)


def test_gate5_lean_separation(pipe):
    g = gate5_lean_separation(pipe["l_he"], pipe["top_he"], pipe["cls_he"])
    assert g.numbers["between_within_ratio"] > 2.0, str(g)
    assert g.numbers["n_decisions"] >= 2, str(g)


def test_gate6_purity_machinery_and_ood_report(pipe):
    """Machinery: purity is high on held-out reads of SEEN decisions. On
    UNSEEN decisions the fixture gives a genuine transfer failure by
    construction (an MLP has no inductive bridge between orthogonal planted
    directions), so the unseen number is asserted well-formed and printed as
    the honest limitation it is -- the real-data value is gate 6's research
    output, not a machinery pass (spec A4)."""
    fx, model = pipe["fx"], pipe["model"]
    g3 = gate3_fork_collocation(pipe["t_he"], pipe["top_he"], pipe["cls_he"])
    g_seen = gate6_ood_transfer(pipe["t_he"], pipe["top_he"], g3.numbers["theta"])
    assert g_seen.numbers["bucket_purity"] >= 0.8, str(g_seen)

    X_un, top_un, _ = _flatten(fx, (2, 5))  # never seen in A2 training
    g_un = gate6_ood_transfer(model.encode_topic(X_un), top_un, g3.numbers["theta"])
    assert 0.0 <= g_un.numbers["bucket_purity"] <= 1.0
    print(f"\nOOD (unseen fixture decisions, honest limitation): {g_un}")


def test_gate7_positive_lead_time(pipe):
    """Full-pipeline lead-time: trained L + calibrated A3 weights; the planted
    action_delay is the ground-truth lead window."""
    fx, model = pipe["fx"], pipe["model"]
    d = build_direction(fx.should_ask_states(), fx.settled_states())
    s_ref = s_reference(fx.should_ask_states() @ d, fx.proceed_states() @ d)

    window = 4
    seqs = [model.encode_lean(fx.h[i, t]) for t in range(fx.cfg.n_ambiguous)
            for i in range(fx.cfg.n_runs)]
    calib = calibrate_tau(seqs, window=window, eps_settle=0.35, delta=0.1)

    per_decision = []
    for t in range(fx.cfg.n_ambiguous):
        r_by_run = np.stack([model.encode_lean(fx.h[i, t]) for i in range(fx.cfg.n_runs)])
        weights = np.zeros(r_by_run.shape[:2])
        for i in range(fx.cfg.n_runs):
            det = CommitmentDetector(tau=calib.tau, s_ref=s_ref, window=window,
                                     l_scale=calib.l_scale)
            s_seq = fx.h[i, t] @ d
            for k in range(r_by_run.shape[1]):
                _, weights[i, k] = det.step(r_by_run[i, k], float(s_seq[k]))
        per_decision.append(gate7_lead_time(
            r_by_run, weights, fx.action_read[:, t], fx.class_id[:, t]))

    g = gate7_aggregate(per_decision, proxy=True)
    assert g.numbers["n_decisions"] >= 2, str(g)
    assert g.numbers["median_K"] > 0, str(g)
    assert g.numbers["frac_positive"] >= 0.5, str(g)
