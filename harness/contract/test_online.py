"""Contract: Part B online trigger (spec B, checks 1-7)."""

import numpy as np
import pytest

from wta.bucketing import LeaderClusters, leader_cluster_points
from wta.online import AskTrigger, TriggerConfig


def _cfg(**kw):
    base = dict(theta=0.7, reference=0.2, slack=0.05, h_threshold=6.0,
                lambda_loop=0.5, hysteresis_m=3)
    base.update(kw)
    return TriggerConfig(**base)


U0 = np.array([1.0, 0.0, 0.0, 0.0])
U1 = np.array([0.0, 1.0, 0.0, 0.0])
RA, RB = np.array([1.0, 0.0]), np.array([0.0, 1.0])


def test_persistent_fork_fires_with_both_options():
    trig = AskTrigger(_cfg())
    trig.register_action("A", "retry all errors")
    trig.register_action("B", "retry transient only")
    fired = None
    for k in range(10):
        assert fired is None or k <= 6, "should have fired within a few reads"
        d = trig.observe("A", U0, RA, s=0.0, weight=1.0)
        fired = fired or d
        d = trig.observe("B", U0, RB, s=0.0, weight=1.0)
        fired = fired or d
        if fired:
            break
    assert fired is not None
    assert sorted(o["run_id"] for o in fired.options) == ["A", "B"]
    texts = {o["run_id"]: o["action_text"] for o in fired.options}
    assert texts["A"] == "retry all errors" and texts["B"] == "retry transient only"


def test_transient_blip_does_not_fire_and_pressure_decays():
    trig = AskTrigger(_cfg())
    fired = []
    for k in range(20):
        rb = RB if k in (3, 4) else RA  # 2-read blip, then votes re-converge
        fired.append(trig.observe("A", U0, RA, s=0.0, weight=1.0))
        fired.append(trig.observe("B", U0, rb, s=0.0, weight=1.0))
    assert not any(fired), "blip must not fire"
    (bucket_id,) = trig.clusters.buckets.keys()
    assert trig._cusum[bucket_id] == 0.0, "pressure must decay to zero"


def test_decommit_retracts_vote():
    trig = AskTrigger(_cfg())
    trig.observe("A", U0, RA, s=0.0, weight=1.0)
    trig.observe("B", U0, RB, s=0.0, weight=1.0)
    (bucket,) = trig.clusters.buckets.values()
    assert len(bucket.votes) == 2
    trig.observe("B", U0, RB, s=0.9, weight=0.0)  # B went back to deliberating
    assert list(bucket.votes) == ["A"]
    assert trig._spread(bucket) == 0.0


def test_hysteresis_holds_then_moves():
    lc = LeaderClusters(theta=0.7, hysteresis_m=3)
    b0 = lc.assign("A", U0)
    assert lc.assign("A", U1) == b0  # out-of-theta read 1: held
    assert lc.assign("A", U1) == b0  # read 2: held
    assert len(lc.buckets) == 1     # and the centroid was never polluted
    b1 = lc.assign("A", U1)          # read 3 = M: moves
    assert b1 != b0
    assert np.allclose(lc.buckets[b0].centroid, U0)


def test_merge_makes_bucketing_order_invariant():
    # three vectors pairwise >= theta around U0; greedy order can split them
    v1 = np.array([1.0, 0.30, 0.0, 0.0])
    v2 = np.array([1.0, -0.30, 0.0, 0.0])
    v3 = np.array([1.0, 0.0, 0.0, 0.0])
    for order in ([v1, v2, v3], [v2, v3, v1], [v3, v1, v2]):
        labels = leader_cluster_points(np.array(order), theta=0.9)
        assert len(set(labels.tolist())) == 1, f"order split buckets: {labels}"


def test_loop_channel_fires_for_thrashing_run():
    trig = AskTrigger(_cfg(lambda_loop=0.5))
    fired = None
    for k in range(12):
        trig.notify_env_state("L", state_hash="same-state")  # stuck
        d = trig.observe("L", U0, RA, s=0.9, weight=0.0)     # never commits
        fired = fired or d
        if fired:
            break
    assert fired is not None, "loop channel must fire for a stuck run"
    assert fired.looping_runs == ["L"] and fired.options == []


def test_inject_resolution_clears_bucket():
    trig = AskTrigger(_cfg())
    d = None
    for _ in range(30):  # bounded: the fire can land on EITHER run's observe
        d = (trig.observe("A", U0, RA, s=0.0, weight=1.0)
             or trig.observe("B", U0, RB, s=0.0, weight=1.0))
        if d is not None:
            break
    assert d is not None, "persistent fork must fire"
    trig.inject_resolution(d.bucket_id)
    assert trig.clusters.buckets[d.bucket_id].votes == {}
    assert trig._cusum[d.bucket_id] == 0.0


def test_zero_norm_topic_rejected():
    trig = AskTrigger(_cfg())
    with pytest.raises(ValueError):
        trig.observe("A", np.zeros(4), RA, s=0.0, weight=1.0)
