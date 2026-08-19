"""Contract: B2 registry-free behavioral divergence features (decisions/027,
spec B2-behavioral-divergence). Pins the spec's four observable behaviours:
planted fork fires / identical twin quiet, blip tolerance, the leak rule, and
hash determinism."""

from pathlib import Path

import numpy as np

from wta.divergence import (TriggerConfig, behavior_features, replay_task,
                            signed_hash_vec)
from wta.logging_schema import ActionEvent

CFG = TriggerConfig(theta=0.3, reference=0.05, slack=0.0, h_threshold=3.0,
                    min_votes=2)


def _run(edit_cmd: str, n_turns: int = 8):
    """A synthetic run: explore, one mutating edit at turn 2, then explores.
    All turns share topic evidence (same file + subgoal)."""
    obs = {"files": ["src/app.py"], "subgoal": "fix the timeout handling",
           "region": [], "error_signature": "exit 0"}
    actions = [ActionEvent(token_idx=5, action_text="cat src/app.py",
                           observables=dict(obs), segment_idx=0),
               ActionEvent(token_idx=5, action_text="grep timeout src/app.py",
                           observables=dict(obs), segment_idx=1),
               ActionEvent(token_idx=5, action_text=edit_cmd,
                           observables=dict(obs), segment_idx=2)]
    for k in range(3, n_turns):
        actions.append(ActionEvent(token_idx=5, action_text="cat src/app.py",
                                   observables=dict(obs), segment_idx=k))
    return actions


def test_planted_fork_fires_identical_twin_quiet():
    fork = {"r0": behavior_features(_run("sed -i 's/x/timeout = 30/' src/app.py")),
            "r1": behavior_features(_run("sed -i 's/x/timeout = 60/' src/app.py"))}
    twin = {"r0": behavior_features(_run("sed -i 's/x/timeout = 30/' src/app.py")),
            "r1": behavior_features(_run("sed -i 's/x/timeout = 30/' src/app.py"))}
    assert replay_task(fork, CFG), "diverging edits on a shared topic must fire"
    assert not replay_task(twin, CFG), "identical edits must never fire"


def test_blip_stays_under_then_agreement_quiet():
    """One divergent round followed by convergence to the same edit: the
    later vote REPLACES the earlier one (sticky, latest-wins), spread returns
    to zero, and a high-enough h is never crossed."""
    a = _run("sed -i 's/x/timeout = 30/' src/app.py", n_turns=10)
    b = _run("sed -i 's/x/timeout = 60/' src/app.py", n_turns=10)
    b[3] = ActionEvent(token_idx=5,
                       action_text="sed -i 's/x/timeout = 30/' src/app.py",
                       observables={"files": ["src/app.py"],
                                    "subgoal": "fix the timeout handling",
                                    "region": [], "error_signature": "exit 0"},
                       segment_idx=3)
    cfg = TriggerConfig(theta=0.3, reference=0.05, slack=0.0, h_threshold=6.0,
                        min_votes=2)
    runs = {"r0": behavior_features(a), "r1": behavior_features(b)}
    assert not replay_task(runs, cfg), \
        "a one-round blip resolved by re-commitment must stay under h"


def test_leak_rule_detector_never_touches_the_registry():
    src = (Path(__file__).resolve().parents[2] / "src" / "wta"
           / "divergence.py").read_text(encoding="utf-8")
    for banned in ("interpretation_classes", "load_class_artifact",
                   "blocker_registry", '["anchors"]', '["signatures"]',
                   '"anchors"', '"signatures"'):
        assert banned not in src, f"leak rule: divergence.py mentions {banned!r}"


def test_signed_hash_determinism_and_order_invariance():
    a = signed_hash_vec(["alpha", "beta", "gamma"], 64, seed=0)
    b = signed_hash_vec(["gamma", "alpha", "beta"], 64, seed=0)
    assert np.array_equal(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-12
    c = signed_hash_vec(["alpha", "beta", "gamma"], 64, seed=1)
    assert not np.array_equal(a, c), "seed must change the projection"
    assert float(np.linalg.norm(signed_hash_vec([], 64, seed=0))) == 0.0
