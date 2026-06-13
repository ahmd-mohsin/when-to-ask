"""Asynchronous cross-trajectory alignment (OURS, novel) + scheme sensitivity."""

import numpy as np

from xtid.agent.loop import StepResult
from xtid.agent.multi import run_task
from xtid.backbone.model import Generation, build_model
from xtid.harness.tasks import make_synthetic_tasks
from xtid.signals.alignment import align, build_dp_observations


def _step(traj_id, step, anchor):
    return StepResult(
        traj_id=traj_id, step=step, anchor=anchor,
        hidden=np.zeros(4), action_text="a", generation=Generation("a", np.zeros(4)),
    )


def _async_trajectories():
    # Two trajectories that reach the same files in a different ORDER (out of sync).
    return [
        [_step(0, 0, "step0"), _step(0, 1, "file:a.py"), _step(0, 2, "file:b.py")],
        [_step(1, 0, "step0"), _step(1, 1, "file:b.py"), _step(1, 2, "file:a.py")],
    ]


def test_semantic_anchor_aligns_across_different_steps():
    groups = {g["anchor"]: g["members"] for g in align(_async_trajectories(), "semantic_anchor")}
    # a.py reached at step 1 by traj0 and step 2 by traj1 -- still grouped together.
    assert set(groups["file:a.py"]) == {0, 1}
    assert set(groups["file:b.py"]) == {0, 1}
    assert groups["file:a.py"][1].step == 2  # traj1 reached a.py late


def test_step_index_scheme_misaligns_async_trajectories():
    # Under pure step-index, step 1 mixes traj0's a.py with traj1's b.py -> different result.
    groups = {g["anchor"]: g["members"] for g in align(_async_trajectories(), "step_index")}
    anchors_at_step1 = {groups["step1"][t].anchor for t in groups["step1"]}
    assert anchors_at_step1 == {"file:a.py", "file:b.py"}  # genuinely misaligned -> sensitivity


def test_earliest_occurrence_wins_per_trajectory():
    traj = [[_step(0, 1, "file:a.py"), _step(0, 3, "file:a.py")]]
    members = align(traj, "semantic_anchor")[0]["members"]
    assert members[0].step == 1


def test_build_dp_observations_attaches_gold_regime():
    model = build_model({"kind": "fake"})
    fork = next(t for t in make_synthetic_tasks(n=3, seed=0) if t.meta["kind"] == "fork")
    obs = build_dp_observations(run_task(model, fork, n_trajectories=4), fork)
    assert [o.anchor for o in obs] == [f"dp{d.index}" for d in fork.decision_points]
    assert [o.regime for o in obs] == [d.regime for d in fork.decision_points]
    assert any(o.should_ask for o in obs)
    assert obs[-1].hiddens.shape == (4, model.hidden_dim)
