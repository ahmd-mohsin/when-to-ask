"""Asynchronous cross-trajectory hidden-state alignment -- OURS (novel).

The N trajectories reach comparable states at *different* steps, so before we can read
cross-trajectory divergence we must decide which steps line up. Brief S5c: align by step
index for early steps, then by a semantic anchor (same sub-goal / file under edit) once
they diverge. No prior work aligns hidden states across asynchronous multi-step agent
rollouts, so this is a named methodological contribution -- and alignment-scheme
sensitivity is a reportable result, hence the pluggable `scheme`.

The anchors are produced upstream (`agent.decision_points`): "step{i}" for the early
synchronous phase, a semantic key ("file:...", "dp{i}") afterwards. So:

  * scheme="step_index"            -- group purely by step index (ablation lower bound).
  * scheme="semantic_anchor"       -- group purely by the semantic anchor.
  * scheme="step_index_then_anchor"-- the brief's default: anchors already encode step for
                                      the early phase, so we group by anchor (which is
                                      step-index early, semantic later).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..agent.loop import StepResult
from ..harness.tasks import Task


@dataclass
class DPObservation:
    """The N trajectories' aligned states at one decision point."""

    task_id: str
    anchor: str
    step: int  # representative (min) step index across members
    members: dict[int, StepResult]  # traj_id -> aligned step
    regime: str | None = None  # gold (synthetic) or assigned later (real)
    should_ask: bool = False
    blocker_id: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.members)

    @property
    def hiddens(self) -> np.ndarray:
        return np.stack([self.members[t].hidden for t in sorted(self.members)])

    @property
    def actions(self) -> list[str]:
        return [self.members[t].action_text for t in sorted(self.members)]

    @property
    def generations(self) -> list:
        return [self.members[t].generation for t in sorted(self.members)]


def _key(step: StepResult, scheme: str) -> str:
    if scheme == "step_index":
        return f"step{step.step}"
    return step.anchor  # semantic_anchor / step_index_then_anchor


def align(trajectories: list[list[StepResult]], scheme: str = "step_index_then_anchor") -> list[dict]:
    """Group steps across trajectories into aligned decision points.

    Returns a list of {anchor, step, members} dicts, ordered by representative step.
    Handles asynchronous trajectories (different lengths, anchors appearing at different
    steps); a trajectory contributes at most one step per anchor (its earliest).
    """
    groups: dict[str, dict[int, StepResult]] = {}
    order: dict[str, int] = {}
    for traj in trajectories:
        for step in traj:
            key = _key(step, scheme)
            members = groups.setdefault(key, {})
            if step.traj_id not in members:  # earliest occurrence wins
                members[step.traj_id] = step
            order[key] = min(order.get(key, step.step), step.step)
    aligned = [
        {"anchor": k, "step": order[k], "members": v}
        for k, v in groups.items()
    ]
    aligned.sort(key=lambda g: (g["step"], g["anchor"]))
    return aligned


def build_dp_observations(
    trajectories: list[list[StepResult]],
    task: Task,
    scheme: str = "step_index_then_anchor",
) -> list[DPObservation]:
    """Align trajectories and attach gold labels (synthetic) by anchor -> decision point."""
    by_anchor = {f"dp{dp.index}": dp for dp in task.decision_points}
    obs: list[DPObservation] = []
    for g in align(trajectories, scheme):
        gold = by_anchor.get(g["anchor"])
        obs.append(
            DPObservation(
                task_id=task.instance_id,
                anchor=g["anchor"],
                step=g["step"],
                members=g["members"],
                regime=gold.regime if gold else None,
                should_ask=gold.should_ask if gold else False,
                blocker_id=(gold.blocker.id if gold and gold.blocker else None),
            )
        )
    return obs
