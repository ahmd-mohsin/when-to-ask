"""Single-trajectory agent loop -- OURS (adapted from mini-swe-agent).

Base: third_party/mini-swe-agent/.../agents/default.py -- a minimal observe -> query ->
act loop. We adapt it so that (a) the model is our white-box backbone and (b) at every
decision point we capture the mid-layer hidden state alongside the action.

Synthetic mode (CPU smoke) walks the task's known decision points, feeding each step's
control marker to the FakeWhiteBoxModel. Real mode (GPU) would drive a real environment;
that seam is `RealDecisionPointDetector` + an env (deferred to the GPU run).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backbone.model import Generation, WhiteBoxModel
from ..harness.tasks import DecisionPoint, Task
from .decision_points import synthetic_anchor


@dataclass
class StepResult:
    """One trajectory's commitment at one decision point."""

    traj_id: int
    step: int
    anchor: str  # alignment key (step index early, semantic anchor once diverged)
    hidden: np.ndarray  # (hidden_dim,) mid-layer residual
    action_text: str
    generation: Generation
    dp: DecisionPoint | None = None  # gold decision point (synthetic)


@dataclass
class Trajectory:
    model: WhiteBoxModel
    task: Task
    traj_id: int
    temperature: float = 0.8
    diverse_prompt: str = ""  # optional per-trajectory prompt suffix (diverse prompting)
    steps: list[StepResult] = field(default_factory=list)

    def _build_prompt(self, dp: DecisionPoint) -> str:
        # The control marker drives the FakeWhiteBoxModel; a real model just sees text.
        return (
            f"{self.task.statement}\n{self.diverse_prompt}\n"
            f"Step {dp.index}: {dp.ctrl} Decide the next action."
        ).strip()

    def run(self) -> list[StepResult]:
        """Synthetic rollout: one step per gold decision point.

        The trajectory's *interpretation* is fixed by its id (seed=traj_id), so it holds
        a consistent interpretation across steps -- only the per-step noise varies.
        """
        self.steps = []
        for dp in self.task.decision_points:
            gen = self.model.generate(
                self._build_prompt(dp), temperature=self.temperature, seed=self.traj_id
            )
            self.steps.append(
                StepResult(
                    traj_id=self.traj_id,
                    step=dp.index,
                    anchor=synthetic_anchor(dp),
                    hidden=gen.hidden,
                    action_text=gen.text,
                    generation=gen,
                    dp=dp,
                )
            )
        return self.steps
