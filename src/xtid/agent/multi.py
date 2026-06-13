"""N-trajectory runner -- OURS.

Runs N trajectories of one task from the same scaffold (best-of-N spine, brief S5b),
diversified by temperature sampling and optional per-trajectory diverse prompts. Returns
each trajectory's step sequence; `signals.alignment` then groups them into aligned
decision points across the N.

A *controller seam* sits above the N: in the full method it reads cross-trajectory
divergence at each aligned decision point and decides whether to call ask_human() once
(dedup across the N -- brief S5f hazard 1). For the de-risk pass we only record the
signals; the controller/ask loop is deferred (interfaces are in place).
"""

from __future__ import annotations

from ..backbone.model import WhiteBoxModel
from ..harness.tasks import Task
from .loop import StepResult, Trajectory

# A few generic "diverse prompting" suffixes to spread real-model trajectories beyond
# temperature alone. Harmless for the FakeWhiteBoxModel (it diversifies via seed).
DIVERSE_PROMPTS = [
    "",
    "Think step by step before acting.",
    "Consider edge cases first.",
    "Prefer the simplest correct approach.",
    "Double-check the requirements before deciding.",
]


def run_task(
    model: WhiteBoxModel,
    task: Task,
    n_trajectories: int = 5,
    temperature: float = 0.8,
    diverse_prompting: bool = False,
) -> list[list[StepResult]]:
    """Run N trajectories of one task; return one StepResult list per trajectory."""
    trajectories: list[list[StepResult]] = []
    for tid in range(n_trajectories):
        suffix = DIVERSE_PROMPTS[tid % len(DIVERSE_PROMPTS)] if diverse_prompting else ""
        trajectories.append(
            Trajectory(model, task, traj_id=tid, temperature=temperature, diverse_prompt=suffix).run()
        )
    return trajectories


def run_tasks(
    model: WhiteBoxModel,
    tasks: list[Task],
    n_trajectories: int = 5,
    temperature: float = 0.8,
    diverse_prompting: bool = False,
) -> dict[str, list[list[StepResult]]]:
    """Run every task; return {instance_id -> per-trajectory step sequences}."""
    return {
        t.instance_id: run_task(model, t, n_trajectories, temperature, diverse_prompting)
        for t in tasks
    }
