"""Per-decision-point recording -- OURS.

Turns the raw N-trajectory runs into the flat table the C1 analysis consumes. For each
aligned decision point we record every signal compared in the de-risk experiment:

  * internal cross-trajectory divergence (all metrics)      -- ours
  * output cross-trajectory divergence (B1)                 -- ClarifyGPT
  * verbalized-across-N (B2)                                -- aggregate self-report
  * single-stream correctness probe (B3), out-of-fold       -- OPENIA-style

plus the gold regime / should-ask label and the raw per-trajectory hidden states and
correctness labels (so the probe is trained here, leakage-free, before scoring).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..agent.loop import StepResult
from ..harness.executor import Executor
from ..harness.tasks import Task
from ..signals import internal_divergence as idiv
from ..signals.alignment import DPObservation, build_dp_observations
from ..signals.output_divergence import output_divergence
from ..signals.probe import dp_ask_score, oof_incorrect_proba
from ..signals.verbalized import verbalized_ask_score


@dataclass
class DPRecord:
    task_id: str
    anchor: str
    step: int
    regime: str | None
    should_ask: bool
    blocker_id: str | None
    n: int
    internal: dict  # all internal-divergence metrics
    output: dict  # output-divergence dict (B1)
    verbalized: float  # B2 ask-score
    probe: float = 0.0  # B3 dp ask-score (filled after OOF probe training)
    correct: list = field(default_factory=list)  # per-trajectory correctness


@dataclass
class RunRecords:
    records: list[DPRecord]
    config: dict = field(default_factory=dict)

    def signal_columns(self) -> dict[str, list[float]]:
        """Flatten the per-dp scalar ask-scores for every signal -> {signal: [values]}."""
        cols: dict[str, list[float]] = {
            "internal_mean_pairwise_cosine": [],
            "internal_total_variance": [],
            "internal_eigenscore": [],
            "internal_stiefel_volume": [],
            "output_divergence_b1": [],
            "verbalized_b2": [],
            "probe_b3": [],
        }
        for r in self.records:
            cols["internal_mean_pairwise_cosine"].append(r.internal["mean_pairwise_cosine"])
            cols["internal_total_variance"].append(r.internal["total_variance"])
            cols["internal_eigenscore"].append(r.internal["eigenscore"])
            cols["internal_stiefel_volume"].append(r.internal["stiefel_volume"])
            cols["output_divergence_b1"].append(r.output["dispersion"])
            cols["verbalized_b2"].append(r.verbalized)
            cols["probe_b3"].append(r.probe)
        return cols

    def labels(self) -> dict[str, list]:
        return {
            "regime": [r.regime for r in self.records],
            "should_ask": [r.should_ask for r in self.records],
            "task_id": [r.task_id for r in self.records],
            "step": [r.step for r in self.records],
        }


def _correct_flags(obs: DPObservation) -> list[bool]:
    return [bool(g.extra.get("correct", True)) for g in obs.generations]


def build_records(
    trajectories_by_task: dict[str, list[list[StepResult]]],
    tasks: list[Task],
    executor: Executor,
    *,
    scheme: str = "step_index_then_anchor",
    probe_splits: int = 5,
    seed: int = 0,
    config: dict | None = None,
) -> RunRecords:
    """Assemble all per-dp records, including the leakage-free B3 probe score."""
    tasks_by_id = {t.instance_id: t for t in tasks}

    # First pass: compute alignment + internal/output/verbalized signals, gather hiddens.
    records: list[DPRecord] = []
    H_all: list[np.ndarray] = []
    incorrect_all: list[int] = []
    sample_owner: list[tuple[int, int]] = []  # (record_index, member_position)

    for task_id, trajs in trajectories_by_task.items():
        task = tasks_by_id[task_id]
        for obs in build_dp_observations(trajs, task, scheme):
            signatures = [executor.signature(g, task=task, dp=obs) for g in obs.generations]
            correct = _correct_flags(obs)
            rec = DPRecord(
                task_id=obs.task_id,
                anchor=obs.anchor,
                step=obs.step,
                regime=obs.regime,
                should_ask=obs.should_ask,
                blocker_id=obs.blocker_id,
                n=obs.n,
                internal=idiv.compute_all(obs.hiddens),
                output=output_divergence(signatures),
                verbalized=verbalized_ask_score(obs.generations),
                correct=correct,
            )
            ridx = len(records)
            records.append(rec)
            for pos, (vec, c) in enumerate(zip(obs.hiddens, correct)):
                H_all.append(vec)
                incorrect_all.append(int(not c))
                sample_owner.append((ridx, pos))

    # Second pass: train the correctness probe out-of-fold, aggregate to dp ask-scores.
    if H_all:
        oof = oof_incorrect_proba(np.array(H_all), np.array(incorrect_all), probe_splits, seed)
        per_record: dict[int, list[float]] = {}
        for (ridx, _pos), score in zip(sample_owner, oof):
            per_record.setdefault(ridx, []).append(float(score))
        for ridx, scores in per_record.items():
            records[ridx].probe = dp_ask_score(scores)

    return RunRecords(records=records, config=config or {})


def save_records(run: RunRecords, out_dir: str | Path) -> Path:
    """Write a JSON summary (no raw hidden arrays) for inspection."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "records.json"
    path.write_text(
        json.dumps(
            {"config": run.config, "records": [asdict(r) for r in run.records]},
            indent=2,
            default=float,
        )
    )
    return path
