"""Lead-time analysis -- OURS. The temporal half of the C1 test (brief S5e / S7).

Does internal cross-trajectory divergence fire *earlier* than output divergence on fork
blockers? For each fork task we find the first decision point (by step) where the internal
primary signal rises noticeably above its clear-regime baseline, and likewise for output
divergence (B1). lead = first_output_cross_step - first_internal_cross_step.

A positive median lead is the temporal early-warning result the brief calls the most
defensible open ground. Thresholds are derived from the CLEAR-regime distribution so
"crossing" means "noticeably above what clear decision points look like".
"""

from __future__ import annotations

import numpy as np

from ..harness.tasks import CLEAR, FORK
from ..recording.recorder import DPRecord, RunRecords

INTERNAL_PRIMARY = "internal_mean_pairwise_cosine"
OUTPUT_B1 = "output_divergence_b1"


def _get(r: DPRecord, signal: str) -> float:
    if signal == OUTPUT_B1:
        return r.output["dispersion"]
    if signal.startswith("internal_"):
        return r.internal[signal[len("internal_") :]]
    raise ValueError(signal)


def _threshold(run: RunRecords, signal: str, q: float) -> float:
    clear_vals = [_get(r, signal) for r in run.records if r.regime == CLEAR]
    if not clear_vals:
        return 0.0
    return float(np.quantile(clear_vals, q))


def _first_cross(records: list[DPRecord], signal: str, tau: float) -> int | None:
    for r in sorted(records, key=lambda x: x.step):
        if _get(r, signal) > tau:
            return r.step
    return None


def lead_time_analysis(
    run: RunRecords,
    *,
    internal_signal: str = INTERNAL_PRIMARY,
    output_signal: str = OUTPUT_B1,
    q: float = 0.95,
) -> dict:
    """Per-fork-task internal-vs-output crossing lead-time."""
    tau_int = _threshold(run, internal_signal, q)
    tau_out = _threshold(run, output_signal, q)

    by_task: dict[str, list[DPRecord]] = {}
    for r in run.records:
        by_task.setdefault(r.task_id, []).append(r)

    leads: list[int] = []
    internal_only = 0  # internal crosses, output never does (B1 misses entirely)
    neither = 0
    for task_id, recs in by_task.items():
        if not any(r.regime == FORK for r in recs):
            continue
        i_cross = _first_cross(recs, internal_signal, tau_int)
        o_cross = _first_cross(recs, output_signal, tau_out)
        if i_cross is None and o_cross is None:
            neither += 1
        elif o_cross is None:
            internal_only += 1
        elif i_cross is not None:
            leads.append(o_cross - i_cross)

    return {
        "tau_internal": tau_int,
        "tau_output": tau_out,
        "n_fork_tasks": sum(1 for recs in by_task.values() if any(r.regime == FORK for r in recs)),
        "n_both_cross": len(leads),
        "internal_only_count": internal_only,  # B1 missed the fork; internal caught it
        "neither_count": neither,
        "median_lead": float(np.median(leads)) if leads else None,
        "mean_lead": float(np.mean(leads)) if leads else None,
        "fraction_positive_lead": (
            float(np.mean([l > 0 for l in leads])) if leads else None
        ),
    }
