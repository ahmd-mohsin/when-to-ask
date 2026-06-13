"""Separation analysis -- OURS. Half of the C1 test.

For each signal, how well does it separate should-ask decision points from
fine-to-proceed ones? We report AUROC overall and, crucially, sliced by regime
(brief S5e): the headline is the **fork-vs-clear** slice, where internal cross-trajectory
divergence must beat output divergence (B1); the **confident_wrong-vs-clear** slice is
where the single-stream probe (B3) is expected to win and internal divergence to be at
chance (the honest limitation).

All signals are oriented so higher = more reason to ask, so AUROC uses the raw score.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from ..harness.tasks import CLEAR, CONFIDENT_WRONG, FORK
from ..recording.recorder import RunRecords


def _auroc(scores: list[float], positive: list[bool]) -> float | None:
    y = np.asarray(positive, dtype=int)
    if len(np.unique(y)) < 2:
        return None  # only one class present in this slice
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def _slice(records, scores, pos_regime: str):
    """Keep records whose regime is `pos_regime` (positive) or CLEAR (negative)."""
    s, y = [], []
    for r, sc in zip(records, scores):
        if r.regime == pos_regime:
            s.append(sc); y.append(True)
        elif r.regime == CLEAR:
            s.append(sc); y.append(False)
    return s, y


def separation_table(run: RunRecords) -> dict[str, dict[str, float | None]]:
    """{signal -> {overall, fork_vs_clear, confident_wrong_vs_clear} AUROC}."""
    cols = run.signal_columns()
    records = run.records
    should_ask = [r.should_ask for r in records]
    table: dict[str, dict[str, float | None]] = {}
    for signal, scores in cols.items():
        fork_s, fork_y = _slice(records, scores, FORK)
        cw_s, cw_y = _slice(records, scores, CONFIDENT_WRONG)
        table[signal] = {
            "overall": _auroc(scores, should_ask),
            "fork_vs_clear": _auroc(fork_s, fork_y),
            "confident_wrong_vs_clear": _auroc(cw_s, cw_y),
        }
    return table


# Signals grouped for the C1 read-out.
INTERNAL_PRIMARY = "internal_mean_pairwise_cosine"
OUTPUT_B1 = "output_divergence_b1"
PROBE_B3 = "probe_b3"


def c1_verdict(table: dict[str, dict[str, float | None]]) -> dict:
    """Summarise the make-or-break comparison on the fork slice (brief S7).

    C1 (separation half): internal primary AUROC on fork-vs-clear > output (B1) AUROC.
    Also reports whether B3 wins on confident-convergent (expected complementarity).
    """
    internal = table[INTERNAL_PRIMARY]["fork_vs_clear"]
    output = table[OUTPUT_B1]["fork_vs_clear"]
    probe_cw = table[PROBE_B3]["confident_wrong_vs_clear"]
    internal_cw = table[INTERNAL_PRIMARY]["confident_wrong_vs_clear"]
    beats_b1 = internal is not None and output is not None and internal > output
    return {
        "internal_fork_auroc": internal,
        "output_b1_fork_auroc": output,
        "internal_beats_b1_on_fork": beats_b1,
        "probe_confident_wrong_auroc": probe_cw,
        "internal_confident_wrong_auroc": internal_cw,
        "probe_covers_blind_spot": (
            probe_cw is not None
            and (internal_cw is None or probe_cw > internal_cw)
        ),
    }
