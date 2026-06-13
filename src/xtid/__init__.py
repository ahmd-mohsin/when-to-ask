"""xtid -- Cross-Trajectory Internal Divergence.

De-risk experiment for claim C1 of the pre-implementation brief:

    At matched N, on the fork-blocker slice, does *internal* cross-trajectory
    divergence separate "should-have-asked" from "fine-to-proceed" better than
    *output* cross-trajectory divergence (ClarifyGPT / B1), and does it fire
    earlier (positive lead-time)?

Package layout:
    harness/    MIGRATED from HiL-Bench: tasks, judge, Ask-F1, executor.
    backbone/   OURS: white-box model wrapper exposing mid-layer hidden states.
    agent/      OURS (base from mini-swe-agent): N-trajectory agent loop.
    signals/    The divergence signals: alignment + internal (ours) + B1/B2/B3.
    recording/  OURS: per-decision-point, per-trajectory record logging.
    analysis/   OURS: regime labels, AUROC separation, lead-time -> the C1 table.
"""

__version__ = "0.1.0"
