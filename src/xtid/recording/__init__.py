"""Recording -- OURS.

Per-decision-point, per-trajectory records written during a run: the aligned
mid-layer hidden state, the candidate action/output (for B1), the verbalized
confidence (B2), the probe score (B3), plus the gold blocker/regime label needed
by the offline C1 analysis.

(Named `recording` rather than `logging` to avoid shadowing the stdlib module.)
"""
