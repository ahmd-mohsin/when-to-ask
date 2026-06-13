"""Single-stream correctness probe (B3) -- MIGRATED from OPENIA.

Source idea: third_party/OPENIA (arXiv 2501.12934) -- a classifier on a model's internal
representations predicts whether generated code is correct, outperforming logprob ranking
and verbalized confidence. The Confidence Manifold (2602.08159) shows this signal is
low-dimensional and linearly separable, so a linear probe is the right tool.

We train a linear probe on mid-layer hidden states to predict *incorrectness*, per
trajectory. This is the "is your divergence just a probe run N times?" control (brief B3):
ours must add value over B3 on fork blockers, while B3 should (correctly) win on
confident-convergent blockers, where all N agree but are all wrong.

`oof_incorrect_proba` returns out-of-fold predictions so the separation analysis uses
honest (non-leaked) probe scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _make_clf() -> "LogisticRegression":
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))


class CorrectnessProbe:
    """Linear probe: mid-layer hidden state -> P(incorrect)."""

    def __init__(self) -> None:
        self.clf = _make_clf()
        self.constant: float | None = None  # set if only one class is present

    def fit(self, H: np.ndarray, incorrect: np.ndarray) -> "CorrectnessProbe":
        H = np.asarray(H, dtype=np.float64)
        y = np.asarray(incorrect, dtype=int)
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean()) if len(y) else 0.0
        else:
            self.clf.fit(H, y)
            self.constant = None
        return self

    def proba_incorrect(self, H: np.ndarray) -> np.ndarray:
        H = np.asarray(H, dtype=np.float64)
        if self.constant is not None:
            return np.full(H.shape[0], self.constant)
        return self.clf.predict_proba(H)[:, 1]


def oof_incorrect_proba(
    H: np.ndarray, incorrect: np.ndarray, n_splits: int = 5, seed: int = 0
) -> np.ndarray:
    """Out-of-fold P(incorrect) for each sample (honest, leakage-free probe scores)."""
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(incorrect, dtype=int)
    out = np.full(len(y), float(y.mean()) if len(y) else 0.0)
    if len(np.unique(y)) < 2:
        return out
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(H, y):
        clf = _make_clf().fit(H[tr], y[tr])
        out[te] = clf.predict_proba(H[te])[:, 1]
    return out


def dp_ask_score(member_scores: list[float] | np.ndarray, agg: str = "mean") -> float:
    """Aggregate the N per-trajectory P(incorrect) into a decision-point ask-score."""
    s = np.asarray(member_scores, dtype=np.float64)
    if s.size == 0:
        return 0.0
    return float(s.max()) if agg == "max" else float(s.mean())
