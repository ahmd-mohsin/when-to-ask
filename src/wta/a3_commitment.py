"""A3: commitment = ``r`` steady + ``s`` dropped; threshold ``tau`` by split
conformal (spec A3).

MIGRATED: ``conformal_quantile`` is a faithful port of LYNX
(third_party/LYNX @ e62c6c2, ``lynx/utils.py::conformal_quantile``) -- the
finite-sample ceil((n+1)(1-delta)) order statistic. We borrow LYNX's
calibration *procedure*, NOT their correctness target: our calibration label
is the stabilization point, observable from the trajectory itself (the method
doc's reason this stays computable at forks).

OURS: the commitment definition (instability of ``r`` over a window,
corroborated by the s-gate), the stabilization-point label, and the streaming
detector. Numpy only -- runs online without torch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Migrated: LYNX's split-conformal quantile
# ---------------------------------------------------------------------------


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    """Finite-sample split conformal quantile (LYNX utils.py:339, verbatim).

    scores are nonconformity values; larger = more nonconforming.
    Returns q such that P(score <= q) >= 1 - delta for a fresh exchangeable
    score.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = int(len(scores))
    if n <= 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - delta)))
    k = max(1, min(k, n))
    return float(np.sort(scores)[k - 1])


# ---------------------------------------------------------------------------
# Ours: instability, stabilization label, calibration, streaming detector
# ---------------------------------------------------------------------------


def instability(r_seq: np.ndarray, window: int) -> np.ndarray:
    """Per-read instability: max ||r_j - r_k|| over the last ``window`` reads
    (j in (k-window, k]). +inf where fewer than ``window`` reads exist -- a
    run cannot look committed before it has a history."""
    r = np.asarray(r_seq, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"expected (R, d_lean), got {r.shape}")
    n = len(r)
    out = np.full(n, np.inf)
    for k in range(window - 1, n):
        seg = r[k - window + 1 : k + 1]
        out[k] = float(np.linalg.norm(seg - r[k], axis=1).max())
    return out


def stabilization_point(r_seq: np.ndarray, eps_settle: float,
                        min_tail: int = 1) -> int | None:
    """Earliest k* such that every later read stays within ``eps_settle`` of
    the final lean. None if the run only 'settles' inside its last
    ``min_tail`` reads (i.e. never really settled -- loop-channel material,
    not calibration material)."""
    r = np.asarray(r_seq, dtype=np.float64)
    d_end = np.linalg.norm(r - r[-1], axis=1)
    beyond = np.where(d_end > eps_settle)[0]
    k_star = 0 if len(beyond) == 0 else int(beyond[-1]) + 1
    if k_star > len(r) - min_tail:
        return None
    return k_star


@dataclass
class CalibrationReport:
    tau: float
    delta: float
    n_points: int
    n_skipped_never_settled: int
    n_skipped_short_history: int
    scores: np.ndarray
    l_scale: float = 1.0


def lean_scale(r_seqs: list[np.ndarray]) -> float:
    """Global L-space scale: RMS distance of reads from their sequence mean.

    The learned lean space has ARBITRARY scale (nothing in A2 pins it), so
    absolute thresholds -- eps_settle, tau, the CUSUM reference -- are
    meaningless in raw units (measured: on one fixture 23/24 calibration
    sequences were skipped and clear decisions fired the trigger). All A3/B
    thresholds live in units of this scale; per-bucket refinement stays the
    Phase-2 sweep (decisions/010)."""
    sq, n = 0.0, 0
    for r in r_seqs:
        r = np.asarray(r, dtype=np.float64)
        d2 = ((r - r.mean(axis=0)) ** 2).sum(axis=1)
        sq += float(d2.sum())
        n += len(r)
    return float(np.sqrt(sq / max(n, 1))) or 1.0


def calibrate_tau(r_seqs: list[np.ndarray], window: int, eps_settle: float,
                  delta: float = 0.1) -> CalibrationReport:
    """One nonconformity score per (run, decision): instability measured at the
    true stabilization point (one point each -- exchangeability,
    decisions/010). tau = LYNX conformal quantile of those scores.

    eps_settle and the returned tau are in l_scale units (see lean_scale)."""
    scale = lean_scale(r_seqs)
    r_seqs = [np.asarray(r, dtype=np.float64) / scale for r in r_seqs]
    scores, never, short = [], 0, 0
    for r in r_seqs:
        k = stabilization_point(r, eps_settle, min_tail=window)
        if k is None:
            never += 1
            continue
        inst = instability(r, window)
        # Score the first read whose trailing window lies WHOLLY in the settled
        # regime (k* + w - 1), not k* itself: at k* the window still spans the
        # settle transition, so its instability measures the transition
        # magnitude -- calibrating tau on that would set the threshold an
        # order of magnitude too high and admit pre-settle commits.
        k_eff = max(k + window - 1, window - 1)
        if k_eff >= len(r):
            short += 1
            continue
        scores.append(inst[k_eff])
    arr = np.asarray(scores, dtype=np.float64)
    return CalibrationReport(
        tau=conformal_quantile(arr, delta), delta=delta, n_points=len(arr),
        n_skipped_never_settled=never, n_skipped_short_history=short, scores=arr,
        l_scale=scale,
    )


def benign_spread_reference(decision_groups: list[list[tuple[np.ndarray, int]]],
                            window: int, eps_settle: float,
                            quantile: float = 0.9) -> tuple[float, int]:
    """Part B's CUSUM reference, measured offline: the benign cross-run spread
    of committed leans among runs that settled on the SAME interpretation.

    decision_groups: per decision, a list of (r_seq, class_id) over runs.
    Returns (reference in l_scale units, n_pairs). Spread above this level is
    evidence of a fork; below is L-space noise. (Introduced 2026-07-03 after
    the pipeline smoke showed a hand-set reference below the learned-L noise
    floor makes clear decisions fire slowly; the reference is an offline
    teacher-derived artifact, like tau.)"""
    all_seqs = [r for grp in decision_groups for r, _ in grp]
    scale = lean_scale(all_seqs)
    dists = []
    for grp in decision_groups:
        settled = []
        for r, c in grp:
            r = np.asarray(r, dtype=np.float64) / scale
            k = stabilization_point(r, eps_settle, min_tail=window)
            if k is None:
                continue
            settled.append((r[min(k + window - 1, len(r) - 1):].mean(axis=0), c))
        for i in range(len(settled)):
            for j in range(i + 1, len(settled)):
                if settled[i][1] == settled[j][1]:
                    dists.append(float(np.linalg.norm(settled[i][0] - settled[j][0])))
    if not dists:
        return float("nan"), 0
    return float(np.quantile(dists, quantile)), len(dists)


def s_reference(s_should_ask: np.ndarray, s_proceed: np.ndarray) -> float:
    """The s-gate threshold: ROC-optimal (Youden J) crossover of the held-out
    should-ask vs proceed s distributions from A1."""
    from sklearn.metrics import roc_curve

    y = np.r_[np.ones(len(s_should_ask)), np.zeros(len(s_proceed))]
    fpr, tpr, thr = roc_curve(y, np.r_[s_should_ask, s_proceed])
    return float(thr[np.argmax(tpr - fpr)])


@dataclass
class CommitmentDetector:
    """Streaming commitment per (run, decision): no lookahead, no actions.

    committed  <=>  instability over the trailing window <= tau  AND  s <= s_ref
    weight     =    sigmoid(alpha * (tau - instability)) gated by s (soft --
                    Part B never hard-excludes a run).
    """

    tau: float
    s_ref: float
    window: int
    alpha: float = 8.0
    l_scale: float = 1.0  # from CalibrationReport; tau lives in these units
    _buf: deque = field(default_factory=deque, repr=False)

    def step(self, r: np.ndarray, s: float) -> tuple[bool, float]:
        self._buf.append(np.asarray(r, dtype=np.float64) / self.l_scale)
        if len(self._buf) > self.window:
            self._buf.popleft()
        if len(self._buf) < self.window:
            inst = np.inf
        else:
            cur = self._buf[-1]
            inst = max(float(np.linalg.norm(v - cur)) for v in self._buf)
        if s > self.s_ref or not np.isfinite(inst):
            return False, 0.0
        committed = inst <= self.tau
        weight = float(1.0 / (1.0 + np.exp(-self.alpha * (self.tau - inst))))
        return committed, weight

    def reset(self) -> None:
        self._buf.clear()
