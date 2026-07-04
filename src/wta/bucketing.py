"""Topic bucketing: leader clustering + merge + hysteresis -- OURS (spec B).

Buckets are keyed by the frozen topic vector T(h) and NOTHING else -- never a
step index, never an action (the ground rule). Async-safe by construction: a
slow run's read lands in the right bucket whenever it arrives because its
topic is resolution-invariant.

Two known failure modes of bare nearest-bucket assignment, both handled here
exactly as the method doc prescribes:
- order-dependence: greedy leader clustering can seed one decision as two
  buckets depending on arrival order -> the merge pass collapses centroids
  that come within theta of each other, making the final bucketing
  order-invariant in practice;
- within-run fragmentation: one run's topic wobbles across reads and spawns a
  spurious second bucket -> hysteresis holds a run in its bucket unless its
  topic leaves theta for M consecutive reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-norm topic vector")
    return v / n


@dataclass
class Vote:
    r: np.ndarray
    weight: float
    seq: int  # monotone stamp; on merge, a run keeps its most recent vote


@dataclass
class Bucket:
    bucket_id: int
    centroid: np.ndarray  # unit-norm running mean direction
    n_updates: int = 1
    votes: dict = field(default_factory=dict)  # run_id -> Vote


class LeaderClusters:
    """Streaming leader clustering over topic vectors with hysteresis + merge."""

    def __init__(self, theta: float, hysteresis_m: int = 3,
                 merge_theta: float | None = None):
        if not -1.0 < theta < 1.0:
            raise ValueError("theta must be a cosine in (-1, 1)")
        self.theta = theta
        self.hysteresis_m = max(1, hysteresis_m)
        self.merge_theta = theta if merge_theta is None else merge_theta
        self.buckets: dict[int, Bucket] = {}
        self._assign: dict = {}       # run_id -> bucket_id
        self._out_count: dict = {}    # run_id -> consecutive out-of-theta reads
        self._next_id = 0

    # -- internals ----------------------------------------------------------

    def _new_bucket(self, v: np.ndarray) -> Bucket:
        b = Bucket(bucket_id=self._next_id, centroid=v)
        self.buckets[self._next_id] = b
        self._next_id += 1
        return b

    def _update_centroid(self, b: Bucket, v: np.ndarray) -> None:
        c = b.centroid * b.n_updates + v
        b.centroid = _unit(c)
        b.n_updates += 1

    def _nearest(self, v: np.ndarray, exclude: int | None = None):
        best_id, best_cos = None, -2.0
        for bid, b in self.buckets.items():
            if bid == exclude:
                continue
            c = float(v @ b.centroid)
            if c > best_cos:
                best_id, best_cos = bid, c
        return best_id, best_cos

    def _merge_pass(self) -> None:
        """Collapse any centroid pair within merge_theta (repeat to fixpoint)."""
        changed = True
        while changed:
            changed = False
            ids = sorted(self.buckets)
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    ba, bb = self.buckets[a], self.buckets[b]
                    if float(ba.centroid @ bb.centroid) >= self.merge_theta:
                        self._merge(ba, bb)
                        changed = True
                        break
                if changed:
                    break

    def _merge(self, keep: Bucket, gone: Bucket) -> None:
        w = keep.centroid * keep.n_updates + gone.centroid * gone.n_updates
        keep.centroid = _unit(w)
        keep.n_updates += gone.n_updates
        for run_id, vote in gone.votes.items():
            cur = keep.votes.get(run_id)
            if cur is None or vote.seq > cur.seq:  # most recent vote wins
                keep.votes[run_id] = vote
        for run_id, bid in list(self._assign.items()):
            if bid == gone.bucket_id:
                self._assign[run_id] = keep.bucket_id
        del self.buckets[gone.bucket_id]

    # -- API ----------------------------------------------------------------

    def assign(self, run_id, topic_vec: np.ndarray) -> int:
        """Assign this read's topic to a bucket; returns the bucket id."""
        v = _unit(topic_vec)
        cur_id = self._assign.get(run_id)

        if cur_id is not None and cur_id in self.buckets:
            cur = self.buckets[cur_id]
            if float(v @ cur.centroid) >= self.theta:
                self._out_count[run_id] = 0
                self._update_centroid(cur, v)
                self._merge_pass()
                return self._assign[run_id]  # merge may have re-pointed it
            self._out_count[run_id] = self._out_count.get(run_id, 0) + 1
            if self._out_count[run_id] < self.hysteresis_m:
                return cur_id  # hysteresis: hold (no centroid pollution)

        best_id, best_cos = self._nearest(v)
        if best_id is not None and best_cos >= self.theta:
            target = self.buckets[best_id]
            self._update_centroid(target, v)
        else:
            target = self._new_bucket(v)
        self._assign[run_id] = target.bucket_id
        self._out_count[run_id] = 0
        self._merge_pass()
        return self._assign[run_id]

    def runs_in(self, bucket_id: int) -> list:
        return [r for r, b in self._assign.items() if b == bucket_id]


def leader_cluster_points(points: np.ndarray, theta: float) -> np.ndarray:
    """Batch helper (used by A4 gate 6): cluster row vectors, return labels.
    Each point is its own 'run', so hysteresis is inert; merge still applies."""
    lc = LeaderClusters(theta=theta, hysteresis_m=1)
    labels = [lc.assign(f"p{i}", p) for i, p in enumerate(np.asarray(points))]
    # merges may have re-pointed earlier assignments
    return np.array([lc._assign[f"p{i}"] for i in range(len(labels))])
