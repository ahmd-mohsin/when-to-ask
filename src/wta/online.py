"""Part B: the online When-to-Ask trigger -- OURS, the contribution (spec B).

Per live run, per read, the caller supplies ONLY frozen forward-pass products:
topic ``T(h)``, lean ``r = L(h)``, ambiguity ``s``, and the A3 commitment
weight. Nothing here trains, decodes, reconstructs, or looks at actions/step
indices -- actions enter exactly once, as optional TEXT for assembling the
question after the trigger has already fired (the ground rule).

Trigger: per bucket with >= min_votes committed votes, the commitment-weighted
dispersion of the ``r`` vectors feeds an online CUSUM; fire only on
PERSISTENT dispersion. A spread that collapses as runs keep reasoning was a
recoverable wobble (mutable votes have updated by then) -- don't interrupt.
Runs stuck in a loop never commit and thus never vote; their repeated
environment states feed a separate loop channel so a thrashing run still
counts: ``D_total = D_disagreement + lambda_loop * D_loop``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from wta.bucketing import LeaderClusters, Vote


@dataclass
class TriggerConfig:
    theta: float                 # bucketing cosine threshold (from A4 gate 3)
    hysteresis_m: int = 3
    merge_theta: float | None = None
    reference: float = 0.2       # CUSUM reference level (expected benign spread)
    slack: float = 0.05          # CUSUM slack
    # Fire level, in units of READ-EVENTS on the bucket: every observe of any
    # run assigned to the bucket pumps the CUSUM, so a spread lasting one
    # "round" contributes ~N increments at N runs. Swept in Phase 2 (the
    # per-bucket normalization open item); default sized so a 2-read blip at
    # N=2 (~4 pumps) stays under while a persistent fork fires in ~3 rounds.
    h_threshold: float = 6.0
    lambda_loop: float = 0.5
    min_votes: int = 2
    loop_repeat_floor: int = 2   # env-state repeats before a run counts as looping


@dataclass
class AskDecision:
    bucket_id: int
    runs: list
    options: list            # [{run_id, r, weight, action_text|None}] per vote
    looping_runs: list
    spread: float
    cusum: float


class AskTrigger:
    def __init__(self, cfg: TriggerConfig):
        self.cfg = cfg
        self.clusters = LeaderClusters(cfg.theta, cfg.hysteresis_m, cfg.merge_theta)
        self._cusum: dict[int, float] = {}
        self._env_counts: dict = {}       # run_id -> Counter(env_state_hash)
        self._last_action: dict = {}      # run_id -> str (question text ONLY)
        self._seq = 0

    # -- side channels (never triggers) -------------------------------------

    def register_action(self, run_id, action_text: str) -> None:
        """Latest committed action TEXT, kept only to phrase the question's
        options on fire. Never read by the trigger logic itself."""
        self._last_action[run_id] = action_text

    def notify_env_state(self, run_id, state_hash) -> None:
        """Environment-state signature for the loop channel (repeated states =
        a stuck run)."""
        self._env_counts.setdefault(run_id, Counter())[state_hash] += 1

    def _loop_score(self, run_id) -> float:
        counts = self._env_counts.get(run_id)
        if not counts:
            return 0.0
        return float(max(0, max(counts.values()) - self.cfg.loop_repeat_floor + 1))

    # -- the per-read update -------------------------------------------------

    def observe(self, run_id, topic_vec: np.ndarray, r_vec: np.ndarray,
                s: float, weight: float) -> AskDecision | None:
        """One read of one run. Returns an AskDecision when the CUSUM fires."""
        b_id = self.clusters.assign(run_id, topic_vec)
        bucket = self.clusters.buckets[b_id]

        self._seq += 1
        if weight > 0.0:
            bucket.votes[run_id] = Vote(r=np.asarray(r_vec, dtype=np.float64),
                                        weight=float(weight), seq=self._seq)
        else:
            bucket.votes.pop(run_id, None)  # de-committed -> retract

        spread = self._spread(bucket)
        loop = sum(self._loop_score(r) for r in self.clusters.runs_in(b_id))
        d_total = spread + self.cfg.lambda_loop * loop

        s_prev = self._cusum.get(b_id, 0.0)
        s_new = max(0.0, s_prev + d_total - self.cfg.reference - self.cfg.slack)
        self._cusum[b_id] = s_new

        if s_new > self.cfg.h_threshold:
            decision = AskDecision(
                bucket_id=b_id,
                runs=self.clusters.runs_in(b_id),
                options=[{"run_id": rid, "r": v.r, "weight": v.weight,
                          "action_text": self._last_action.get(rid)}
                         for rid, v in sorted(bucket.votes.items(), key=lambda kv: -kv[1].weight)],
                looping_runs=[r for r in self.clusters.runs_in(b_id)
                              if self._loop_score(r) > 0],
                spread=spread, cusum=s_new,
            )
            self._cusum[b_id] = 0.0  # fired; caller injects the answer
            return decision
        return None

    def _spread(self, bucket) -> float:
        """Commitment-weighted mean pairwise distance of the votes' r vectors."""
        votes = list(bucket.votes.values())
        if len(votes) < self.cfg.min_votes:
            return 0.0
        num, den = 0.0, 0.0
        for i in range(len(votes)):
            for j in range(i + 1, len(votes)):
                w = votes[i].weight * votes[j].weight
                num += w * float(np.linalg.norm(votes[i].r - votes[j].r))
                den += w
        return num / den if den > 0 else 0.0

    def inject_resolution(self, bucket_id: int) -> None:
        """The human answered: the decision is resolved for every run. Clear
        the bucket's votes and pressure; runs continue."""
        if bucket_id in self.clusters.buckets:
            self.clusters.buckets[bucket_id].votes.clear()
        self._cusum[bucket_id] = 0.0
