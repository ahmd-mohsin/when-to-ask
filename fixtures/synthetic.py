"""Synthetic activation fixtures with KNOWN structure -- OURS.

Toy residual states h built from planted, mutually orthogonal directions so that
every later stage (A1 direction, A2 autoencoder, A3 commitment, B clustering /
voting / CUSUM) can be validated end-to-end on data where the right answer is
known, before any real model activations exist (build brief: "synthetic fixtures
first"; method doc PART A).

Planted generative model, per read k of run i facing decision (topic) t:

    h = topic_scale * u_t                                  (which decision)
      + lean_scale  * lean_k                               (which interpretation)
      + amb_scale   * (1 - p_k) * d_star   [ambiguous t]   (how unsettled)
      + noise

where u_t, v_{t,j} (interpretation j of decision t) and d_star are columns of one
orthonormal basis (exact ground truth, no accidental overlap), and p_k in [0,1]
is deliberation progress: p ramps to 1 at the run's settle read k*.

  * lean_k: before k*, a wobbling mixture of the decision's class vectors that
    approaches the run's eventual class j; from k* on, exactly v_{t,j}
    (plus optional blips, below). So "r steady + s dropped" == committed, by
    construction, and the internal signal leads any "action".
  * clear decisions: no d_star component ever, one shared class, p = 1 from the
    first read (nothing to deliberate).
  * fork guarantee: on ambiguous decisions classes are assigned round-robin over
    runs, so N >= 2 runs always disagree -- the confident fork: every run ends
    with s ~ 0 yet different v_{t,j}.
  * blip decisions (optional): after settling, runs briefly deviate to another
    class for `blip_len` reads and return -- the transient the CUSUM must NOT
    fire on.
  * loop runs (optional): p stays 0 and the same pre-settle state repeats (with
    tiny noise) -- never commits; the online loop channel's target.

Labels emitted are exactly the observables the offline teacher would produce:
decision identity, interpretation class, settle index, plus masks for
should-ask/proceed states (A1) and committed reads (A3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FixtureConfig:
    hidden_dim: int = 64
    n_topics: int = 6                 # decisions; first `n_ambiguous` are under-specified
    n_ambiguous: int = 3
    n_classes: int = 3                # interpretation classes per ambiguous decision
    n_runs: int = 8
    reads: int = 20                   # reads per run per decision
    topic_scale: float = 1.0
    lean_scale: float = 1.0
    amb_scale: float = 1.0
    noise: float = 0.08
    wobble_alpha: float = 0.5         # Dirichlet alpha for pre-settle lean swings
                                      # (spiky by default: deliberation lurches
                                      # between interpretations, it doesn't hover)
    settle_lo: float = 0.3            # settle read k* ~ U[settle_lo, settle_hi] * reads
    settle_hi: float = 0.7
    blip_topics: tuple[int, ...] = () # ambiguous topic ids that get a post-settle blip
    blip_at: float = 0.85             # blip start, as fraction of reads (after settle)
    blip_len: int = 2
    loop_runs: tuple[int, ...] = ()   # run ids that never commit (loop channel)
    action_delay: int = 4             # reads between lean-settle and the emitted
                                      # action (the planted lead-time window:
                                      # internal commitment precedes behaviour)
    seed: int = 0

    def validate(self) -> None:
        needed = 1 + self.n_topics + self.n_ambiguous * self.n_classes
        if needed > self.hidden_dim:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} < {needed} planted directions; raise hidden_dim"
            )
        if not 0 < self.n_ambiguous <= self.n_topics:
            raise ValueError("need 0 < n_ambiguous <= n_topics")
        if self.n_classes < 2:
            raise ValueError("need >= 2 interpretation classes for a fork to exist")
        if any(t >= self.n_ambiguous for t in self.blip_topics):
            raise ValueError("blip_topics must be ambiguous topic ids")


@dataclass
class SyntheticFixture:
    cfg: FixtureConfig
    d_star: np.ndarray            # (H,)          planted ambiguity direction
    topic_dirs: np.ndarray        # (T, H)        planted decision directions u_t
    lean_dirs: np.ndarray         # (Ta, C, H)    planted class directions v_{t,j} (ambiguous t only)
    h: np.ndarray                 # (N, T, R, H)  float32 reads
    progress: np.ndarray          # (N, T, R)     p_k in [0, 1]
    settle_idx: np.ndarray        # (N, T)        k*; reads >= k* are committed (-1: never)
    class_id: np.ndarray          # (N, T)        eventual interpretation (0 for clear topics)
    ambiguous: np.ndarray         # (T,) bool
    is_blip: np.ndarray = field(default=None)  # (N, T, R) bool, True during a blip

    # ---- teacher masks -----------------------------------------------------

    @property
    def action_read(self) -> np.ndarray:
        """(N, T): the read at which the run would emit its action --
        settle + action_delay, capped at the last read; -1 for never-settling
        (loop) runs. Offline teacher only (gate 7's behavioural reference)."""
        r = self.h.shape[2]
        acted = np.minimum(self.settle_idx + self.cfg.action_delay, r - 1)
        return np.where(self.settle_idx < 0, -1, acted)

    @property
    def committed(self) -> np.ndarray:
        """(N, T, R) bool: read k is at/after this run's settle point."""
        n, t, r, _ = self.h.shape
        k = np.arange(r)[None, None, :]
        settle = np.where(self.settle_idx < 0, r + 1, self.settle_idx)  # never-settle -> False
        return k >= settle[:, :, None]

    def should_ask_states(self) -> np.ndarray:
        """(M, H): reads facing an under-specified decision, pre-commitment (s high)."""
        mask = self.ambiguous[None, :, None] & ~self.committed
        return self.h[mask]

    def settled_states(self) -> np.ndarray:
        """(M, H): reads on ambiguous decisions at/after commitment.

        The matched negative class for BUILDING `d` (spec A1): same decisions
        as `should_ask_states`, so the decision-identity component cancels in
        the difference of means and only the ambiguity axis remains -- the
        CAA contrastive-pair analog. Contrasting should-ask against *clear*
        (different) decisions instead leaks decision identity into `d`
        (measured on this fixture: cos(d, d_star) 0.53 confounded vs 0.99
        matched).
        """
        mask = self.ambiguous[None, :, None] & self.committed
        return self.h[mask]

    def clear_states(self) -> np.ndarray:
        """(M, H): reads facing a specified (clear) decision -- nothing to ask."""
        mask = ~self.ambiguous[None, :, None] & np.ones_like(self.committed)
        return self.h[mask]

    def proceed_states(self) -> np.ndarray:
        """(M, H): everything that should score LOW on s -- settled ambiguous
        reads (the doc: a committed run "no longer looks ambiguous") plus all
        clear-decision reads. The evaluation-side negative class."""
        return np.concatenate([self.settled_states(), self.clear_states()])


def _orthonormal(dim: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """`count` exactly-orthonormal directions in R^dim (rows)."""
    m = rng.standard_normal((dim, count))
    q, _ = np.linalg.qr(m)
    return q[:, :count].T


def generate(cfg: FixtureConfig | None = None) -> SyntheticFixture:
    cfg = cfg or FixtureConfig()
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    H, T, Ta, C, N, R = (
        cfg.hidden_dim, cfg.n_topics, cfg.n_ambiguous, cfg.n_classes, cfg.n_runs, cfg.reads,
    )

    basis = _orthonormal(H, 1 + T + Ta * C, rng)
    d_star = basis[0]
    topic_dirs = basis[1 : 1 + T]
    lean_dirs = basis[1 + T :].reshape(Ta, C, H)

    ambiguous = np.arange(T) < Ta
    h = np.zeros((N, T, R, H), dtype=np.float32)
    progress = np.zeros((N, T, R), dtype=np.float32)
    settle_idx = np.zeros((N, T), dtype=np.int64)
    class_id = np.zeros((N, T), dtype=np.int64)
    is_blip = np.zeros((N, T, R), dtype=bool)

    for t in range(T):
        for i in range(N):
            if not ambiguous[t]:
                # Specified decision: settled from the start, one shared reading.
                settle_idx[i, t] = 0
                progress[i, t, :] = 1.0
                core = cfg.topic_scale * topic_dirs[t]
                h[i, t] = core + cfg.noise * rng.standard_normal((R, H))
                continue

            j = i % C  # round-robin -> guaranteed fork across runs
            class_id[i, t] = j
            if i in cfg.loop_runs:
                # Never settles: p stays 0, the same unsettled state repeats.
                settle_idx[i, t] = -1
                base = (
                    cfg.topic_scale * topic_dirs[t]
                    + cfg.amb_scale * d_star
                    + cfg.lean_scale * lean_dirs[t].mean(axis=0)
                )
                h[i, t] = base + cfg.noise * 0.3 * rng.standard_normal((R, H))
                continue

            k_star = int(rng.uniform(cfg.settle_lo, cfg.settle_hi) * R)
            k_star = max(1, min(R - 2, k_star))
            settle_idx[i, t] = k_star
            for k in range(R):
                p = min(1.0, k / k_star)
                progress[i, t, k] = p
                # Pre-settle lean: drift toward class j plus a CONSTANT-amplitude
                # wobble that stops dead at k* -- commitment is an event ("r has
                # STOPPED moving", method doc A3), not a smooth fade; a fading
                # wobble would make the lean stabilize well before the planted
                # settle point and the settle label would be wrong ground truth.
                if k < k_star:
                    w = rng.dirichlet(np.ones(C) * cfg.wobble_alpha)
                    drift = (1 - p) * lean_dirs[t].mean(axis=0) + p * lean_dirs[t, j]
                    mix = drift + (w - 1.0 / C) @ lean_dirs[t]
                else:
                    mix = lean_dirs[t, j]
                h[i, t, k] = (
                    cfg.topic_scale * topic_dirs[t]
                    + cfg.lean_scale * mix
                    + cfg.amb_scale * (1.0 - p) * d_star
                    + cfg.noise * rng.standard_normal(H)
                )
            if t in cfg.blip_topics:
                b0 = max(k_star + 1, int(cfg.blip_at * R))
                b1 = min(R, b0 + cfg.blip_len)
                alt = (j + 1) % C
                for k in range(b0, b1):
                    is_blip[i, t, k] = True
                    h[i, t, k] = (
                        cfg.topic_scale * topic_dirs[t]
                        + cfg.lean_scale * lean_dirs[t, alt]
                        + cfg.noise * rng.standard_normal(H)
                    )

    return SyntheticFixture(
        cfg=cfg, d_star=d_star, topic_dirs=topic_dirs, lean_dirs=lean_dirs,
        h=h, progress=progress, settle_idx=settle_idx, class_id=class_id,
        ambiguous=ambiguous, is_blip=is_blip,
    )
