"""B2 (spec B2-behavioral-divergence, decisions/027): registry-free
behavioral features for the Part B trigger.

The trigger machinery is wta.online.AskTrigger VERBATIM; this module only
produces its inputs from what a run observably DID -- files touched, edit
regions, mutating-command text, error signatures, subgoal prose. It must
never read hidden states or the interpretation-class artifact (leak rule,
pinned by contract test).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from wta.labeling import _is_mutating, _norm
from wta.online import AskTrigger, TriggerConfig  # noqa: F401 (re-export for callers)

R_DIM = 512          # resolution-vector dim, hash seed 0
TOPIC_DIM = 256      # topic-vector dim, hash seed 1
TOPIC_WINDOW = 3     # turns of topic evidence accumulated


def signed_hash_vec(tokens: list[str], dim: int, seed: int) -> np.ndarray:
    """Deterministic signed-hash TF vector (sha1, NOT the salted builtin
    hash), l2-normalized; zero vector if no tokens."""
    v = np.zeros(dim, dtype=np.float64)
    for t in tokens:
        d = hashlib.sha1(f"{seed}|{t}".encode("utf-8")).digest()
        idx = int.from_bytes(d[:4], "little") % dim
        v[idx] += 1.0 if (d[4] & 1) else -1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _char3grams(s: str) -> list[str]:
    return [s[i:i + 3] for i in range(len(s) - 2)] if len(s) >= 3 else [s]


@dataclass
class TurnFeatures:
    round: int
    topic_vec: np.ndarray | None   # None -> run has no topic yet; skip observe
    r_vec: np.ndarray              # the run's CURRENT (sticky) vote vector
    weight: float                  # 0.0 pre-commitment, 1.0 after


def behavior_features(actions: list) -> list[TurnFeatures]:
    """Per-turn features for one run, from its ActionEvents in generation
    order. Sticky votes: weight flips to 1.0 at the first mutating action and
    stays; r_vec is the latest mutating action's vector (spec B2)."""
    ordered = sorted(actions, key=lambda a: a.segment_idx)
    feats: list[TurnFeatures] = []
    r_vec = np.zeros(R_DIM, dtype=np.float64)
    weight = 0.0
    topic_vec: np.ndarray | None = None
    turn_tokens: list[list[str]] = []
    for k, a in enumerate(ordered):
        obs = a.observables or {}
        if _is_mutating(a.action_text or ""):
            body = " ".join([_norm(a.action_text or ""),
                             " ".join(obs.get("region") or []),
                             str(obs.get("error_signature") or "")])
            r_vec = signed_hash_vec(_char3grams(body), R_DIM, seed=0)
            weight = 1.0
        toks = [w for f in (obs.get("files") or []) for w in _norm(str(f)).split()]
        toks += _norm(str(obs.get("subgoal") or "")).split()
        turn_tokens.append(toks)
        window = [t for turn in turn_tokens[-TOPIC_WINDOW:] for t in turn]
        if window:
            v = signed_hash_vec(window, TOPIC_DIM, seed=1)
            if float(np.linalg.norm(v)) > 0:
                topic_vec = v
        feats.append(TurnFeatures(round=k, topic_vec=topic_vec,
                                  r_vec=r_vec, weight=weight))
    return feats


@dataclass
class FireEvent:
    round: int
    bucket_id: int
    run_ids: list
    n_votes: int
    spread: float


def replay_task(runs: dict[str, list[TurnFeatures]],
                cfg: TriggerConfig) -> list[FireEvent]:
    """Offline lockstep replay (spec B2): round k feeds every run's turn k
    (runs shorter than k drop out); one observe per (run, turn) when the run
    has a topic. On fire, the resolution is 'injected' (bucket cleared) as
    the live orchestrator would after answering, and replay continues."""
    trig = AskTrigger(cfg)
    fires: list[FireEvent] = []
    n_rounds = max((len(f) for f in runs.values()), default=0)
    for k in range(n_rounds):
        for run_id in sorted(runs):
            feats = runs[run_id]
            if k >= len(feats):
                continue
            f = feats[k]
            if f.topic_vec is None:
                continue
            dec = trig.observe(run_id, f.topic_vec, f.r_vec,
                               s=0.0, weight=f.weight)
            if dec is not None:
                fires.append(FireEvent(round=k, bucket_id=dec.bucket_id,
                                       run_ids=list(dec.runs),
                                       n_votes=len(dec.options),
                                       spread=dec.spread))
                trig.inject_resolution(dec.bucket_id)
    return fires
