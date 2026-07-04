"""A1: ambiguity direction `d` and scalar signal `s = dot(h, d)` (spec A1).

MIGRATED mechanism -- difference-in-means / mass-mean direction:
  * CAA, `generate_vectors.py`: `vec = (all_pos_layer - all_neg_layer).mean(dim=0)`
    (nrimsky/CAA @ 5dabbbd, MIT; third_party/CAA/PROVENANCE.md)
  * Geometry of Truth, `probes.py::MMProbe`:
    `direction = pos_acts.mean(0) - neg_acts.mean(0)`
    (saprmarks/geometry-of-truth @ 5d1c630; third_party/geometry-of-truth/PROVENANCE.md)
Ported to numpy behind our interface; unpaired class means (equivalent to
CAA's paired mean for balanced pairs). Normalization to unit length is ours
(so `s` is comparable across layers/configs).

`s` gates and times commitment. It is NEVER the disagreement trigger -- the
trigger fires on spread of `r = L(h)` within a topic bucket (method doc,
"Three signals, named precisely").
"""

from __future__ import annotations

import numpy as np


def build_direction(h_should_ask: np.ndarray, h_proceed: np.ndarray) -> np.ndarray:
    """`d = normalize(mean(should-ask) - mean(proceed))` -> float32[H]."""
    pos = np.asarray(h_should_ask, dtype=np.float64)
    neg = np.asarray(h_proceed, dtype=np.float64)
    if pos.ndim != 2 or neg.ndim != 2:
        raise ValueError(f"expected 2-D (n, H) arrays, got {pos.shape} and {neg.shape}")
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise ValueError("both classes must be non-empty")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(f"hidden-dim mismatch: {pos.shape[1]} vs {neg.shape[1]}")
    if not (np.isfinite(pos).all() and np.isfinite(neg).all()):
        raise ValueError("non-finite values in inputs")
    d = pos.mean(axis=0) - neg.mean(axis=0)
    n = np.linalg.norm(d)
    if n < 1e-12:
        raise ValueError("class means coincide; no direction (check labels)")
    return (d / n).astype(np.float32)


def ambiguity_signal(h: np.ndarray, d: np.ndarray) -> np.ndarray | float:
    """`s = dot(h, d)`; batched over leading dims. Scalar in -> scalar out."""
    h = np.asarray(h, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    if h.shape[-1] != d.shape[-1]:
        raise ValueError(f"hidden-dim mismatch: {h.shape[-1]} vs {d.shape[-1]}")
    s = h @ d
    return float(s) if s.ndim == 0 else s


def auroc(s_should_ask: np.ndarray, s_proceed: np.ndarray) -> float:
    """AUROC of `s` separating should-ask (positive) from proceed (negative).

    Delegates to scikit-learn (already a project dependency); tie-aware.
    """
    from sklearn.metrics import roc_auc_score

    s_pos = np.asarray(s_should_ask).ravel()
    s_neg = np.asarray(s_proceed).ravel()
    if s_pos.size == 0 or s_neg.size == 0:
        raise ValueError("both classes must be non-empty")
    y = np.concatenate([np.ones(s_pos.size), np.zeros(s_neg.size)])
    return float(roc_auc_score(y, np.concatenate([s_pos, s_neg])))
