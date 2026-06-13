"""Cross-trajectory INTERNAL divergence -- OURS (the candidate winning signal).

At an aligned decision point we have N mid-layer hidden vectors H (shape N x dim). The
signal is their *dispersion*. Reading dispersion ACROSS parallel agent trajectories at an
aligned decision point is the novel part (brief contribution 1); the individual
dispersion measures include two migrated ones:

  * mean_pairwise_cosine -- PRIMARY (brief S5c). Mean over pairs of (1 - cos).
  * total_variance       -- trace of the covariance = mean sq. distance to the centroid.
  * eigenscore           -- MIGRATED from INSIDE/EigenScore (eigenscore/func/metric.py):
                            mean(log10(svd(cov(H)+alpha I))). cov is taken across the N
                            rows exactly as upstream (torch.cov of (num_seq, dim)).
  * stiefel_volume       -- reimplemented from STARS (arXiv 2601.22010): the geometric
                            log-volume spanned by the centred vectors (STARS *maximises*
                            this to inject diversity; we *read* it to detect divergence).

Higher = more divergent for every metric. N<2 -> 0.
"""

from __future__ import annotations

import numpy as np

METRICS = ("mean_pairwise_cosine", "total_variance", "eigenscore", "stiefel_volume")
PRIMARY = "mean_pairwise_cosine"


def _as_matrix(H) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    return H.reshape(1, -1) if H.ndim == 1 else H


def mean_pairwise_cosine(H) -> float:
    """Mean pairwise cosine *distance* across the N vectors (the primary metric)."""
    H = _as_matrix(H)
    n = H.shape[0]
    if n < 2:
        return 0.0
    norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    sims = norm @ norm.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sims[iu]))


def total_variance(H) -> float:
    """Trace of the covariance across the N vectors (mean sq. dist. to centroid)."""
    H = _as_matrix(H)
    if H.shape[0] < 2:
        return 0.0
    return float(np.sum(np.var(H, axis=0)))


def eigenscore(H, alpha: float = 1e-3) -> float:
    """MIGRATED EigenScore: mean(log10(svd(cov(H)+alpha I))), cov across the N rows.

    Mirrors getEigenScore in third_party/eigenscore/func/metric.py (np.cov treats rows as
    variables, giving the N x N covariance over trajectories).
    """
    H = _as_matrix(H)
    if H.shape[0] < 2:
        return 0.0
    cov = np.cov(H)  # (N, N): each row (trajectory) is a variable
    cov = np.atleast_2d(cov)
    s = np.linalg.svd(cov + alpha * np.eye(cov.shape[0]), compute_uv=False)
    return float(np.mean(np.log10(s)))


def stiefel_volume(H, eps: float = 1e-8) -> float:
    """Reimplemented STARS geometric volume: log-volume spanned by the centred vectors.

    vol = sum(log(sigma_i)) over the non-negligible singular values of the centred H.
    Larger volume = the N internal states occupy more of the space = more divergent.
    """
    H = _as_matrix(H)
    if H.shape[0] < 2:
        return 0.0
    Hc = H - H.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Hc, compute_uv=False)
    s = s[s > eps]
    return float(np.sum(np.log(s))) if s.size else 0.0


_FUNCS = {
    "mean_pairwise_cosine": mean_pairwise_cosine,
    "total_variance": total_variance,
    "eigenscore": eigenscore,
    "stiefel_volume": stiefel_volume,
}


def divergence(H, metric: str = PRIMARY) -> float:
    if metric not in _FUNCS:
        raise ValueError(f"unknown internal-divergence metric: {metric!r}")
    return _FUNCS[metric](H)


def compute_all(H) -> dict[str, float]:
    return {m: _FUNCS[m](H) for m in METRICS}
