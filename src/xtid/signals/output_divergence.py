"""Cross-trajectory OUTPUT divergence (B1) -- MIGRATED from ClarifyGPT.

Source: third_party/ClarifyGPT/.../run_clarify_chatgpt_mbpp.py::runTests_getTaskID.
ClarifyGPT samples N solutions, runs them on shared test inputs, groups by the output
signature `str(test_result)`, and treats `len(groups) > 1` as ambiguous -> ask. This is
the exact baseline the whole claim rests on (brief B1), with the *output* signal where
ours puts the *internal* one.

Here the per-trajectory signatures come from `harness.executor` (synthetic: the chosen
interpretation; real: a hash of the test-output vector). We report the same decision plus
continuous dispersion scores for AUROC.
"""

from __future__ import annotations

import math
from collections import Counter


def output_divergence(signatures: list[str]) -> dict:
    """Cluster output signatures (ClarifyGPT grouping) and score divergence.

    Returns:
      n_clusters  -- number of distinct output signatures.
      ambiguous   -- n_clusters > 1 (ClarifyGPT's ask trigger).
      dispersion  -- 1 - (largest cluster / N)  in [0, 1).
      entropy     -- normalised Shannon entropy of cluster sizes in [0, 1].
    """
    n = len(signatures)
    if n < 2:
        return {"n_clusters": max(n, 0), "ambiguous": False, "dispersion": 0.0, "entropy": 0.0}
    counts = Counter(signatures)
    n_clusters = len(counts)
    largest = max(counts.values())
    dispersion = 1.0 - largest / n
    if n_clusters > 1:
        probs = [c / n for c in counts.values()]
        entropy = -sum(p * math.log(p) for p in probs) / math.log(n_clusters)
    else:
        entropy = 0.0
    return {
        "n_clusters": n_clusters,
        "ambiguous": n_clusters > 1,
        "dispersion": float(dispersion),
        "entropy": float(entropy),
    }
