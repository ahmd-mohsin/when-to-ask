"""Single-stream correctness probe (B3, migrated from OPENIA)."""

import numpy as np
from sklearn.metrics import roc_auc_score

from xtid.signals.probe import CorrectnessProbe, dp_ask_score, oof_incorrect_proba


def _two_clusters(n=60, dim=16, seed=0):
    """Correct vs incorrect = two separated Gaussian blobs -> a linear probe should win."""
    rng = np.random.default_rng(seed)
    correct = rng.standard_normal((n, dim))
    incorrect = rng.standard_normal((n, dim)) + 3.0
    H = np.vstack([correct, incorrect])
    y = np.array([0] * n + [1] * n)  # 1 = incorrect
    return H, y


def test_probe_separates_correct_from_incorrect():
    H, y = _two_clusters()
    probe = CorrectnessProbe().fit(H, y)
    assert roc_auc_score(y, probe.proba_incorrect(H)) > 0.95


def test_oof_predictions_are_honest_and_separable():
    H, y = _two_clusters(seed=1)
    oof = oof_incorrect_proba(H, y, n_splits=5)
    assert oof.shape == (len(y),)
    assert roc_auc_score(y, oof) > 0.9  # holds out of fold, so not leakage


def test_single_class_falls_back_to_constant():
    H = np.random.default_rng(0).standard_normal((10, 8))
    probe = CorrectnessProbe().fit(H, np.zeros(10, dtype=int))
    assert np.allclose(probe.proba_incorrect(H), 0.0)
    assert np.allclose(oof_incorrect_proba(H, np.ones(10, dtype=int)), 1.0)


def test_dp_ask_score_aggregations():
    assert dp_ask_score([0.1, 0.9]) == 0.5
    assert dp_ask_score([0.1, 0.9], agg="max") == 0.9
    assert dp_ask_score([]) == 0.0
