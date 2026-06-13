"""Output divergence (B1, migrated from ClarifyGPT's identical-output grouping)."""

import pytest

from xtid.signals.output_divergence import output_divergence


def test_all_agree_is_unambiguous():
    r = output_divergence(["0", "0", "0", "0"])
    assert r["ambiguous"] is False and r["n_clusters"] == 1 and r["dispersion"] == 0.0 and r["entropy"] == 0.0


def test_split_is_ambiguous_with_dispersion_and_entropy():
    r = output_divergence(["0", "1", "2", "0"])
    assert r["ambiguous"] is True
    assert r["n_clusters"] == 3
    assert r["dispersion"] == pytest.approx(1 - 2 / 4)  # largest cluster has 2 of 4
    assert 0.0 < r["entropy"] <= 1.0


def test_even_split_maximises_normalised_entropy():
    assert output_divergence(["a", "b"])["entropy"] == pytest.approx(1.0)


def test_single_signature_not_ambiguous():
    assert output_divergence(["x"])["ambiguous"] is False
