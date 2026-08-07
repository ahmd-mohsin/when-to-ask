"""Contract: sampling params are passed EXPLICITLY, never inherited.

decisions/021 §1: `generate()` used to receive only do_sample+temperature, so
top_k/top_p came from the model's shipped generation_config. Qwen3 ships
top_k=20, which truncates the candidate set to 20 tokens and makes temperature
nearly inert as a DIVERSITY lever — the likely cause of the single-run-minority
forks in the 32B collection. These tests pin the contract without needing torch
or a GPU (they exercise the pure kwargs builder)."""

import pytest

from wta.hf_reader import HFStreamReader


@pytest.fixture
def reader():
    """An un-constructed reader: __init__ needs torch + weights, so bind only
    the attributes _sampling_kwargs reads."""
    r = HFStreamReader.__new__(HFStreamReader)
    r.top_p, r.top_k, r.min_p = 1.0, 0, 0.0
    return r


def test_defaults_disable_both_truncations(reader):
    kw = reader._sampling_kwargs(0.9)
    assert kw["top_k"] == 0, "top-k truncation must be disabled by default"
    assert kw["top_p"] == 1.0
    assert kw["min_p"] == 0.0
    assert kw["do_sample"] is True
    assert kw["temperature"] == 0.9


def test_none_means_inherit_explicitly(reader):
    """None is the opt-in to the model's own config — the key must be ABSENT
    so generate() falls back, rather than being passed as a literal None."""
    reader.top_k = None
    kw = reader._sampling_kwargs(0.9)
    assert "top_k" not in kw
    assert "top_p" in kw


def test_greedy_when_temperature_zero(reader):
    kw = reader._sampling_kwargs(0.0)
    assert kw["do_sample"] is False
    assert kw["temperature"] > 0, "temperature must stay positive for HF"


def test_overrides_are_recorded_for_the_manifest(reader):
    reader.model = None  # no generation_config available
    cfg = HFStreamReader.effective_generation_config(reader)
    assert cfg["overrides"] == {"top_p": 1.0, "top_k": 0, "min_p": 0.0}
