"""Contract: read-position policy (spec A0). Reads happen across the reasoning
span -- cadence + cues, cue spans token boundaries, never boundary-only."""

import pytest

from wta.reads import StreamReadSelector, read_positions


def test_cadence_positions():
    hits = read_positions(["tok "] * 100, cadence=10, cues=())
    assert [h.token_idx for h in hits] == [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    assert all(h.trigger == "cadence" for h in hits)


def test_cue_single_token():
    hits = read_positions(["I ", "will ", "wait", " here"], cadence=1000)
    assert [(h.token_idx, h.cue) for h in hits] == [(2, "wait")]


def test_cue_spans_token_boundary():
    hits = read_positions(["so ", "let", " me", " think"], cadence=1000)
    assert [(h.token_idx, h.cue) for h in hits] == [(2, "let me")]


def test_cue_not_inside_words():
    # "awaits" contains "wait", "hmmm" ends with "hmm": neither is a cue.
    assert read_positions(["awaits ", "hmmm ", "actual "], cadence=1000) == []


def test_cue_with_trailing_punctuation():
    hits = read_positions(["Hmm", ", ", "yes"], cadence=1000)
    assert [(h.token_idx, h.cue) for h in hits] == [(0, "hmm")]
    # single token carrying cue + punctuation
    hits = read_positions(["Wait,", " no"], cadence=1000)
    assert [(h.token_idx, h.cue) for h in hits] == [(0, "wait")]


def test_punctuation_does_not_refire():
    hits = read_positions(["wait", ",", "!", " "], cadence=1000)
    assert len(hits) == 1


def test_cue_fires_after_buffer_saturation():
    """The rolling text buffer saturates long before the cue arrives (as it
    always will in real generation); the cue must still fire. Guards against
    using buffer length as the 'new text arrived' signal."""
    toks = [f"filler{i} " for i in range(300)] + ["wait", " ok"]
    hits = read_positions(toks, cadence=10**6)
    assert [(h.token_idx, h.cue) for h in hits] == [(300, "wait")]


def test_cue_priority_over_cadence():
    hits = read_positions(["a ", "b ", "wait ", "c "], cadence=3)
    # position 2 is both the 3rd token (cadence) and a cue -> recorded as cue.
    assert [(h.token_idx, h.trigger) for h in hits] == [(2, "cue")]


def test_one_read_per_position_strictly_increasing():
    toks = ["hmm ", "wait ", "x "] * 40
    hits = read_positions(toks, cadence=5)
    idxs = [h.token_idx for h in hits]
    assert idxs == sorted(set(idxs))


def test_case_insensitive():
    hits = read_positions(["Actually", " so"], cadence=1000)
    assert [(h.token_idx, h.cue) for h in hits] == [(0, "actually")]


def test_config_validation():
    with pytest.raises(ValueError):
        StreamReadSelector(cadence=0)
    with pytest.raises(ValueError):
        StreamReadSelector(cues=("",))
