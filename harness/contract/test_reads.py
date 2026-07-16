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


def test_value_reads_fire_on_literals_with_cooldown():
    from wta.reads import DEFAULT_VALUE_PATTERN

    sel_kw = dict(cadence=10**6, cues=(), value_pattern=DEFAULT_VALUE_PATTERN,
                  value_cooldown=8)
    # fires on a multi-digit literal, records the matched text in cue
    sel = StreamReadSelector(**sel_kw)
    hits = [sel.step(t) for t in ["timeout", " = ", "30", " seconds"]]
    fired = [h for h in hits if h]
    assert [(h.token_idx, h.trigger, h.cue) for h in fired] == [(2, "value", "30")]

    # cooldown: a burst of literals yields one read per cooldown window
    sel = StreamReadSelector(**sel_kw)
    hits = [sel.step(f"{n} ") for n in range(100, 120)]  # 20 literals in a row
    fired = [h for h in hits if h and h.trigger == "value"]
    assert len(fired) == 3  # positions 0, 8, 16
    # single digits do NOT fire (loop indices would flood otherwise)
    sel = StreamReadSelector(**sel_kw)
    assert all(sel.step(t) is None for t in ["i = 1", "j = 2", "k = 3"])


def test_value_reads_fire_across_token_boundaries():
    """Qwen-family tokenizers emit numbers digit by digit ('120' -> '1','2',
    '0'), so no single token ever contains >= 2 digits. The literal must fire
    the moment it becomes matchable in the stream (here: the 2nd digit), not
    require the whole literal inside one token -- the 14B v2 collection
    produced ZERO value reads before this was fixed."""
    from wta.reads import DEFAULT_VALUE_PATTERN

    sel = StreamReadSelector(cadence=10**6, cues=(),
                             value_pattern=DEFAULT_VALUE_PATTERN,
                             value_cooldown=8)
    hits = [sel.step(t) for t in ["timeout", " ", "1", "2", "0", " ok"]]
    fired = [h for h in hits if h]
    assert [(h.token_idx, h.trigger, h.cue) for h in fired] == [(3, "value", "12")]
    # a literal ending mid-token still fires exactly once
    sel = StreamReadSelector(cadence=10**6, cues=(),
                             value_pattern=DEFAULT_VALUE_PATTERN,
                             value_cooldown=8)
    hits = [sel.step(t) for t in ["port ", "80", "80,", " done"]]
    fired = [h for h in hits if h]
    assert [(h.token_idx, h.trigger) for h in fired] == [(1, "value")]


def test_value_reads_off_by_default_and_cue_priority():
    sel = StreamReadSelector(cadence=10**6, cues=())
    assert all(sel.step(t) is None for t in ["x = ", "3000"])  # no pattern -> off
    from wta.reads import DEFAULT_VALUE_PATTERN
    sel = StreamReadSelector(cadence=10**6, value_pattern=DEFAULT_VALUE_PATTERN)
    hit = sel.step("30 wait")  # cue (at end) and value in one token -> cue wins
    assert hit.trigger == "cue" and hit.cue == "wait"
