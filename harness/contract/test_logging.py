"""Contract: trajectory log round-trip and invariants (spec A0)."""

import numpy as np
import pytest

from wta.logging_schema import ActionEvent, ReadRecord, RunLog, load_run_log, save_run_log


def _make_log(run_id="r0", n_reads=6, H=16):
    rng = np.random.default_rng(1)
    reads = [
        ReadRecord(token_idx=5 * i + 3, trigger="cue" if i % 2 else "cadence",
                   cue="wait" if i % 2 else None,
                   h=rng.standard_normal(H).astype(np.float16))
        for i in range(n_reads)
    ]
    actions = [ActionEvent(token_idx=40, action_text="edit foo.py",
                           observables={"file": "foo.py", "region": "retry()",
                                        "subgoal": "add retry", "error_signature": None})]
    return RunLog(run_id=run_id, task_id="t0", seed=0, temperature=0.8,
                  model_id="fake", mid_layer=4, reads=reads, actions=actions)


def test_round_trip(tmp_path):
    log = _make_log()
    save_run_log(log, tmp_path)
    back = load_run_log(tmp_path, "r0")
    assert back.task_id == log.task_id and back.seed == log.seed
    assert back.mid_layer == log.mid_layer and back.temperature == log.temperature
    assert back.model_id == log.model_id
    assert [(r.segment_idx, r.token_idx, r.trigger, r.cue) for r in back.reads] == [
        (r.segment_idx, r.token_idx, r.trigger, r.cue) for r in log.reads
    ]
    assert np.array_equal(back.read_matrix(), log.read_matrix())
    assert back.actions[0].token_idx == log.actions[0].token_idx
    assert back.actions[0].action_text == log.actions[0].action_text
    assert back.actions[0].observables == log.actions[0].observables


def test_json_contains_no_h(tmp_path):
    """Residuals live in the npz only -- the JSON is human-inspectable metadata."""
    save_run_log(_make_log(run_id="r1"), tmp_path)
    assert '"h"' not in (tmp_path / "r1.json").read_text(encoding="utf-8")


def test_non_increasing_token_idx_rejected(tmp_path):
    log = _make_log(run_id="r2")
    log.reads[1] = ReadRecord(token_idx=log.reads[0].token_idx, trigger="cadence",
                              cue=None, h=log.reads[1].h)
    with pytest.raises(ValueError):
        save_run_log(log, tmp_path)


def test_token_idx_may_restart_across_segments(tmp_path):
    """A real agent run is several generate() calls; token_idx restarts per
    segment and ordering is on (segment_idx, token_idx)."""
    log = _make_log(run_id="r5", n_reads=3)
    log.reads.append(ReadRecord(token_idx=0, trigger="cadence", cue=None,
                                h=log.reads[0].h, segment_idx=1))
    save_run_log(log, tmp_path)
    back = load_run_log(tmp_path, "r5")
    assert back.reads[-1].segment_idx == 1 and back.reads[-1].token_idx == 0


def test_cue_field_consistency_rejected(tmp_path):
    log = _make_log(run_id="r6")
    log.reads[0] = ReadRecord(token_idx=log.reads[0].token_idx, trigger="cadence",
                              cue="wait", h=log.reads[0].h)  # cue on a cadence read
    with pytest.raises(ValueError):
        save_run_log(log, tmp_path)


def test_inconsistent_h_shape_rejected(tmp_path):
    log = _make_log(run_id="r3")
    log.reads[2] = ReadRecord(token_idx=log.reads[2].token_idx, trigger="cadence",
                              cue=None, h=np.zeros(7, dtype=np.float16))
    with pytest.raises(ValueError):
        save_run_log(log, tmp_path)


def test_empty_log_round_trips(tmp_path):
    log = RunLog(run_id="r4", task_id="t0", seed=1, temperature=0.7,
                 model_id="fake", mid_layer=2)
    save_run_log(log, tmp_path)
    back = load_run_log(tmp_path, "r4")
    assert back.reads == [] and back.actions == []


def _make_multilayer_log(run_id="ml", n_reads=6, H=16, layers=(11, 14, 17, 20)):
    rng = np.random.default_rng(2)
    reads = [
        ReadRecord(token_idx=5 * i + 3, trigger="cadence", cue=None,
                   h=rng.standard_normal((len(layers), H)).astype(np.float16))
        for i in range(n_reads)
    ]
    return RunLog(run_id=run_id, task_id="t0", seed=0, temperature=0.8,
                  model_id="fake", mid_layer=14, reads=reads, layers=list(layers))


def test_multilayer_capture_and_select_at_load(tmp_path):
    """decisions/014: store (R, L, H) on disk; load_run_log(layer=) slices to a
    single (R, H) so every downstream caller keeps 1-D per-read h."""
    log = _make_multilayer_log()
    save_run_log(log, tmp_path)
    disk = np.load(tmp_path / "ml.npz")["h"]
    assert disk.shape == (6, 4, 16)  # (R, L, H) on disk

    back = load_run_log(tmp_path, "ml", layer=17)  # 17 is layers[2]
    assert back.reads[0].h.shape == (16,)  # sliced to 1-D
    assert np.array_equal(back.reads[0].h, disk[0, 2, :])
    # default layer -> the stored mid_layer (14 == layers[1])
    assert np.array_equal(load_run_log(tmp_path, "ml").reads[0].h, disk[0, 1, :])
    # position index also accepted
    assert np.array_equal(load_run_log(tmp_path, "ml", layer=0).reads[0].h, disk[0, 0, :])


def test_multilayer_validate_layer_axis_mismatch():
    from wta.logging_schema import RunLog, ReadRecord
    bad = RunLog(run_id="b", task_id="t", seed=0, temperature=0.7, model_id="m",
                 mid_layer=14, layers=[11, 14, 17],  # says 3 layers
                 reads=[ReadRecord(token_idx=1, trigger="cadence", cue=None,
                                   h=np.zeros((4, 8), dtype=np.float16))])  # but 4
    with pytest.raises(ValueError):
        bad.validate()


def test_resolve_layers_pure():
    """Layer-spec resolution is unit-testable without torch (decisions/014)."""
    from wta.hf_reader import resolve_layers
    assert resolve_layers(28, [0.4, 0.5, 0.6, 0.7]) == [11, 14, 17, 20]
    assert resolve_layers(28, [14, 14, 0.5]) == [14]  # dedup + sort
    from wta.logging_schema import resolve_layer_pos
    assert resolve_layer_pos([11, 14, 17, 20], None, 14) == 1
    assert resolve_layer_pos([11, 14, 17, 20], 20, 14) == 3
    assert resolve_layer_pos(None, None, 14) == 0
