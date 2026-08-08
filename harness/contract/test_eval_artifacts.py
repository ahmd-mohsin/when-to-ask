"""Contract: offline artifacts load into a ready DetectorRuntime (spec eval,
decisions/022).

Pins: (1) the real 14B artifact set loads with its exact recorded values;
(2) both a3 key dialects and both gate-report theta shapes load; (3) the
l_scale asymmetry -- raw r to CommitmentDetector.step, r/l_scale to
AskTrigger.observe (the run_pipeline_smoke replay contract); (4) sealed runs
refuse trigger overrides.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wta.eval.artifacts import DetectorRuntime, load_artifacts  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ART_DIR = REPO / "models" / "v2_14b_gates_single"

pytestmark = pytest.mark.skipif(
    not (ART_DIR / "a2.pt").exists(),
    reason="models/v2_14b_gates_single artifact set not present")


def test_real_14b_artifacts_load_with_recorded_values():
    art = load_artifacts(ART_DIR)
    assert art.tau == pytest.approx(0.8914, abs=1e-3)
    assert art.l_scale == pytest.approx(5.7286, abs=1e-3)
    assert art.s_ref == pytest.approx(21.051, abs=1e-2)
    assert art.window == 3
    assert art.benign_reference == pytest.approx(0.5948, abs=1e-3)
    assert art.theta == pytest.approx(0.70499, abs=1e-4)
    assert art.direction.shape == (art.a2.cfg.in_dim,)
    assert np.isfinite(art.direction).all()
    # the run_full_gates dialect carries eps_settle in the extras
    assert "eps_settle" in art.meta["a3_extra"]


def _clone_with(tmp_path, a3_extra_key, theta_nested):
    """Synthesize the OTHER on-disk dialect from the real artifact set."""
    art_src = ART_DIR
    out = tmp_path / "variant"
    out.mkdir()
    (out / "a1_direction.npy").write_bytes((art_src / "a1_direction.npy").read_bytes())
    (out / "a2.pt").write_bytes((art_src / "a2.pt").read_bytes())
    with np.load(art_src / "a3_calibration.npz") as z:
        core = {k: z[k] for k in ("tau", "l_scale", "s_ref", "window",
                                  "benign_reference")}
    core[a3_extra_key] = np.float64(7.0)
    np.savez(out / "a3_calibration.npz", **core)
    theta = json.loads((art_src / "gate_report.json").read_text())[
        "gate3_fork_collocation"]["theta"]
    g3 = {"numbers": {"theta": theta}} if theta_nested else {"theta": theta}
    (out / "gate_report.json").write_text(
        json.dumps({"gate3_fork_collocation": g3}))
    return out


def test_train_offline_dialect_and_nested_theta_load(tmp_path):
    out = _clone_with(tmp_path, a3_extra_key="benign_n_pairs", theta_nested=True)
    art = load_artifacts(out)
    assert art.theta == pytest.approx(0.70499, abs=1e-4)
    assert art.meta["a3_extra"] == {"benign_n_pairs": 7.0}


def test_missing_a3_core_key_raises(tmp_path):
    out = _clone_with(tmp_path, a3_extra_key="eps_settle", theta_nested=False)
    with np.load(out / "a3_calibration.npz") as z:
        partial = {k: z[k] for k in z.files if k != "benign_reference"}
    np.savez(out / "a3_calibration.npz", **partial)
    with pytest.raises(KeyError, match="benign_reference"):
        load_artifacts(out)


def test_scaling_contract_raw_to_step_scaled_to_observe(monkeypatch):
    art = load_artifacts(ART_DIR)
    runtime = DetectorRuntime(art)
    task = runtime.new_task()

    seen = {}
    from wta import a3_commitment, online

    orig_step = a3_commitment.CommitmentDetector.step
    orig_observe = online.AskTrigger.observe

    def spy_step(self, r, s):
        seen["step_r"] = np.array(r, dtype=np.float64)
        return orig_step(self, r, s)

    def spy_observe(self, run_id, topic_vec, r_vec, s, weight):
        seen["observe_r"] = np.array(r_vec, dtype=np.float64)
        return orig_observe(self, run_id, topic_vec, r_vec, s, weight)

    monkeypatch.setattr(a3_commitment.CommitmentDetector, "step", spy_step)
    monkeypatch.setattr(online.AskTrigger, "observe", spy_observe)

    h = np.random.default_rng(0).normal(size=art.direction.shape[0]).astype(np.float32)
    task.observe_read("run-a", h)

    expected_raw = art.a2.encode_lean(h[None])[0]
    np.testing.assert_allclose(seen["step_r"], expected_raw, rtol=1e-5)
    np.testing.assert_allclose(seen["observe_r"], expected_raw / art.l_scale,
                               rtol=1e-5)


def test_trigger_config_from_artifacts_and_sealed_refusal():
    art = load_artifacts(ART_DIR)
    rt = DetectorRuntime(art)
    assert rt.trigger_config.theta == pytest.approx(art.theta)
    assert rt.trigger_config.reference == pytest.approx(art.benign_reference)
    assert rt.trigger_config.slack == pytest.approx(0.1 * art.benign_reference)
    with pytest.raises(ValueError, match="sealed"):
        DetectorRuntime(art, trigger_overrides={"h_threshold": 1.0}, sealed=True)
    # non-sealed override is allowed (smoke-only escape hatch)
    rt2 = DetectorRuntime(art, trigger_overrides={"h_threshold": 1.0})
    assert rt2.trigger_config.h_threshold == 1.0


def test_multilayer_read_requires_layer_pos():
    art = load_artifacts(ART_DIR)
    task = DetectorRuntime(art).new_task()
    h2 = np.zeros((4, art.direction.shape[0]), dtype=np.float32)
    with pytest.raises(ValueError, match="layer_pos"):
        task.observe_read("r0", h2)
    task2 = DetectorRuntime(art, layer_pos=1).new_task()
    assert task2.observe_read("r0", h2) is None  # slices and runs
