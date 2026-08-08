"""Load frozen offline artifacts into a live detector runtime (spec eval,
decisions/022).

The offline stages leave four artifacts in a model dir (train_offline.py /
run_full_gates.py): ``a1_direction.npy``, ``a2.pt``, ``a3_calibration.npz``,
``gate_report.json``. This module is the ONLY place they are read for online
use, and the ONLY place the l_scale asymmetry lives: raw ``r`` goes to
``CommitmentDetector.step`` (it divides by l_scale internally), ``r/l_scale``
goes to ``AskTrigger.observe`` (it does not) -- the run_pipeline_smoke.py
replay-loop contract, now in one audited spot.

Two artifact dialects exist on disk and both must load (decisions/022):
  * a3_calibration.npz -- train_offline writes {..., benign_n_pairs};
    run_full_gates writes {..., eps_settle}. Common core is required.
  * gate_report.json -- gate3 numbers stored FLAT ({"theta": ...}) by
    run_full_gates, or nested under "numbers" (in-memory GateResult shape).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wta.a1_direction import ambiguity_signal
from wta.a2_autoencoder import A2Model
from wta.a3_commitment import CommitmentDetector
from wta.online import AskDecision, AskTrigger, TriggerConfig

_A3_REQUIRED = ("tau", "l_scale", "s_ref", "window", "benign_reference")


@dataclass
class DetectorArtifacts:
    direction: np.ndarray        # (H,) unit ambiguity direction (A1)
    a2: A2Model                  # frozen encoders, CPU-capable
    tau: float
    l_scale: float
    s_ref: float
    window: int
    benign_reference: float
    theta: float                 # bucketing threshold (A4 gate 3)
    meta: dict = field(default_factory=dict)


def _load_theta(report_path: Path) -> float:
    report = json.loads(report_path.read_text())
    g3 = report.get("gate3_fork_collocation")
    if g3 is None:
        raise KeyError(f"{report_path}: no gate3_fork_collocation entry")
    if "theta" in g3:
        return float(g3["theta"])
    if "numbers" in g3 and "theta" in g3["numbers"]:
        return float(g3["numbers"]["theta"])
    raise KeyError(f"{report_path}: gate3 entry carries no theta (keys={list(g3)})")


def load_artifacts(model_dir: str | Path) -> DetectorArtifacts:
    model_dir = Path(model_dir)
    direction = np.load(model_dir / "a1_direction.npy").astype(np.float64)
    a2 = A2Model.load(model_dir / "a2.pt")
    if int(a2.cfg.in_dim) != direction.shape[0]:
        raise ValueError(
            f"artifact mismatch: a1 direction dim {direction.shape[0]} != "
            f"a2 in_dim {a2.cfg.in_dim} ({model_dir})")

    with np.load(model_dir / "a3_calibration.npz") as z:
        missing = [k for k in _A3_REQUIRED if k not in z.files]
        if missing:
            raise KeyError(f"{model_dir}/a3_calibration.npz missing {missing}")
        a3 = {k: z[k].item() for k in z.files}

    benign = float(a3["benign_reference"])
    if not np.isfinite(benign) or benign <= 0:
        raise ValueError(
            f"benign_reference={benign!r} unusable as CUSUM reference ({model_dir})")

    return DetectorArtifacts(
        direction=direction, a2=a2,
        tau=float(a3["tau"]), l_scale=float(a3["l_scale"]),
        s_ref=float(a3["s_ref"]), window=int(a3["window"]),
        benign_reference=benign,
        theta=_load_theta(model_dir / "gate_report.json"),
        meta={"model_dir": str(model_dir),
              "a3_extra": {k: a3[k] for k in a3 if k not in _A3_REQUIRED}},
    )


class TaskDetector:
    """The live Part B state for ONE (task, pass): fresh trigger + per-run
    commitment detectors (within-task bucketing, decisions/018 gate-4)."""

    def __init__(self, art: DetectorArtifacts, cfg: TriggerConfig,
                 layer_pos: int | None = None):
        self.art = art
        self.trigger = AskTrigger(cfg)
        self._layer_pos = layer_pos
        self._dets: dict = {}

    def _slice(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float32)
        if h.ndim == 2:                       # (L, H) multi-layer read
            if self._layer_pos is None:
                raise ValueError("multi-layer h but no layer_pos configured")
            h = h[self._layer_pos]
        if h.shape != (self.art.direction.shape[0],):
            raise ValueError(f"read h shape {h.shape} != (H={self.art.direction.shape[0]},)")
        return h

    def observe_read(self, run_id, h: np.ndarray) -> AskDecision | None:
        h = self._slice(h)
        r = self.art.a2.encode_lean(h[None])[0]
        t = self.art.a2.encode_topic(h[None])[0]
        s = float(ambiguity_signal(h.astype(np.float64), self.art.direction))
        det = self._dets.get(run_id)
        if det is None:
            det = self._dets[run_id] = CommitmentDetector(
                tau=self.art.tau, s_ref=self.art.s_ref,
                window=self.art.window, l_scale=self.art.l_scale)
        _, w = det.step(r, s)                       # raw r: step() scales internally
        return self.trigger.observe(run_id, t, r / self.art.l_scale, s, w)

    # side channels, passthrough
    def register_action(self, run_id, action_text: str) -> None:
        self.trigger.register_action(run_id, action_text)

    def notify_env_state(self, run_id, state_hash) -> None:
        self.trigger.notify_env_state(run_id, state_hash)

    def inject_resolution(self, bucket_id: int) -> None:
        self.trigger.inject_resolution(bucket_id)


class DetectorRuntime:
    """Frozen-artifact factory for TaskDetectors.

    ``trigger_overrides`` exists for fixture smokes ONLY; with ``sealed=True``
    any override is refused (nothing is re-tuned on eval tasks -- specs/eval.md).
    """

    def __init__(self, art: DetectorArtifacts, *, layer_pos: int | None = None,
                 trigger_overrides: dict | None = None, sealed: bool = False):
        if sealed and trigger_overrides:
            raise ValueError("sealed run: trigger overrides are not allowed")
        self.art = art
        self._layer_pos = layer_pos
        kw = dict(theta=art.theta, reference=art.benign_reference,
                  slack=0.1 * art.benign_reference)
        kw.update(trigger_overrides or {})
        self._cfg = TriggerConfig(**kw)
        self.trigger_config = self._cfg

    def new_task(self) -> TaskDetector:
        return TaskDetector(self.art, self._cfg, layer_pos=self._layer_pos)
