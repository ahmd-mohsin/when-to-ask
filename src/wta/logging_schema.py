"""Trajectory log schema -- OURS (spec A0).

One RunLog per (task, run). Residuals go to `<run_id>.npz` (float16 read
matrix); everything else -- read metadata, action events with observables --
goes to `<run_id>.json`. Observables are offline teachers for labels; nothing
at runtime reads them (ground rule).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ReadRecord:
    token_idx: int
    trigger: str  # "cadence" | "cue"
    cue: str | None
    h: np.ndarray  # float16[H]
    # A real agent run is several generate() calls; token_idx restarts at each,
    # so ordering is on (segment_idx, token_idx). Single-pass logs use 0.
    segment_idx: int = 0


@dataclass
class ActionEvent:
    token_idx: int
    action_text: str
    observables: dict = field(default_factory=dict)  # file / region / subgoal / error_signature
    segment_idx: int = 0


@dataclass
class RunLog:
    run_id: str
    task_id: str
    seed: int
    temperature: float
    model_id: str
    mid_layer: int
    reads: list[ReadRecord] = field(default_factory=list)
    actions: list[ActionEvent] = field(default_factory=list)

    def validate(self) -> None:
        keys = [(r.segment_idx, r.token_idx) for r in self.reads]
        if any(b <= a for a, b in zip(keys, keys[1:])):
            raise ValueError(
                f"{self.run_id}: read (segment_idx, token_idx) not strictly increasing"
            )
        dims = {r.h.shape for r in self.reads}
        if len(dims) > 1:
            raise ValueError(f"{self.run_id}: inconsistent h shapes {dims}")
        for r in self.reads:
            if r.h.ndim != 1:
                raise ValueError(f"{self.run_id}: h must be 1-D, got {r.h.shape}")
            if r.trigger not in ("cadence", "cue"):
                raise ValueError(f"{self.run_id}: unknown trigger {r.trigger!r}")
            if (r.trigger == "cue") != (r.cue is not None):
                raise ValueError(
                    f"{self.run_id}: cue must be set iff trigger == 'cue' "
                    f"(got trigger={r.trigger!r}, cue={r.cue!r})"
                )

    def read_matrix(self) -> np.ndarray:
        """(R, H) float16; empty (0, 0) if no reads."""
        if not self.reads:
            return np.zeros((0, 0), dtype=np.float16)
        return np.stack([r.h.astype(np.float16) for r in self.reads])


def save_run_log(log: RunLog, out_dir: str | Path) -> tuple[Path, Path]:
    log.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{log.run_id}.npz"
    json_path = out_dir / f"{log.run_id}.json"
    np.savez_compressed(npz_path, h=log.read_matrix())
    meta = asdict(log)
    for r in meta["reads"]:
        r.pop("h")
    json_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return npz_path, json_path


def load_run_log(out_dir: str | Path, run_id: str) -> RunLog:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / f"{run_id}.json").read_text(encoding="utf-8"))
    h = np.load(out_dir / f"{run_id}.npz")["h"]
    reads_meta = meta.pop("reads")
    actions_meta = meta.pop("actions")
    if len(reads_meta) != h.shape[0]:
        raise ValueError(
            f"{run_id}: {len(reads_meta)} read records vs {h.shape[0]} h rows"
        )
    log = RunLog(
        **meta,
        reads=[ReadRecord(h=h[i], **rm) for i, rm in enumerate(reads_meta)],
        actions=[ActionEvent(**am) for am in actions_meta],
    )
    log.validate()
    return log
