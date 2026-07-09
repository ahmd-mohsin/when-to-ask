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
    # Resolved layer indices captured per read, in the order stacked into a 2-D
    # `h` of shape (L, H). None = legacy single-layer logs (h is 1-D, from
    # `mid_layer`). See decisions/014: capture-all on disk, select-one-at-load.
    layers: list[int] | None = None

    def validate(self) -> None:
        keys = [(r.segment_idx, r.token_idx) for r in self.reads]
        if any(b <= a for a, b in zip(keys, keys[1:])):
            raise ValueError(
                f"{self.run_id}: read (segment_idx, token_idx) not strictly increasing"
            )
        dims = {r.h.shape for r in self.reads}
        if len(dims) > 1:
            raise ValueError(f"{self.run_id}: inconsistent h shapes {dims}")
        n_layers = len(self.layers) if self.layers else None
        for r in self.reads:
            if r.h.ndim not in (1, 2):
                raise ValueError(f"{self.run_id}: h must be 1-D or 2-D, got {r.h.shape}")
            if r.h.ndim == 2:
                if n_layers is None:
                    raise ValueError(f"{self.run_id}: 2-D h needs `layers` set")
                if r.h.shape[0] != n_layers:
                    raise ValueError(
                        f"{self.run_id}: 2-D h layer axis {r.h.shape[0]} != "
                        f"len(layers) {n_layers}"
                    )
            if r.trigger not in ("cadence", "cue"):
                raise ValueError(f"{self.run_id}: unknown trigger {r.trigger!r}")
            if (r.trigger == "cue") != (r.cue is not None):
                raise ValueError(
                    f"{self.run_id}: cue must be set iff trigger == 'cue' "
                    f"(got trigger={r.trigger!r}, cue={r.cue!r})"
                )

    def read_matrix(self) -> np.ndarray:
        """(R, H) single-layer or (R, L, H) multi-layer float16; empty (0, 0)
        if no reads."""
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


def resolve_layer_pos(layers: list[int] | None, layer, mid_layer: int) -> int:
    """Which position along a stored (R, L, H) layer axis to slice.

    `layer` may be: None (default -> the position of `mid_layer` in `layers`,
    else the middle of the list), an int that IS one of the stored layer
    indices, or a small int treated as a direct position into `layers`.
    """
    if not layers:
        return 0
    if layer is None:
        return layers.index(mid_layer) if mid_layer in layers else len(layers) // 2
    if layer in layers:
        return layers.index(layer)
    if 0 <= layer < len(layers):
        return int(layer)
    raise ValueError(f"layer {layer} not in stored layers {layers} (nor a valid position)")


def load_run_log(out_dir: str | Path, run_id: str, layer=None) -> RunLog:
    """Load a run log. Multi-layer logs (on-disk (R, L, H)) are sliced to a
    single (R, H) layer here (`select-at-load`), so every caller downstream
    still sees 1-D per-read `h`. `layer` selects which (see resolve_layer_pos);
    ignored for legacy single-layer logs."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / f"{run_id}.json").read_text(encoding="utf-8"))
    h = np.load(out_dir / f"{run_id}.npz")["h"]
    reads_meta = meta.pop("reads")
    actions_meta = meta.pop("actions")
    layers = meta.get("layers")
    if h.ndim == 3:  # (R, L, H) -> pick one layer
        pos = resolve_layer_pos(layers, layer, meta.get("mid_layer"))
        h = h[:, pos, :]
    if len(reads_meta) != h.shape[0]:
        raise ValueError(
            f"{run_id}: {len(reads_meta)} read records vs {h.shape[0]} h rows"
        )
    # `layers` is retained in meta for provenance, but the in-memory reads are
    # now single-layer -> drop it so validate() takes the 1-D path.
    meta["layers"] = None
    log = RunLog(
        **meta,
        reads=[ReadRecord(h=h[i], **rm) for i, rm in enumerate(reads_meta)],
        actions=[ActionEvent(**am) for am in actions_meta],
    )
    log.validate()
    return log
