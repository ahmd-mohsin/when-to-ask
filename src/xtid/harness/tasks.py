"""Task & blocker loading.

`Blocker` mirrors HiL-Bench's `BlockerEntry` (id / description / resolution /
example_questions / type). Two sources:

  * `synthetic`  -- generated tasks with planted decision-point regimes (fork /
                    confident_wrong / clear) and lead windows, for CPU smoke + unit
                    tests. Each decision point carries a `ctrl` marker consumed by
                    `FakeWhiteBoxModel`.
  * `hil_bench`  -- loads the real 200 public tasks + blocker registries from
                    third_party/hil-bench (harbor_swe / harbor_sql). Used on GPU.

For synthetic tasks the decision points (and thus the gold regime per step) are known
up front. For HiL-Bench the agent discovers decision points at run time; regime labels
come from blocker `type` via `analysis.regimes` (+ light manual annotation on the slice).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HIL_ROOT = REPO_ROOT / "third_party" / "hil-bench"

# The genuine-blocker regimes ("should-ask") vs. the proceed regime.
FORK = "fork"
CONFIDENT_WRONG = "confident_wrong"
CLEAR = "clear"
SHOULD_ASK_REGIMES = (FORK, CONFIDENT_WRONG)


@dataclass
class Blocker:
    """A gold blocker, mirroring HiL-Bench's BlockerEntry."""

    id: str
    description: str
    resolution: str
    example_questions: list[str] = field(default_factory=list)
    type: str | None = None  # missing_parameter / ambiguous_requirement / ...


@dataclass
class DecisionPoint:
    """One step of a task where the agent must commit to a choice."""

    index: int
    regime: str  # gold label: fork / confident_wrong / clear
    ctrl: str  # control marker fed to FakeWhiteBoxModel (synthetic only)
    should_ask: bool
    blocker: Blocker | None = None  # the gold blocker surfacing here, if any


@dataclass
class Task:
    instance_id: str
    domain: str  # synthetic / swe / sql
    statement: str
    source: str  # synthetic / hil_bench
    decision_points: list[DecisionPoint] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthetic tasks
# ---------------------------------------------------------------------------


def _fork_blocker(tid: str) -> Blocker:
    return Blocker(
        id=f"{tid}_fork",
        description="The requirement is ambiguous: two valid interpretations of the target behaviour.",
        resolution="Use interpretation 0 (the gold interpretation).",
        example_questions=["Which interpretation of the requirement should I implement?"],
        type="ambiguous_requirement",
    )


def _missing_blocker(tid: str) -> Blocker:
    return Blocker(
        id=f"{tid}_missing",
        description="A required parameter is unspecified and cannot be inferred from context.",
        resolution="Use the documented default value.",
        example_questions=["What value should the unspecified parameter take?"],
        type="missing_parameter",
    )


def make_synthetic_tasks(n: int = 12, seed: int = 0, n_dp: int = 4, k: int = 3) -> list[Task]:
    """Generate a balanced mix of fork / clear / confident_wrong tasks.

    Fork tasks plant a lead window: internal divergence becomes active one step before
    output divergence (so `analysis.lead_time` has a positive signal to recover).
    """
    rng = random.Random(seed)
    tasks: list[Task] = []
    kinds = [FORK, CLEAR, CONFIDENT_WRONG]
    for i in range(n):
        kind = kinds[i % 3]
        tid = f"syn_{kind}_{i:03d}"
        dps: list[DecisionPoint] = []
        blockers: list[Blocker] = []

        if kind == FORK:
            internal_from = rng.randint(1, max(1, n_dp - 2))
            lead = rng.choice([1, 2])
            output_from = min(n_dp - 1, internal_from + lead)
            gold = 0
            blk = _fork_blocker(tid)
            blockers.append(blk)
            for d in range(n_dp):
                active = d >= internal_from
                ctrl = (
                    f"[[CTRL regime=fork k={k} dp={d} internal_from={internal_from} "
                    f"output_from={output_from} gold={gold}]]"
                    if active
                    else f"[[CTRL regime=clear dp={d}]]"
                )
                dps.append(
                    DecisionPoint(
                        index=d,
                        regime=FORK if active else CLEAR,
                        ctrl=ctrl,
                        should_ask=active,
                        blocker=blk if d == internal_from else None,
                    )
                )
        elif kind == CONFIDENT_WRONG:
            cw_from = rng.randint(1, max(1, n_dp - 1))
            blk = _missing_blocker(tid)
            blockers.append(blk)
            for d in range(n_dp):
                active = d >= cw_from
                ctrl = (
                    f"[[CTRL regime=confident_wrong dp={d}]]"
                    if active
                    else f"[[CTRL regime=clear dp={d}]]"
                )
                dps.append(
                    DecisionPoint(
                        index=d,
                        regime=CONFIDENT_WRONG if active else CLEAR,
                        ctrl=ctrl,
                        should_ask=active,
                        blocker=blk if d == cw_from else None,
                    )
                )
        else:  # clear
            for d in range(n_dp):
                dps.append(
                    DecisionPoint(
                        index=d,
                        regime=CLEAR,
                        ctrl=f"[[CTRL regime=clear dp={d}]]",
                        should_ask=False,
                    )
                )

        tasks.append(
            Task(
                instance_id=tid,
                domain="synthetic",
                statement=f"Synthetic task {tid} ({kind}).",
                source="synthetic",
                decision_points=dps,
                blockers=blockers,
                meta={"kind": kind},
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# HiL-Bench tasks (real; used on GPU)
# ---------------------------------------------------------------------------


def _read_blocker_registry(path: Path) -> list[Blocker]:
    data = json.loads(path.read_text())
    out: list[Blocker] = []
    for b in data.get("blockers", []):
        eq = b.get("example_questions") or b.get("acceptable_questions") or b.get("trigger_questions") or []
        out.append(
            Blocker(
                id=b["id"],
                description=b.get("description", ""),
                resolution=b.get("resolution", ""),
                example_questions=list(eq),
                type=b.get("type"),
            )
        )
    return out


def load_hil_bench_tasks(domain: str = "swe", limit: int | None = None, root: Path | None = None) -> list[Task]:
    """Load HiL-Bench tasks from the vendored repo.

    Each task is a directory containing `metadata.json` and a blocker registry
    (`blocker_registry.json`, or `*_registry.json` for SQL, sometimes nested under
    `shared/ask-human-data/`). Decision points are *not* pre-populated -- they are
    discovered when the agent runs (Phase 3); regime labels come from blocker `type`.
    """
    root = root or HIL_ROOT
    harbor = root / ("harbor_sql" if domain == "sql" else "harbor_swe")
    if not harbor.exists():
        raise FileNotFoundError(
            f"HiL-Bench data not found at {harbor}. Run scripts/clone_third_party first."
        )

    tasks: list[Task] = []
    for task_dir in sorted(p for p in harbor.iterdir() if p.is_dir()):
        reg = _find_registry(task_dir)
        if reg is None:
            continue
        blockers = _read_blocker_registry(reg)
        statement = _read_statement(task_dir)
        instance_id = _read_instance_id(task_dir)
        tasks.append(
            Task(
                instance_id=instance_id,
                domain=domain,
                statement=statement,
                source="hil_bench",
                blockers=blockers,
                meta={"task_dir": str(task_dir), "registry": str(reg)},
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def _find_registry(task_dir: Path) -> Path | None:
    for cand in (
        task_dir / "blocker_registry.json",
        task_dir / "shared" / "ask-human-data" / "blocker_registry.json",
    ):
        if cand.exists():
            return cand
    globbed = list(task_dir.glob("**/*_registry.json"))
    return globbed[0] if globbed else None


def _read_instance_id(task_dir: Path) -> str:
    meta = task_dir / "metadata.json"
    if meta.exists():
        try:
            iid = json.loads(meta.read_text()).get("instance_id")
            if isinstance(iid, str) and iid.strip():
                return iid.strip()
        except Exception:
            pass
    return task_dir.name


def _read_statement(task_dir: Path) -> str:
    for cand in ("problem_statement.md", "task.md", "prompt.txt", "problem.txt"):
        p = task_dir / cand
        if p.exists():
            return p.read_text()[:8000]
    return f"HiL-Bench task {task_dir.name}"


def load_tasks(cfg: dict) -> list[Task]:
    """Dispatch on the `tasks:` config block."""
    source = cfg.get("source", "synthetic")
    if source == "synthetic":
        return make_synthetic_tasks(n=cfg.get("limit", 12), seed=cfg.get("seed", 0))
    if source == "hil_bench":
        return load_hil_bench_tasks(domain=cfg.get("domain", "swe"), limit=cfg.get("limit"))
    raise ValueError(f"unknown task source: {source!r}")
