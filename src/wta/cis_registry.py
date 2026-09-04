"""CIS ground-truth resolutions -- OURS (decisions/029).

The counterfactual context injects R_b, the benchmark's own resolution of
blocker b. Its source is the structured registry
``<task>/shared/ask-human-data/blocker_registry.json`` -- the SAME string the
benchmark's ask_human server returns to an agent that asks
(hil-bench ask-human/server.py:262-264) and the same string
``full_info/instruction.md`` renders (a byte-exact function of baseline +
registry, verified 60/60). Extraction is by blocker id; keyword or signature
matching is forbidden -- a rival class's signature appears inside the
canonical resolution in 116/214 cases because resolutions enumerate
counter-examples.

029 frozen policy: R_b is used VERBATIM (``.strip()`` only). No unescaping of
the literal ``\\n`` / ``\\"`` / ``\\\\`` sequences some resolutions carry, and
no repair of the ``.",`` artifact in swe_38 -- that is what the benchmark
delivers, and the H.1 causal effect was obtained with it. The defective ids
are pinned as a LIST in data/cis_registry_pins.json (counts vary with the
scan's definition; ids do not).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

REGISTRY_REL = Path("shared") / "ask-human-data" / "blocker_registry.json"
_BACKTICK = re.compile(r"`([^`]+)`")
_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:[.\-/][A-Za-z0-9_]+)+\b")


@dataclass(frozen=True)
class Resolution:
    task: str
    blocker_id: str
    type: str
    description: str
    resolution: str            # verbatim, stripped
    canonical_class: str
    n_classes: int
    idx: int                   # position in the registry (== class-artifact order)

    @property
    def key(self) -> tuple[str, str]:
        return self.task, self.blocker_id

    @property
    def has_code(self) -> bool:
        return "`" in self.resolution

    @property
    def n_words(self) -> int:
        return len(self.resolution.split())

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.resolution.encode("utf-8")).hexdigest()


def render_full_info(baseline_text: str, blockers: list[dict]) -> str:
    """The byte-exact template hil-bench used to produce full_info from
    baseline + registry (verified 60/60 tasks)."""
    out = "\n## BLOCKER DETAILS"
    for i, b in enumerate(blockers):
        out += "\n" if i == 0 else "\n\n"
        out += "### " + b["description"].lstrip("\n") + "\n" + b["resolution"].strip()
    return baseline_text + out + "\n"


def load_task_registry(task_dir: str | Path) -> list[dict]:
    p = Path(task_dir) / REGISTRY_REL
    return json.loads(p.read_text(encoding="utf-8"))["blockers"]


def load_resolutions(tasks_dir: str | Path, classes_path: str | Path,
                     task_ids: list[str] | None = None,
                     check_full_info: bool = True) -> dict[tuple[str, str], Resolution]:
    """All resolutions keyed by (task, blocker_id), with the four 029
    assertions: registry id order == class-artifact key order; class 0 is
    canonical for every blocker; every resolution non-empty; every stripped
    resolution appears verbatim in full_info/instruction.md (and full_info is
    exactly render_full_info(baseline, registry))."""
    tasks_dir = Path(tasks_dir)
    art = json.loads(Path(classes_path).read_text(encoding="utf-8"))
    tasks = task_ids or sorted(t for t in art if t.startswith("swe_"))
    out: dict[tuple[str, str], Resolution] = {}
    for task in tasks:
        tdir = tasks_dir / task
        reg = load_task_registry(tdir)
        ids = [b["id"] for b in reg]
        art_ids = list(art[task].keys())
        if ids != art_ids:
            raise AssertionError(f"{task}: registry id order {ids} != class "
                                 f"artifact order {art_ids}")
        if check_full_info:
            base = (tdir / "baseline" / "instruction.md").read_text(encoding="utf-8")
            full = (tdir / "full_info" / "instruction.md").read_text(encoding="utf-8")
            if render_full_info(base, reg) != full:
                raise AssertionError(f"{task}: full_info is not the registry render")
        for i, b in enumerate(reg):
            classes = art[task][b["id"]]["classes"]
            if not classes or not classes[0].get("canonical"):
                raise AssertionError(f"{task}/{b['id']}: class 0 not canonical")
            if any(c.get("canonical") for c in classes[1:]):
                raise AssertionError(f"{task}/{b['id']}: non-zero canonical class")
            res = (b.get("resolution") or "").strip()
            if not res:
                raise AssertionError(f"{task}/{b['id']}: empty resolution")
            if check_full_info and res not in full:
                raise AssertionError(f"{task}/{b['id']}: resolution not verbatim "
                                     f"in full_info")
            out[(task, b["id"])] = Resolution(
                task=task, blocker_id=b["id"], type=str(b.get("type") or ""),
                description=(b.get("description") or "").strip(), resolution=res,
                canonical_class=classes[0]["name"], n_classes=len(classes), idx=i)
    return out


def defective_ids(resolutions: dict[tuple[str, str], Resolution]) -> list[str]:
    """Blockers whose resolution carries a literal backslash escape or the
    leaked JSON terminator. Recorded, never repaired (029 policy)."""
    bad = []
    for r in resolutions.values():
        s = r.resolution
        if ("\\n" in s or '\\"' in s or "\\\\" in s or '.",' in s
                or '.",' in r.description):
            bad.append(f"{r.task}/{r.blocker_id}")
    return sorted(bad)


def identifiers(text: str) -> set[str]:
    """Backticked spans plus dotted / slashed / hyphenated identifiers -- the
    tokens by which an injected resolution could be 'genuinely relevant' to a
    task it was not written for."""
    ids = {m.group(1).strip() for m in _BACKTICK.finditer(text)}
    ids |= {m.group(0) for m in _IDENT.finditer(text)}
    return {i for i in ids if len(i) >= 3}


def foreign_controls(key: tuple[str, str],
                     pool: dict[tuple[str, str], Resolution],
                     instructions: dict[str, str], *, n: int = 2,
                     seed: int = 0, tol: float = 0.25,
                     max_shared_idents: int = 2) -> list[tuple[str, str]]:
    """029 relevance control: n resolutions from OTHER tasks, matched on
    blocker type, backtick presence and word length (+/- tol), whose
    identifiers do not appear in the target task's instruction, from tasks
    whose instruction shares at most `max_shared_idents` identifiers with the
    target's (same-repo exclusion; metadata.json carries no repo field).
    Deterministic under `seed`. Falls back to relaxing the type match, then
    the length window, before giving up -- and records which stage it used
    in the returned keys' order (callers log `len(result)`)."""
    import random
    tgt = pool[key]
    tgt_idents = identifiers(instructions[tgt.task])
    lo, hi = tgt.n_words * (1 - tol), tgt.n_words * (1 + tol)

    def ok(r: Resolution, strict_type: bool, strict_len: bool) -> bool:
        if r.task == tgt.task:
            return False
        if strict_type and r.type != tgt.type:
            return False
        if r.has_code != tgt.has_code:
            return False
        if strict_len and not (lo <= r.n_words <= hi):
            return False
        if identifiers(r.resolution) & tgt_idents:
            return False
        if len(identifiers(instructions[r.task]) & tgt_idents) > max_shared_idents:
            return False
        return True

    rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
    for strict_type, strict_len in ((True, True), (False, True), (True, False),
                                    (False, False)):
        cands = sorted(k for k, r in pool.items() if ok(r, strict_type, strict_len))
        if len(cands) >= n:
            rng.shuffle(cands)
            return cands[:n]
    return []


def load_rival_fixture(path: str | Path) -> dict[tuple[str, str, str], str]:
    """Pilot-only hand-authored rival resolutions (029 G0.11), keyed by
    (task, blocker_id, class_name). Entries must be owner-approved."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for e in d["entries"]:
        if e.get("status") != "approved":
            continue
        out[(e["task"], e["blocker_id"], e["class"])] = e["rival_resolution"].strip()
    return out


def pins(resolutions: dict[tuple[str, str], Resolution]) -> dict:
    """The content of data/cis_registry_pins.json: per-resolution sha256, id
    order per task, and the defective-id list."""
    by_task: dict[str, dict] = {}
    for r in resolutions.values():
        by_task.setdefault(r.task, {"id_order": [], "sha256": {}})
        by_task[r.task]["id_order"].append(r.blocker_id)
        by_task[r.task]["sha256"][r.blocker_id] = r.sha
    return {"n_tasks": len(by_task), "n_blockers": len(resolutions),
            "defective_ids": defective_ids(resolutions), "tasks": by_task}
