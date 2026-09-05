"""Build data/cis_lean_statements_pilot.json (decisions/029 Amendment 029.3).

Mechanical -- no new authoring. For each pilot blocker: one statement per
registry class, all with the frozen template STMT_PREFIX + text, where text is
the registry's canonical `resolution` (verbatim, stripped -- 029 par.3.3) for
class 0 and the independently reviewed rival resolution
(data/cis_rival_resolutions_pilot.json, status=approved) for every other
class. Statements therefore inherit the rival review's form-matching and
need no second review; the template is a constant.

Order inside each blocker follows interpretation_classes.json (class 0 first),
which the loader asserts equals the registry order. A blocker whose rival set
is incomplete (some class without an approved rival) is written with
`complete: false` and is EXCLUDED from p_k(c|b) -- a distribution over a
partial menu is not the model's lean.

    python scripts/cis_build_statements.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wta.cis_lean import NULL_USER, STMT_PREFIX, statement  # noqa: E402
from wta.cis_registry import load_resolutions, load_rival_fixture  # noqa: E402

PILOT_TASKS = ["swe_0", "swe_10", "swe_11"]


def build(tasks_dir: Path, classes_path: Path, rival_path: Path,
          tasks: list[str]) -> dict:
    art = json.loads(classes_path.read_text(encoding="utf-8"))
    res = load_resolutions(tasks_dir, classes_path, task_ids=tasks)
    rivals = load_rival_fixture(rival_path)
    blockers = []
    for (task, bid), r in sorted(res.items(), key=lambda kv: (kv[0][0], kv[1].idx)):
        classes = [c["name"] for c in art[task][bid]["classes"]]
        texts, complete = [], True
        for i, c in enumerate(classes):
            if i == 0:
                texts.append(r.resolution)
            else:
                t = rivals.get((task, bid, c))
                if t is None:
                    complete = False
                    texts.append(None)
                else:
                    texts.append(t)
        blockers.append({
            "task": task, "blocker_id": bid, "type": r.type, "classes": classes,
            "statements": [statement(t) if t is not None else None for t in texts],
            "complete": complete,
            "sha256": [hashlib.sha256(statement(t).encode("utf-8")).hexdigest()
                       if t is not None else None for t in texts],
        })
    return {"_provenance": {
                "purpose": "029.3 lean readout: one templated commitment statement per registry class",
                "template": {"STMT_PREFIX": STMT_PREFIX, "NULL_USER": NULL_USER},
                "sources": "class 0 = registry resolution verbatim (029 par.3.3); other classes = "
                           "data/cis_rival_resolutions_pilot.json entries with status=approved "
                           "(independently reviewed 2026-09-04)",
                "rule": "no new authoring; regenerate with scripts/cis_build_statements.py; the "
                        "contract test asserts idempotence"},
            "tasks": tasks,
            "n_blockers": len(blockers),
            "n_complete": sum(b["complete"] for b in blockers),
            "n_statements": sum(sum(s is not None for s in b["statements"]) for b in blockers),
            "blockers": blockers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--rivals", default="data/cis_rival_resolutions_pilot.json")
    ap.add_argument("--tasks", default=",".join(PILOT_TASKS))
    ap.add_argument("--out", default="data/cis_lean_statements_pilot.json")
    args = ap.parse_args()
    d = build(Path(args.tasks_dir), Path(args.classes), Path(args.rivals),
              [t for t in args.tasks.split(",") if t])
    Path(args.out).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"{d['n_blockers']} blockers ({d['n_complete']} complete), "
          f"{d['n_statements']} statements -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
