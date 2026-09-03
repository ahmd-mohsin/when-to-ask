"""DIAGNOSTIC: does the fork census survive on runs that produced working code?

The census -- 40/60 tasks fork -- is the paper's strongest positive. It is
computed over lexicon commitments with no reference to whether the run's code
does anything. results/test_outcome_vector.json now supplies that reference:
per run, whether the repository changed and whether the result was importable
(pytest could collect it).

Question: recompute the fork census counting ONLY runs that both changed the
repository and produced importable code. If two runs "resolved a blocker
differently" but one of them left the tree non-importable, that is not two
readings of an ambiguous spec -- it is one reading and one broken edit.

SCOPE AND CAUTION. This runs on the 3 python tasks the test-outcome vector
covers, which is 11 blockers and 2 forks. It is far too small to restate the
60-task census and is NOT evidence that the census is wrong. It is a screen:
if the effect is large here it justifies extending the test-outcome vector to
more tasks (cheap, no GPU) before any causal claim rests on the census.

    python scripts/diagnose_forks_on_importable.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from offline_ask_headtohead import load_commitments  # noqa: E402


def census(by_blocker, importable_only: bool):
    forked, total = 0, 0
    ftasks, tasks = set(), set()
    detail = {}
    for (t, b), classes in by_blocker.items():
        kept = {c: [r for r, ok in v if ok or not importable_only]
                for c, v in classes.items()}
        kept = {c: v for c, v in kept.items() if v}
        if not sum(len(v) for v in kept.values()):
            continue
        total += 1
        tasks.add(t)
        if len(kept) >= 2:
            forked += 1
            ftasks.add(t)
        detail[f"{t}/{b}"] = {c: len(v) for c, v in kept.items()}
    return {"forked_blockers": forked, "blockers_with_commitments": total,
            "forked_tasks": len(ftasks), "tasks": len(tasks),
            "per_blocker_class_counts": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed/labels_debug.jsonl")
    ap.add_argument("--testvec-dir", default="/ssd3/wta-testvec")
    ap.add_argument("--out", default="results/diag_forks_on_importable.json")
    args = ap.parse_args()

    recs = {}
    for f in Path(args.testvec_dir).glob("*.json"):
        if "__" in f.name:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        recs[d["run"]] = d

    committed = load_commitments(Path(args.labels_debug))
    by_blocker = defaultdict(lambda: defaultdict(list))
    for (run, b), chosen in committed.items():
        if run not in recs:
            continue
        r = recs[run]
        ok = bool(r.get("repo_changed")) and r.get("status") != "collection_error"
        by_blocker[(r["task"], b)][chosen].append((run, ok))

    all_runs = census(by_blocker, False)
    importable = census(by_blocker, True)
    n_ok = sum(1 for r in recs.values()
               if r.get("repo_changed") and r.get("status") != "collection_error")

    res = {
        "note": "DIAGNOSTIC, not a pre-registered cell and NOT a restatement "
                "of the 60-task census. A screen on the 3 python tasks the "
                "test-outcome vector covers.",
        "run_filter": {
            "n_runs": len(recs), "n_runs_importable_and_changed": n_ok,
            "definition": "repo_changed AND pytest could collect (status != "
                          "collection_error)"},
        "census_all_runs": all_runs,
        "census_importable_only": importable,
        "reading": (
            "Forked blockers go {}/{} -> {}/{} and forked tasks {}/{} -> {}/{} "
            "when only runs that produced working code are counted; {} of {} "
            "runs survive the filter. On this sample every fork is carried by "
            "runs that broke the build or wrote nothing. With {} blockers and "
            "{} forks this CANNOT generalise to the 60-task census -- it is a "
            "reason to extend the test-outcome vector to more tasks (no GPU "
            "needed) before the census carries a causal claim."
        ).format(all_runs["forked_blockers"], all_runs["blockers_with_commitments"],
                 importable["forked_blockers"], importable["blockers_with_commitments"],
                 all_runs["forked_tasks"], all_runs["tasks"],
                 importable["forked_tasks"], importable["tasks"],
                 n_ok, len(recs),
                 all_runs["blockers_with_commitments"], all_runs["forked_blockers"]),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"all runs        : {all_runs['forked_blockers']}/"
          f"{all_runs['blockers_with_commitments']} blockers, "
          f"{all_runs['forked_tasks']}/{all_runs['tasks']} tasks")
    print(f"importable only : {importable['forked_blockers']}/"
          f"{importable['blockers_with_commitments']} blockers, "
          f"{importable['forked_tasks']}/{importable['tasks']} tasks")
    print(f"runs surviving  : {n_ok}/{len(recs)}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
