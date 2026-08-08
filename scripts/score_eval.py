"""Phase-4 scoring: assemble the headline table (spec eval, decisions/022).

    python scripts/score_eval.py --our data/eval \
        --flat-tasks data/hilbench_flat_sealed \
        --bridge results/bridge --resolve --out results/eval

CPU-safe without --resolve (uses --resolved-cache or marks passes
unresolved); --resolve invokes hil-bench's own evaluator (docker, GPU box).
Bridge inputs: every ask_human_logs.json found under --bridge (Ask-F1
recomputed with true registry counts, decisions/022 §3) and an optional
--bridge-rows JSON (a rows list assembled at GPU time from the harness's
pass-level outputs) for bridge pass@3.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.eval.scoring import (  # noqa: E402
    HilBenchResolver, collect_our_results, predictions_by_arm, score,
)
from xtid.harness.tasks import load_hil_bench_tasks  # noqa: E402


def load_fork_annotations(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"WARNING: fork annotations {p} not found -- fork slice "
              f"reported unsplit (decisions/022 §2g requires the frozen file "
              f"before unsealing)")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def gather_bridge_ask_logs(bridge_root: Path) -> dict:
    merged: dict = {}
    for f in sorted(bridge_root.rglob("ask_human_logs.json")):
        try:
            merged.update(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"WARNING: unreadable {f}: {e}")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our", required=True, help="run_eval.py output root")
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--flat-tasks", default=None,
                    help="materialized flat tasks dir (needed for --resolve)")
    ap.add_argument("--bridge", default=None,
                    help="hil-bench harness output root (bridge rows)")
    ap.add_argument("--bridge-rows", default=None,
                    help="JSON rows list for bridge pass@3")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--include-partial", action="store_true")
    ap.add_argument("--fork-annotations", default="data/fork_type_annotations.json")
    ap.add_argument("--resolve", action="store_true",
                    help="run hil-bench's evaluator (docker; GPU box)")
    ap.add_argument("--resolved-cache", default=None,
                    help="JSON {expanded_id: bool}; written after --resolve, "
                         "read when scoring on CPU")
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    records = collect_our_results(Path(args.our))
    if not records:
        raise SystemExit(f"no pass records under {args.our}")
    print(f"{len(records)} pass records "
          f"({len({r.arm for r in records})} arms, "
          f"{len({r.task for r in records})} tasks)")

    tasks = load_hil_bench_tasks(
        domain="swe", root=Path(args.tasks_dir).parent)
    blockers_by_task = {Path(t.meta["task_dir"]).name: t.blockers
                        for t in tasks
                        if Path(t.meta["task_dir"]).name in
                        {r.task for r in records}}

    resolved: dict[str, bool] = {}
    cache = Path(args.resolved_cache) if args.resolved_cache else None
    if cache and cache.exists():
        resolved = json.loads(cache.read_text(encoding="utf-8"))
    if args.resolve:
        if not args.flat_tasks:
            raise SystemExit("--resolve needs --flat-tasks")
        resolver = HilBenchResolver()
        for arm, preds in predictions_by_arm(records).items():
            todo = {k: v for k, v in preds.items() if k not in resolved}
            if not todo:
                continue
            print(f"resolving {len(todo)} predictions for arm {arm}...")
            resolved.update(resolver.resolve(todo, Path(args.flat_tasks)))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(resolved, indent=1), encoding="utf-8")
    elif not resolved:
        print("WARNING: no --resolve and no --resolved-cache: all passes "
              "scored unresolved (Ask-F1 and budgets still valid)")

    bridge_ask_logs = None
    if args.bridge:
        bridge_ask_logs = gather_bridge_ask_logs(Path(args.bridge))
        print(f"bridge: {len(bridge_ask_logs)} ask-log instances")
    bridge_rows = None
    if args.bridge_rows:
        bridge_rows = json.loads(Path(args.bridge_rows).read_text(encoding="utf-8"))

    result = score(records, resolved, blockers_by_task,
                   load_fork_annotations(args.fork_annotations),
                   expected_passes=args.passes,
                   include_partial=args.include_partial,
                   bridge_ask_logs=bridge_ask_logs, bridge_rows=bridge_rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "headline.md").write_text(result["markdown"], encoding="utf-8")
    with (out / "headline.csv").open("w", newline="", encoding="utf-8") as fh:
        if result["csv_rows"]:
            w = csv.DictWriter(fh, fieldnames=list(result["csv_rows"][0]))
            w.writeheader()
            w.writerows(result["csv_rows"])
    (out / "metrics.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "markdown"},
                   indent=1, default=str), encoding="utf-8")
    print(f"\n{result['markdown']}\n\nwritten to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
