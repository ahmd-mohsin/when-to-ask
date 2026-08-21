"""F2 (decisions/028): the rollout-budget ablation — how many runs buy the signal.

Census vs N: for N in {2, 4, 8, 12, 16, 24} seeds (20 resamples each, RNG seed
0), recompute (a) the fraction of tasks with >=1 interpretation fork, (b) the
forked-decision count, and (c) the per-decision run-level TESTABILITY floor
distribution (026's exact-test machinery: min_p = 1 / #distinct run-class
assignments; a decision is testable iff min_p < 0.05).

TWO UNIVERSES, fixed before the first run (pre-run adversarial review finding):

- (a) and (b) are CENSUS quantities: a decision is forked when >=2 committed
  classes appear in the labels trail among the sampled runs (the 40/60 and
  63-fork census facts live here).
- (c) is pre-registered as "026's exact-test machinery", which operates on the
  GATE5 UNIVERSE: per decision, only runs carrying >=1 class-labeled READ
  (labeling.py: a read gets cls>=0 iff anchor-labeled AND at/after the run's
  commit char, i.e. debug-trail phase==1), with gate5 eligibility (>=4
  class-labeled reads total AND >=2 distinct classes among them). On the full
  fixed-label data this universe has 37 eligible decisions at median 6
  runs/decision (results/gate5_permutation_test_fixed.json) vs 63/10 in the
  commitment trail — the floors differ systematically, so (c) is computed in
  the gate5 universe. Commitment-universe floors are ALSO emitted, clearly
  labeled supplementary, because the gap between the two curves is itself
  design guidance (forks exist that the gate machinery cannot even test).

Ground truth: the FIXED labeler's debug trail (models/v3_32b_fixed_debug);
run universe: the R2 collection (data/a0_v3_32b, 1415 runs). Pure CPU on
existing data; deterministic; no model calls.

Sampling rule (frozen): for each (N, resample, task) draw min(N, n_available)
run ids without replacement, uniformly, from one np.random.default_rng(0)
stream in the fixed loop order N asc -> resample asc -> task name asc. Tasks
with fewer than N available runs contribute all their runs and are counted in
`n_tasks_short`.

    python scripts/rollout_budget_ablation.py \
        --out results/rollout_budget_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gate5_permutation_test import n_distinct_assignments  # noqa: E402
from offline_ask_headtohead import (forked_blockers, load_commitments,  # noqa: E402
                                    load_task_actions)
from wta.labeling import load_class_artifact  # noqa: E402

N_GRID = (2, 4, 8, 12, 16, 24)
RESAMPLES = 20
SEED = 0
ALPHA = 0.05
MIN_READS = 4          # gate5 eligibility: >=4 class-labeled reads


def load_class_reads(debug_path: Path) -> dict:
    """{(task, blocker): {run_id: n_class_labeled_reads}} from the debug
    trail. A read is class-labeled iff phase == 1 (labeling.py:469-475:
    anchor-labeled AND at/after the run's commit char), in which case its
    class is the run's committed class on that blocker."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    with debug_path.open(encoding="utf-8") as fh:
        for line in fh:
            if '"phase": 1' not in line and '"phase":1' not in line:
                continue
            row = json.loads(line)
            if row.get("kind") != "read" or row.get("phase") != 1:
                continue
            run = row["run"]
            task = run.rsplit("-s", 1)[0]
            d = out.setdefault((task, row["decision"]), {})
            d[run] = d.get(run, 0) + 1
    return out


def floor_rows(sampled: dict[str, list[str]], class_reads, committed, art,
               universe: str) -> list[dict]:
    """Per-decision testability floors on one subsample.

    universe='gate5': runs = those with >=1 class-labeled read on the
    decision; eligibility >=MIN_READS reads total and >=2 classes.
    universe='commitment': runs = those with a labeled commitment (no read
    or count eligibility beyond >=2 classes) — supplementary only.
    """
    rows = []
    if universe == "gate5":
        sampled_sets = {t: set(r) for t, r in sampled.items()}
        for (task, blocker), by_run in sorted(class_reads.items()):
            if task not in sampled_sets:
                continue
            runs = {r: c for r, c in by_run.items() if r in sampled_sets[task]}
            if not runs:
                continue
            by_cls: dict[str, int] = {}
            n_reads = 0
            for r, c in runs.items():
                cls = committed.get((r, blocker))
                if cls is None:      # cannot happen (phase 1 implies commit)
                    continue
                by_cls[cls] = by_cls.get(cls, 0) + 1
                n_reads += c
            if n_reads < MIN_READS or len(by_cls) < 2:
                continue
            counts = sorted(by_cls.values(), reverse=True)
            n_assign = n_distinct_assignments(counts)
            rows.append({"task": task, "blocker": blocker,
                         "class_counts": counts,
                         "n_runs": int(sum(counts)),
                         "n_class_reads": int(n_reads),
                         "n_assignments": n_assign,
                         "min_p": 1.0 / n_assign,
                         "testable": bool(1.0 / n_assign < ALPHA)})
    else:
        for task in sorted(sampled):
            forks = forked_blockers(sampled[task], committed, list(art[task]))
            for blocker, by_cls in forks.items():
                counts = sorted((len(v) for v in by_cls.values()),
                                reverse=True)
                n_assign = n_distinct_assignments(counts)
                rows.append({"task": task, "blocker": blocker,
                             "class_counts": counts,
                             "n_runs": int(sum(counts)),
                             "n_assignments": n_assign,
                             "min_p": 1.0 / n_assign,
                             "testable": bool(1.0 / n_assign < ALPHA)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--out", default="results/rollout_budget_ablation.json")
    args = ap.parse_args()

    t0 = time.time()
    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels_debug))
    class_reads = load_class_reads(Path(args.labels_debug))
    actions = load_task_actions(Path(args.a0), None)
    actions = {t: r for t, r in actions.items() if t in art}
    run_ids = {t: sorted(runs) for t, runs in actions.items()}
    tasks = sorted(run_ids)
    n_tasks = len(tasks)
    print(f"tasks: {n_tasks}; runs: {sum(len(r) for r in run_ids.values())}; "
          f"commitments: {len(committed)}; decisions with class reads: "
          f"{len(class_reads)} ({time.time() - t0:.0f}s)")

    # consistency anchor: full-universe gate5 floors vs the recorded 026 run
    full = floor_rows(run_ids, class_reads, committed, art, "gate5")
    anchor = {"n_eligible_full": len(full),
              "runs_per_decision_median_full":
                  float(np.median([r["n_runs"] for r in full])) if full else None,
              "n_testable_full": int(sum(r["testable"] for r in full)),
              "recorded_026": {"n_eligible": 37, "runs_median": 6,
                               "n_testable": 13,
                               "source": "results/gate5_permutation_test_fixed.json"}}
    print(f"gate5-universe consistency: eligible {anchor['n_eligible_full']} "
          f"(recorded 37), runs/dec median "
          f"{anchor['runs_per_decision_median_full']} (recorded 6), "
          f"testable {anchor['n_testable_full']} (recorded 13)")

    def agg(vals):
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                "per_resample": vals}

    def floor_summary(rows):
        floors = np.array([r["min_p"] for r in rows])
        rpd = np.array([r["n_runs"] for r in rows])
        return {"n": len(rows),
                "frac_testable": float(np.mean(floors < ALPHA)) if len(rows) else None,
                "min_p_median": float(np.median(floors)) if len(rows) else None,
                "min_p_q25": float(np.quantile(floors, .25)) if len(rows) else None,
                "min_p_q75": float(np.quantile(floors, .75)) if len(rows) else None,
                "runs_per_decision_median": float(np.median(rpd)) if len(rows) else None}

    rng = np.random.default_rng(SEED)
    per_n = {}
    for n in N_GRID:
        frac_forked, dec_counts = [], []
        g5_eligible, g5_testable = [], []
        pooled_g5, pooled_commit = [], []
        n_tasks_short = 0
        for resample in range(RESAMPLES):
            sample = {}
            for task in tasks:
                avail = run_ids[task]
                k = min(n, len(avail))
                if resample == 0 and len(avail) < n:
                    n_tasks_short += 1
                idx = rng.choice(len(avail), size=k, replace=False)
                sample[task] = [avail[i] for i in sorted(idx)]

            # (a)+(b): census universe
            n_forked = 0
            n_dec = 0
            commit_rows = floor_rows(sample, class_reads, committed, art,
                                     "commitment")
            forked_tasks = {r["task"] for r in commit_rows}
            n_forked = len(forked_tasks)
            n_dec = len(commit_rows)
            frac_forked.append(n_forked / n_tasks)
            dec_counts.append(n_dec)

            # (c): gate5 universe floors
            g5_rows = floor_rows(sample, class_reads, committed, art, "gate5")
            g5_eligible.append(len(g5_rows))
            g5_testable.append(int(sum(r["testable"] for r in g5_rows)))
            for r in g5_rows:
                r["resample"] = resample
            for r in commit_rows:
                r["resample"] = resample
            pooled_g5.extend(g5_rows)
            pooled_commit.extend(commit_rows)

        per_n[str(n)] = {
            "n_tasks_short": n_tasks_short,
            "frac_tasks_forked": agg(frac_forked),
            "forked_decisions": agg(dec_counts),
            "gate5_eligible_decisions": agg(g5_eligible),
            "gate5_testable_decisions": agg(g5_testable),
            "floors_gate5_universe": floor_summary(pooled_g5),
            "floors_commitment_universe_supplementary":
                floor_summary(pooled_commit),
            "decisions_gate5": pooled_g5,
            "decisions_commitment_supplementary": pooled_commit,
        }
        fs = per_n[str(n)]["floors_gate5_universe"]
        print(f"N={n:>2}: frac forked {np.mean(frac_forked):.3f}"
              f"+-{np.std(frac_forked):.3f}  forked dec "
              f"{np.mean(dec_counts):.1f}+-{np.std(dec_counts):.1f}  "
              f"g5 eligible {np.mean(g5_eligible):.1f}"
              f"+-{np.std(g5_eligible):.1f}  g5 testable "
              f"{np.mean(g5_testable):.1f}+-{np.std(g5_testable):.1f}  "
              f"g5 frac_testable {fs['frac_testable']}"
              f"  ({time.time() - t0:.0f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"n_grid": N_GRID, "resamples": RESAMPLES, "seed": SEED,
                   "alpha": ALPHA, "min_reads": MIN_READS, "a0": args.a0,
                   "labels_debug": args.labels_debug,
                   "n_tasks": n_tasks,
                   "n_runs_total": sum(len(r) for r in run_ids.values())},
        "gate5_universe_consistency": anchor,
        "per_n": per_n}, indent=1), encoding="utf-8")
    print(f"wrote {out} ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
