"""Stage-1 offline detection head-to-head (decisions/027 §5, spec B2).

Replays the R2 collection task-by-task through the registry-free behavioral
divergence detector (wta.divergence + the frozen wta.online CUSUM trigger)
and scores task-level fork detection against FIXED-labeler ground truth,
with budget-matched offline baselines and gate7-style lead time.

The DETECTOR path never touches the class artifact; the artifact and the
labels debug trail are read only to build ground truth and calibration
targets (the offline-teacher discipline of A3, per 027 §3).

    python scripts/offline_ask_headtohead.py \
        --a0 data/a0_v3_32b --classes data/interpretation_classes.json \
        --labels-debug models/v3_32b_fixed_debug/labels_debug.jsonl \
        --out results/offline_headtohead_stage1.json

Deterministic; CPU-only; no model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.a4_gates import kfold_group_indices  # noqa: E402
from wta.divergence import (TriggerConfig, behavior_features,  # noqa: E402
                            replay_task)
from wta.labeling import _is_mutating, _norm, load_class_artifact  # noqa: E402
from wta.logging_schema import ActionEvent  # noqa: E402

THETA_GRID = (0.3, 0.5, 0.7)
H_GRID = (3.0, 6.0, 12.0)
REF_QUANTILE = 0.9
RANDOM_SEEDS = 20


def load_task_actions(a0: Path, n_runs: int | None):
    """{task: {run_id: [ActionEvent, ...] ordered}} — reads run JSONs only."""
    out = {}
    for task_dir in sorted(p for p in a0.iterdir() if p.is_dir()):
        runs = {}
        for jf in sorted(task_dir.glob("*.json")):
            rid = jf.stem
            if rid.endswith(".segments") or rid.startswith("collection_manifest"):
                continue
            if not (task_dir / f"{rid}.npz").exists():
                continue
            meta = json.loads(jf.read_text(encoding="utf-8"))
            runs[rid] = sorted((ActionEvent(**a) for a in meta["actions"]),
                               key=lambda a: a.segment_idx)
        if not runs:
            continue
        if n_runs is not None:
            keep = sorted(runs, key=lambda r: int(r.rsplit("-s", 1)[1]))[:n_runs]
            runs = {r: runs[r] for r in keep}
        out[task_dir.name] = runs
    return out


def load_commitments(debug_path: Path):
    """{(run_id, blocker): chosen_class_name} from the FIXED debug trail."""
    out = {}
    with debug_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("kind") == "commitment" and row.get("chosen"):
                out[(row["run"], row["blocker"])] = row["chosen"]
    return out


def commit_rounds(actions_by_run, committed, art, task):
    """{(run_id, blocker): turn index of the first mutating action carrying a
    signature of the committed class} — real_action_reads logic at turn
    granularity. None when the commitment was trace-sourced (no such action)."""
    sig_cache = {}
    out = {}
    for rid, acts in actions_by_run.items():
        for blocker, spec in art[task].items():
            chosen = committed.get((rid, blocker))
            if chosen is None:
                continue
            key = (blocker, chosen)
            if key not in sig_cache:
                cls = next(c for c in spec["classes"] if c["name"] == chosen)
                sig_cache[key] = [_norm(s) for s in cls["signatures"]]
            rnd = None
            for k, a in enumerate(acts):
                if _is_mutating(a.action_text or "") and any(
                        s in _norm(a.action_text or "") for s in sig_cache[key]):
                    rnd = k
                    break
            out[(rid, blocker)] = rnd
    return out


def forked_blockers(task_runs, committed, task_blockers):
    """{blocker: {class: [run_ids]}} restricted to blockers with >=2 classes."""
    forks = {}
    for blocker in task_blockers:
        by_cls = {}
        for rid in task_runs:
            c = committed.get((rid, blocker))
            if c is not None:
                by_cls.setdefault(c, []).append(rid)
        if len(by_cls) >= 2:
            forks[blocker] = by_cls
    return forks


def action_divergence_round(forks, rounds):
    """gate7 formula at turn granularity: per fork, min over differing-class
    run pairs of max(commit_round_i, commit_round_j); task-level = min over
    forks. None if no pair has both rounds defined."""
    best = None
    for blocker, by_cls in forks.items():
        classes = sorted(by_cls)
        for i, ca in enumerate(classes):
            for cb in classes[i + 1:]:
                for ra in by_cls[ca]:
                    for rb in by_cls[cb]:
                        ka, kb = rounds.get((ra, blocker)), rounds.get((rb, blocker))
                        if ka is not None and kb is not None:
                            v = max(ka, kb)
                            best = v if best is None else min(best, v)
    return best


def benign_reference(tasks, feats, committed, rounds, art):
    """0.9 quantile of ||r_i - r_j|| at commitment over SAME-class run pairs
    (the registry-labeled analog of a3 benign_spread_reference)."""
    dists = []
    for task in tasks:
        for blocker in art[task]:
            by_cls = {}
            for rid in feats[task]:
                c = committed.get((rid, blocker))
                k = rounds[task].get((rid, blocker))
                if c is not None and k is not None and k < len(feats[task][rid]):
                    by_cls.setdefault(c, []).append(feats[task][rid][k].r_vec)
            for vecs in by_cls.values():
                for i in range(len(vecs)):
                    for j in range(i + 1, len(vecs)):
                        dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))
    return float(np.quantile(dists, REF_QUANTILE)) if dists else 0.2


def prf(fired: set, truth: set, universe: list) -> dict:
    tp = len(fired & truth)
    p = tp / len(fired) if fired else 0.0
    r = tp / len(truth) if truth else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "n_fired": len(fired),
            "n_true": len(truth), "n_tasks": len(universe)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-runs", type=int, default=None,
                    help="subsample to first N seeds per task (live-eval sim)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/offline_headtohead_stage1.json")
    args = ap.parse_args()

    t0 = time.time()
    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels_debug))
    print(f"labeled commitments: {len(committed)}")
    actions = load_task_actions(Path(args.a0), args.n_runs)
    actions = {t: r for t, r in actions.items() if t in art}
    print(f"tasks: {len(actions)}; runs: {sum(len(r) for r in actions.values())} "
          f"({time.time() - t0:.0f}s)")

    feats = {t: {rid: behavior_features(acts) for rid, acts in runs.items()}
             for t, runs in actions.items()}
    rounds = {t: commit_rounds(actions[t], committed, art, t) for t in actions}
    forks = {t: forked_blockers(actions[t], committed, list(art[t])) for t in actions}
    truth_all = {t for t in actions if forks[t]}
    print(f"needs-ask tasks (ground truth): {len(truth_all)}/{len(actions)} "
          f"({time.time() - t0:.0f}s)")

    tasks = sorted(actions)
    gid = np.array(tasks)
    per_task_result = {}   # task -> dict (filled on its eval fold)
    fold_info = []
    for fold_i, (tr, te) in enumerate(
            kfold_group_indices(gid, args.folds, args.seed)):
        train_tasks = [tasks[i] for i in tr]
        eval_tasks = [tasks[i] for i in te]
        ref = benign_reference(train_tasks, feats, committed, rounds, art)

        best, best_f1 = None, -1.0
        for theta in THETA_GRID:
            for h in H_GRID:
                cfg = TriggerConfig(theta=theta, reference=ref, h_threshold=h)
                fired = {t for t in train_tasks
                         if replay_task(feats[t], cfg)}
                f1 = prf(fired, {t for t in train_tasks if forks[t]},
                         train_tasks)["f1"]
                if f1 > best_f1:
                    best, best_f1 = (theta, h), f1
        theta, h = best
        cfg = TriggerConfig(theta=theta, reference=ref, h_threshold=h)
        fold_info.append({"fold": fold_i, "reference": ref, "theta": theta,
                          "h": h, "train_f1": best_f1})
        print(f"fold {fold_i}: ref={ref:.3f} theta={theta} h={h} "
              f"train_f1={best_f1:.3f} ({time.time() - t0:.0f}s)")

        for t in eval_tasks:
            fires = replay_task(feats[t], cfg)
            res = {"fired": bool(fires),
                   "first_fire_round": fires[0].round if fires else None,
                   "n_fires": len(fires), "needs_ask": bool(forks[t]),
                   "n_rounds": max((len(f) for f in feats[t].values()),
                                   default=0),
                   "has_mutating": any(f and f[-1].weight > 0
                                       for f in feats[t].values())}
            if fires and forks[t]:
                attributed = any(
                    len({committed.get((rid, b)) for rid in fe.run_ids
                         if committed.get((rid, b)) is not None}) >= 2
                    for fe in fires for b in forks[t])
                res["attributed"] = attributed
                adr = action_divergence_round(forks[t], rounds[t])
                res["action_div_round"] = adr
                res["lead_turns"] = (adr - res["first_fire_round"]
                                     if adr is not None else None)
            per_task_result[t] = res

    # pooled eval metrics (every task is in eval exactly once)
    fired = {t for t, r in per_task_result.items() if r["fired"]}
    metrics = prf(fired, truth_all, tasks)
    tp = [t for t in fired if t in truth_all]
    leads = [per_task_result[t]["lead_turns"] for t in tp
             if per_task_result[t].get("lead_turns") is not None]
    attr = [per_task_result[t].get("attributed", False) for t in tp]

    # budget-matched baselines on the same universe
    rng = np.random.default_rng(args.seed)
    rand_f1 = [prf(set(rng.choice(tasks, size=len(fired), replace=False)),
                   truth_all, tasks)["f1"] for _ in range(RANDOM_SEEDS)]
    med_round = float(np.median([per_task_result[t]["first_fire_round"]
                                 for t in fired])) if fired else 0.0
    fixed_fired = {t for t in tasks
                   if per_task_result[t]["n_rounds"] > med_round}
    first_mut = {t for t in tasks if per_task_result[t]["has_mutating"]}
    baselines = {
        "random_budget_matched": {"f1_mean": float(np.mean(rand_f1)),
                                  "f1_sd": float(np.std(rand_f1)),
                                  "seeds": RANDOM_SEEDS},
        "fixed_step_at_median_fire": prf(fixed_fired, truth_all, tasks),
        "ask_at_first_mutating": prf(first_mut, truth_all, tasks),
    }

    print("\n=== STAGE 1: eval-fold detection (pooled over folds) ===")
    print(f"  divergence detector : P {metrics['precision']:.3f} "
          f"R {metrics['recall']:.3f} F1 {metrics['f1']:.3f} "
          f"({metrics['n_fired']} fired / {metrics['n_true']} needs-ask "
          f"/ {metrics['n_tasks']} tasks)")
    print(f"  random (matched n)  : F1 {baselines['random_budget_matched']['f1_mean']:.3f}"
          f" +- {baselines['random_budget_matched']['f1_sd']:.3f}")
    print(f"  fixed-step@{med_round:.0f}       : "
          f"F1 {baselines['fixed_step_at_median_fire']['f1']:.3f}")
    print(f"  first-mutating      : "
          f"F1 {baselines['ask_at_first_mutating']['f1']:.3f}")
    if leads:
        print(f"  lead time (turns)   : median {np.median(leads):+.1f} "
              f"(n={len(leads)}; positive = fired before the fork completed)")
    if attr:
        print(f"  attribution         : {sum(attr)}/{len(attr)} true positives "
              f"fired a bucket containing the forked runs")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"folds": args.folds, "n_runs": args.n_runs,
                   "theta_grid": THETA_GRID, "h_grid": H_GRID,
                   "ref_quantile": REF_QUANTILE, "seed": args.seed},
        "folds": fold_info, "detector": metrics, "baselines": baselines,
        "lead_turns": leads, "attribution": {"n_tp": len(attr),
                                             "n_attributed": int(sum(attr))},
        "per_task": per_task_result}, indent=1, default=str),
        encoding="utf-8")
    print(f"\nwrote {out} ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
