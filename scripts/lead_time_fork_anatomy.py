"""F4 (decisions/028): lead-time and fork anatomy — descriptive, all data exists.

Commitment-dating distributions; fork onset vs completion (gate7 formula at
turn granularity, per blocker); splits by fork type (structural/value,
data/fork_type_annotations.json) and by temperature arm.

Definitions (frozen here, all reusing the stage-1 loaders' semantics):
- commitment round: turn index of the first mutating action carrying a
  signature of the committed class (offline_ask_headtohead.commit_rounds);
  None when the commitment was trace-sourced -> "not datable".
- fork onset: min datable commitment round across the fork's committed runs
  (the moment the first run silently commits).
- fork completion: min over differing-class run pairs of
  max(round_i, round_j) — the earliest turn at which two runs have
  observably resolved the blocker differently (gate7's formula, per blocker).
- ask window: completion - onset (turns in which a detector could still see
  the fork forming before it is behaviorally complete).
- modal class: the fork's largest committed class (ties broken by class-name
  sort order); minority runs = committed runs outside the modal class.

Pure CPU on existing data; deterministic; no model calls.

    python scripts/lead_time_fork_anatomy.py \
        --out results/lead_time_fork_anatomy.json
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

from offline_ask_headtohead import (commit_rounds, forked_blockers,  # noqa: E402
                                    load_commitments, load_task_actions)
from wta.labeling import load_class_artifact  # noqa: E402


def load_temperatures(a0: Path) -> dict[str, float]:
    """{run_id: temperature} — light second pass over the run JSONs."""
    out = {}
    for task_dir in sorted(p for p in a0.iterdir() if p.is_dir()):
        for jf in sorted(task_dir.glob("*.json")):
            rid = jf.stem
            if rid.endswith(".segments") or rid.startswith("collection_manifest"):
                continue
            if not (task_dir / f"{rid}.npz").exists():
                continue
            out[rid] = json.loads(jf.read_text(encoding="utf-8"))["temperature"]
    return out


def fork_completion_round(by_cls: dict, rounds, blocker: str):
    """gate7 formula for ONE blocker: min over differing-class run pairs of
    max(commit_round_i, commit_round_j); None if no pair is fully datable."""
    best = None
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


def dist(vals: list) -> dict:
    v = np.array([x for x in vals if x is not None], dtype=float)
    return {"n": int(len(v)),
            "median": float(np.median(v)) if len(v) else None,
            "q25": float(np.quantile(v, .25)) if len(v) else None,
            "q75": float(np.quantile(v, .75)) if len(v) else None,
            "mean": float(v.mean()) if len(v) else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--fork-types", default="data/fork_type_annotations.json")
    ap.add_argument("--out", default="results/lead_time_fork_anatomy.json")
    args = ap.parse_args()

    t0 = time.time()
    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels_debug))
    fork_types = json.loads(Path(args.fork_types).read_text(encoding="utf-8"))
    actions = load_task_actions(Path(args.a0), None)
    actions = {t: r for t, r in actions.items() if t in art}
    temps = load_temperatures(Path(args.a0))
    print(f"tasks: {len(actions)}; runs: {sum(len(r) for r in actions.values())}; "
          f"commitments: {len(committed)} ({time.time() - t0:.0f}s)")

    rounds = {t: commit_rounds(actions[t], committed, art, t) for t in actions}
    forks = {t: forked_blockers(actions[t], committed, list(art[t]))
             for t in actions}
    print(f"rounds computed ({time.time() - t0:.0f}s)")

    # --- per-commitment dating rows -------------------------------------
    forked_keys = {(t, b) for t in forks for b in forks[t]}
    commit_rows = []
    for t in sorted(actions):
        for b in art[t]:
            for rid in actions[t]:
                chosen = committed.get((rid, b))
                if chosen is None:
                    continue
                rnd = rounds[t].get((rid, b))
                n_turns = len(actions[t][rid])
                commit_rows.append({
                    "task": t, "blocker": b, "run": rid, "class": chosen,
                    "round": rnd, "datable": rnd is not None,
                    "n_turns": n_turns,
                    "rel_pos": (rnd / max(n_turns - 1, 1)
                                if rnd is not None else None),
                    "temperature": temps.get(rid),
                    "fork_type": fork_types.get(b),
                    "on_forked_blocker": (t, b) in forked_keys,
                })

    # --- per-fork anatomy rows -------------------------------------------
    fork_rows = []
    for t in sorted(forks):
        for b, by_cls in forks[t].items():
            datable = [rounds[t].get((rid, b))
                       for runs in by_cls.values() for rid in runs]
            datable = [r for r in datable if r is not None]
            onset = min(datable) if datable else None
            completion = fork_completion_round(by_cls, rounds[t], b)
            counts = {c: len(v) for c, v in by_cls.items()}
            modal = sorted(counts, key=lambda c: (-counts[c], c))[0]
            minority_runs = [rid for c, v in by_cls.items() if c != modal
                             for rid in v]
            fork_rows.append({
                "task": t, "blocker": b, "fork_type": fork_types.get(b),
                "class_counts": dict(sorted(counts.items())),
                "n_runs_committed": sum(counts.values()),
                "n_classes": len(counts),
                "onset_round": onset,
                "completion_round": completion,
                "ask_window_turns": (completion - onset
                                     if completion is not None
                                     and onset is not None else None),
                "modal_class": modal,
                "minority_temps": sorted(temps[r] for r in minority_runs
                                         if r in temps),
                "modal_temps": sorted(temps[r] for r in by_cls[modal]
                                      if r in temps),
            })

    # --- aggregates --------------------------------------------------------
    def commit_stats(rows):
        return {"n": len(rows),
                "n_datable": sum(r["datable"] for r in rows),
                "frac_datable": (sum(r["datable"] for r in rows) / len(rows)
                                 if rows else None),
                "round": dist([r["round"] for r in rows]),
                "rel_pos": dist([r["rel_pos"] for r in rows])}

    temp_arms = sorted({r["temperature"] for r in commit_rows
                        if r["temperature"] is not None})
    by_temp = {}
    for arm in temp_arms:
        rows = [r for r in commit_rows if r["temperature"] == arm]
        on_fork = [r for r in rows if r["on_forked_blocker"]]
        minority = 0
        for fr in fork_rows:
            minority += sum(1 for x in fr["minority_temps"] if x == arm)
        n_arm_fork_commits = sum(
            len([x for x in fr["minority_temps"] + fr["modal_temps"]
                 if x == arm]) for fr in fork_rows)
        by_temp[str(arm)] = {
            "commitments": commit_stats(rows),
            "on_forked_blocker": len(on_fork),
            "minority_share_on_forks": (minority / n_arm_fork_commits
                                        if n_arm_fork_commits else None),
        }

    def fork_stats(rows):
        return {"n_forks": len(rows),
                "onset_round": dist([r["onset_round"] for r in rows]),
                "completion_round": dist([r["completion_round"] for r in rows]),
                "ask_window_turns": dist([r["ask_window_turns"] for r in rows]),
                "n_fully_datable": sum(r["completion_round"] is not None
                                       for r in rows)}

    summary = {
        "commitments": {
            "all": commit_stats(commit_rows),
            "on_forked_blockers": commit_stats(
                [r for r in commit_rows if r["on_forked_blocker"]]),
            "by_fork_type": {ft: commit_stats(
                [r for r in commit_rows if r["fork_type"] == ft])
                for ft in ("structural", "value")},
            "by_temperature": by_temp,
        },
        "forks": {
            "all": fork_stats(fork_rows),
            "by_fork_type": {ft: fork_stats(
                [r for r in fork_rows if r["fork_type"] == ft])
                for ft in ("structural", "value")},
            "n_forked_tasks": sum(1 for t in forks if forks[t]),
            "n_tasks": len(actions),
        },
    }

    s = summary
    print(f"\n=== F4: fork anatomy ({s['forks']['n_forked_tasks']}"
          f"/{s['forks']['n_tasks']} tasks forked, "
          f"{s['forks']['all']['n_forks']} forks) ===")
    print(f"  commitments: {s['commitments']['all']['n']} "
          f"({s['commitments']['all']['frac_datable']:.2f} datable); "
          f"round median {s['commitments']['all']['round']['median']}, "
          f"rel pos median {s['commitments']['all']['rel_pos']['median']:.2f}")
    print(f"  fork onset median {s['forks']['all']['onset_round']['median']}, "
          f"completion median {s['forks']['all']['completion_round']['median']}, "
          f"ask window median {s['forks']['all']['ask_window_turns']['median']} "
          f"turns (n={s['forks']['all']['ask_window_turns']['n']})")
    for ft in ("structural", "value"):
        f = s["forks"]["by_fork_type"][ft]
        print(f"  {ft:10s}: {f['n_forks']} forks, window median "
              f"{f['ask_window_turns']['median']} "
              f"(n={f['ask_window_turns']['n']})")
    for arm in temp_arms:
        bt = by_temp[str(arm)]
        ms = bt["minority_share_on_forks"]
        print(f"  temp {arm}: {bt['commitments']['n']} commitments, "
              f"minority share on forks "
              f"{ms:.3f}" if ms is not None else
              f"  temp {arm}: {bt['commitments']['n']} commitments")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"a0": args.a0, "labels_debug": args.labels_debug,
                   "fork_types": args.fork_types},
        "summary": summary,
        "fork_rows": fork_rows,
        "commit_rows": commit_rows}, indent=1), encoding="utf-8")
    print(f"\nwrote {out} ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
