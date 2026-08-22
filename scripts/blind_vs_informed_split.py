"""Construct validity: trace-blind vs trace-informed registry tasks.

028 Amendment A item 8. The interpretation-class artifact was authored in
two passes with different exposure to the collected data:

  trace-informed (20): swe_0, swe_1, swe_2, swe_10..swe_26 — drafted
      "grounded in registry+traces", then anchor-leak repaired (77 uniquely
      predictive anchors dropped).
  trace-blind (40): swe_3..swe_9, swe_27..swe_59 — derived REGISTRY-ONLY on
      2026-07-18, before any traces for those tasks existed.

If the fork census and the separation AUROCs agree across the two groups,
neither the fork construct nor the negatives can be an artifact of registry
authors having seen traces. This partially substitutes for the never-run
collection-R3 (OOD + sealed) leg. Reported as-run either way.

Consumes the cached pair rows written by scripts/t1_auroc_ci.py
(results/t1_pair_rows_{tag}_{rep}.json), so no embedder pass is repeated.

    python scripts/blind_vs_informed_split.py --rep hashed
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

from offline_ask_headtohead import (forked_blockers, load_commitments,  # noqa: E402
                                    load_task_actions)
from t1_auroc_ci import auroc_from  # noqa: E402
from wta.labeling import load_class_artifact  # noqa: E402

INFORMED = {"swe_0", "swe_1", "swe_2"} | {f"swe_{i}" for i in range(10, 27)}
BOOT = 2000
SEED = 0


def group_of(task: str) -> str:
    return "trace_informed" if task in INFORMED else "trace_blind"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="hashed")
    ap.add_argument("--tag", default="32b")
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--out", default="results/blind_vs_informed_split.json")
    args = ap.parse_args()

    t0 = time.time()
    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels_debug))
    actions = load_task_actions(Path(args.a0), None)
    actions = {t: r for t, r in actions.items() if t in art}

    out = {"split": {"trace_informed": sorted(INFORMED),
                     "n_informed": len(INFORMED),
                     "n_blind": len(actions) - len(INFORMED)},
           "census": {}, "pair_pool": {}}

    # ---- census per group -------------------------------------------
    for grp in ("trace_informed", "trace_blind"):
        ts = [t for t in actions if group_of(t) == grp]
        forks = {t: forked_blockers(actions[t], committed, list(art[t]))
                 for t in ts}
        n_commit = sum(1 for (rid, b) in committed
                       if rid.rsplit("-s", 1)[0] in set(ts))
        n_forked_tasks = sum(1 for t in ts if forks[t])
        out["census"][grp] = {
            "n_tasks": len(ts),
            "n_forked_tasks": n_forked_tasks,
            "frac_tasks_forked": n_forked_tasks / len(ts) if ts else None,
            "n_forked_blockers": sum(len(f) for f in forks.values()),
            "n_blockers": sum(len(art[t]) for t in ts),
            "n_commitments": n_commit,
        }
        c = out["census"][grp]
        print(f"{grp:15s}: {c['n_forked_tasks']}/{c['n_tasks']} tasks forked "
              f"({c['frac_tasks_forked']:.3f}), {c['n_forked_blockers']} "
              f"forked of {c['n_blockers']} blockers, "
              f"{c['n_commitments']} commitments")

    # ---- pair-pool AUROC per group ------------------------------------
    cache = Path(f"results/t1_pair_rows_{args.tag}_{args.rep}.json")
    if cache.exists():
        rows = [tuple(r) for r in json.loads(cache.read_text(encoding="utf-8"))]
        tasks = np.array([r[0] for r in rows])
        labels = np.array([r[2] for r in rows], dtype=int)
        dists = np.array([r[3] for r in rows], dtype=float)
        rng = np.random.default_rng(SEED)
        print()
        for grp in ("trace_informed", "trace_blind"):
            m = np.array([group_of(t) == grp for t in tasks])
            lab, dst = labels[m], dists[m]
            a = auroc_from(lab, dst)
            uniq = sorted(set(tasks[m].tolist()))
            idx = {t: np.where(tasks[m] == t)[0] for t in uniq}
            boots = []
            for _ in range(BOOT):
                pick = rng.choice(len(uniq), size=len(uniq), replace=True)
                sel = np.concatenate([idx[uniq[i]] for i in pick])
                v = auroc_from(lab[sel], dst[sel])
                if not np.isnan(v):
                    boots.append(v)
            b = np.array(boots)
            out["pair_pool"][grp] = {
                "rep": args.rep, "auroc": a,
                "n_pairs": int(m.sum()), "n_diff": int((lab == 1).sum()),
                "n_same": int((lab == 0).sum()), "n_tasks": len(uniq),
                "ci95": [float(np.percentile(b, 2.5)),
                         float(np.percentile(b, 97.5))]}
            p = out["pair_pool"][grp]
            print(f"{grp:15s}: {args.rep} AUROC {a:.3f}  "
                  f"clustered 95% CI [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]"
                  f"  (n={p['n_same']} same / {p['n_diff']} diff, "
                  f"{p['n_tasks']} tasks)")
    else:
        print(f"\n(no cached pair rows at {cache}; run t1_auroc_ci.py --rep "
              f"{args.rep} first — census reported above)")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {p} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
