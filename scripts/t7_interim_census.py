"""T7 INTERIM look — fork census + R3/R4 pair-pool AUROC on the PARTIAL
cross-family collection.

*** THIS IS NOT THE PRE-REGISTERED T7 RESULT. ***

The T7 collection is still running. The tasks finished so far are the first
ones in each shard's lexicographic slice, NOT a random subset of the 60, so
every number here is subject to task-selection bias and will move. It exists
to answer "which way is it pointing" mid-flight, not to be reported.

Construction is identical to scripts/t5_cross_scale.py (the cross-SCALE
sibling) so the interim numbers are read on the same footing as T5 and the
frozen 32B reference: pair pool from feature_signal_gate.pair_distances,
commitment-anchored at the first mutating action carrying the committed class
signature.

    python scripts/t7_interim_census.py \
        --a0 data/xfam_mistralai-mistral-small-24b-instruct-2501 \
        --labels models/xfam_2501_interim/labels_debug.jsonl \
        --out results/t7_interim_partial.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from feature_signal_gate import auroc, pair_distances  # noqa: E402
from offline_ask_headtohead import (forked_blockers, load_commitments,  # noqa: E402
                                    load_task_actions)
from wta.labeling import load_class_artifact  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="results/t7_interim_partial.json")
    ap.add_argument("--complete-only", action="store_true",
                    help="restrict to tasks with all 24 seeds present -- the "
                         "only subset where the census is not truncated")
    args = ap.parse_args()

    t0 = time.time()
    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels))
    actions = load_task_actions(Path(args.a0), None)
    actions = {t: r for t, r in actions.items() if t in art}

    depth = {t: len(r) for t, r in actions.items()}
    complete = sorted(t for t, n in depth.items() if n >= 24)
    if args.complete_only:
        actions = {t: r for t, r in actions.items() if t in set(complete)}

    n_runs = sum(len(r) for r in actions.values())
    forks = {t: forked_blockers(actions[t], committed, list(art[t]))
             for t in actions}
    n_actions = sum(len(a) for runs in actions.values() for a in runs.values())
    census = {
        "n_tasks": len(actions),
        "n_tasks_at_full_24_seeds": len(complete),
        "n_runs": n_runs,
        "n_action_events": n_actions,
        "n_commitments": len(committed),
        "n_forked_blockers": sum(len(f) for f in forks.values()),
        "n_forked_tasks": sum(1 for t in forks if forks[t]),
        "frac_tasks_forked": (sum(1 for t in forks if forks[t]) / len(actions)
                              if actions else None),
        "seed_depth_min": min(depth.values()) if depth else None,
        "seed_depth_max": max(depth.values()) if depth else None,
    }

    rows = {}
    minilm = None
    for row, emb_name in (("r3_hashed", None), ("r4_minilm", "minilm")):
        emb = None
        if emb_name == "minilm":
            from wta.embed import MiniLMEmbedder
            minilm = minilm or MiniLMEmbedder()
            emb = minilm
        same, diff = pair_distances(actions, art, committed, emb)
        if len(same) == 0 or len(diff) == 0:
            rows[row] = {"auroc": None, "n_same": int(len(same)),
                         "n_diff": int(len(diff)), "note": "empty pair pool"}
            continue
        rows[row] = {"auroc": auroc(same, diff), "n_same": int(len(same)),
                     "n_diff": int(len(diff)),
                     "same_mean": float(same.mean()),
                     "diff_mean": float(diff.mean())}

    out = {
        "INTERIM": True,
        "warning": ("PARTIAL collection, non-random task subset. Not the "
                    "pre-registered T7 result. Numbers will move."),
        "a0": args.a0, "labels": args.labels,
        "complete_only": bool(args.complete_only),
        "model": "mistralai/Mistral-Small-24B-Instruct-2501",
        "census": census, "pair_pool": rows,
        "reference_32b_R2": {"r3_hashed_auroc": 0.555, "r4_minilm_auroc": 0.580,
                             "frac_tasks_forked": 0.667,
                             "source": "results/feature_signal_gate.json"},
        "reference_r6b_llm": {"auroc": 0.5786, "ci": [0.481, 0.670]},
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"tasks {census['n_tasks']} ({census['n_tasks_at_full_24_seeds']} at "
          f"full 24 seeds), runs {n_runs}, actions {n_actions}")
    print(f"forked {census['n_forked_tasks']}/{census['n_tasks']} tasks "
          f"= {census['frac_tasks_forked']:.3f}   (R2 32B: 0.667)")
    for r, v in rows.items():
        print(f"  {r:10s} AUROC " + (f"{v['auroc']:.4f} "
              f"(n_same={v['n_same']}, n_diff={v['n_diff']})"
              if v["auroc"] is not None else f"N/A ({v.get('note')})"))
    print(f"wrote {p} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
