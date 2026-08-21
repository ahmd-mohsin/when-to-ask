"""T5 (decisions/028): cross-scale replication — R3/R4 AUROC + fork census
at 14B (data/a0_v2) and 7B (data/a0).

Labels come from the FIXED labeler (026 pipeline, current code) regenerated
per collection with tokenizer='auto' (manifest model_id):
  models/t5_v2_14b_fixed  (Qwen2.5-Coder-14B-Instruct, 160 runs)
  models/t5_a0_7b_fixed   (Qwen2.5-Coder-7B-Instruct, 160 runs)

The pair pool is the SAME construction as the frozen 32B pool
(feature_signal_gate.pair_distances): commitment-pair distances anchored at
the first mutating action carrying the committed class signature. The v1 7B
collection recorded NO actions (predates composite action logging), so its
pool is structurally empty — reported as-run with that reason, census only.

Activations rows at these scales exist historically (015/018) and are cited
with 026's corrected-baseline caveats, NOT rerun (028 T5).

    python scripts/t5_cross_scale.py --out results/t5_cross_scale.json
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

from feature_signal_gate import auroc, pair_distances  # noqa: E402
from offline_ask_headtohead import (forked_blockers, load_commitments,  # noqa: E402
                                    load_task_actions)
from wta.labeling import load_class_artifact  # noqa: E402

SCALES = (
    ("14b", "data/a0_v2", "models/t5_v2_14b_fixed/labels_debug.jsonl"),
    ("7b", "data/a0", "models/t5_a0_7b_fixed/labels_debug.jsonl"),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="results/t5_cross_scale.json")
    args = ap.parse_args()

    t0 = time.time()
    art = load_class_artifact(args.classes)
    minilm = None
    out = {}
    for name, a0, debug in SCALES:
        committed = load_commitments(Path(debug))
        actions = load_task_actions(Path(a0), None)
        actions = {t: r for t, r in actions.items() if t in art}
        n_runs = sum(len(r) for r in actions.values())
        forks = {t: forked_blockers(actions[t], committed, list(art[t]))
                 for t in actions}
        n_actions = sum(len(a) for runs in actions.values()
                        for a in runs.values())
        census = {
            "n_tasks": len(actions), "n_runs": n_runs,
            "n_action_events": n_actions,
            "n_commitments": len(committed),
            "n_forked_blockers": sum(len(f) for f in forks.values()),
            "n_forked_tasks": sum(1 for t in forks if forks[t]),
            "frac_tasks_forked": (sum(1 for t in forks if forks[t])
                                  / len(actions) if actions else None),
        }
        rows = {}
        for row, emb_name in (("r3_hashed", None), ("r4_minilm", "minilm")):
            emb = None
            if emb_name == "minilm":
                if n_actions == 0:
                    rows[row] = {"auroc": None, "n_same": 0, "n_diff": 0,
                                 "note": "no recorded actions in collection"}
                    continue
                if minilm is None:
                    from wta.embed import MiniLMEmbedder
                    minilm = MiniLMEmbedder()
                emb = minilm
            same, diff = pair_distances(actions, art, committed, emb)
            if len(same) == 0 or len(diff) == 0:
                rows[row] = {"auroc": None, "n_same": int(len(same)),
                             "n_diff": int(len(diff)),
                             "note": ("empty pair pool — no action-anchored "
                                      "commitments (v1 collection predates "
                                      "composite action logging)"
                                      if n_actions == 0 else
                                      "empty pair pool")}
                continue
            rows[row] = {"auroc": auroc(same, diff),
                         "n_same": int(len(same)), "n_diff": int(len(diff)),
                         "same_mean": float(same.mean()),
                         "diff_mean": float(diff.mean())}
        out[name] = {"a0": a0, "labels_debug": debug,
                     "census": census, "pair_pool": rows}
        print(f"{name}: {census['n_forked_tasks']}/{census['n_tasks']} tasks "
              f"forked, {census['n_forked_blockers']} forked blockers, "
              f"{census['n_commitments']} commitments; "
              + "; ".join(
                  f"{r} AUROC "
                  + (f"{v['auroc']:.3f} (n={v['n_same']}/{v['n_diff']})"
                     if v["auroc"] is not None else f"N/A ({v.get('note')})")
                  for r, v in rows.items())
              + f" ({time.time() - t0:.0f}s)")

    out["reference_32b"] = {"r3_hashed_auroc": 0.555,
                            "r4_minilm_auroc": 0.580,
                            "source": "results/feature_signal_gate.json"}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p} ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
