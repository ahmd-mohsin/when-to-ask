"""027 Amendment A feature-signal gate.

Do the resolution vectors separate SAME-class from DIFFERENT-class
commitments at all? Computes the separation AUROC on the commitment-pair
pool for v1 (hashed char-3grams) and v2 (MiniLM embeddings). The Amendment's
hard gate: v2 AUROC >= 0.75, else automatic fallback (no stage-1 rerun).

This is a feature-VALIDITY measure — no detection thresholds, no eval-fold
detection numbers are touched.

    python scripts/feature_signal_gate.py --out results/feature_signal_gate.json
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

from offline_ask_headtohead import (commit_rounds, load_commitments,  # noqa: E402
                                    load_task_actions)
from wta.divergence import behavior_features  # noqa: E402
from wta.labeling import load_class_artifact  # noqa: E402


def pair_distances(actions, art, committed, r_embedder=None):
    same, diff = [], []
    for task, runs in actions.items():
        feats = {rid: behavior_features(a, r_embedder=r_embedder)
                 for rid, a in runs.items()}
        rounds = commit_rounds(runs, committed, art, task)
        for blocker in art[task]:
            vecs: dict[str, list] = {}
            for rid in runs:
                c = committed.get((rid, blocker))
                k = rounds.get((rid, blocker))
                if c is not None and k is not None and k < len(feats[rid]):
                    vecs.setdefault(c, []).append(feats[rid][k].r_vec)
            classes = sorted(vecs)
            for i, ca in enumerate(classes):
                va = vecs[ca]
                for x in range(len(va)):
                    for y in range(x + 1, len(va)):
                        same.append(float(np.linalg.norm(va[x] - va[y])))
                for cb in classes[i + 1:]:
                    for u in vecs[ca]:
                        for v in vecs[cb]:
                            diff.append(float(np.linalg.norm(u - v)))
    return np.array(same), np.array(diff)


def auroc(same: np.ndarray, diff: np.ndarray) -> float:
    lab = np.r_[np.ones(len(diff)), np.zeros(len(same))]
    score = np.r_[diff, same]
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(len(score))
    return float((ranks[lab == 1].mean() - (lab == 1).sum() / 2 + 0.5)
                 / (lab == 0).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--gate", type=float, default=0.75)
    ap.add_argument("--out", default="results/feature_signal_gate.json")
    args = ap.parse_args()

    t0 = time.time()
    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels_debug))
    actions = load_task_actions(Path(args.a0), None)
    actions = {t: r for t, r in actions.items() if t in art}

    out = {}
    for name, embedder in (("v1_hashed", None), ("v2_minilm", "load")):
        if embedder == "load":
            from wta.embed import MiniLMEmbedder
            embedder = MiniLMEmbedder()
        same, diff = pair_distances(actions, art, committed, embedder)
        a = auroc(same, diff)
        out[name] = {"auroc": a, "n_same": len(same), "n_diff": len(diff),
                     "same_mean": float(same.mean()),
                     "diff_mean": float(diff.mean())}
        print(f"{name:10s}: AUROC {a:.3f}  same {same.mean():.3f} "
              f"(n={len(same)})  diff {diff.mean():.3f} (n={len(diff)}) "
              f"({time.time() - t0:.0f}s)")

    passed = out["v2_minilm"]["auroc"] >= args.gate
    out["gate"] = {"threshold": args.gate, "passed": bool(passed)}
    print(f"\nGATE (v2 AUROC >= {args.gate}): "
          f"{'PASS -> stage-1 rerun authorized' if passed else 'FAIL -> automatic fallback'}")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
