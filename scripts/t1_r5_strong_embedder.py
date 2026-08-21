"""R5 (decisions/028 T1): the strong-embedder row of the representation matrix.

Same frozen commitment-pair pool, same pair_distances/auroc machinery as the
027 gate (scripts/feature_signal_gate.py), with BAAI/bge-large-en-v1.5 as the
resolution-vector embedder (wta.embed.BgeEmbedder: CLS pooling, l2 norm,
truncation 256 to match the MiniLM row — the encoder is the only change).

v1 (hashed) and v2 (MiniLM) are recomputed alongside as consistency checks
against the recorded 0.555 / 0.580; the bge number is a NEW T1 row, reported
as-run. There is NO gate here — 027 Amendment B already locked the fallback;
this row measures how far a strong generic embedder gets.

    python scripts/t1_r5_strong_embedder.py \
        --out results/t1_r5_strong_embedder.json
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
from offline_ask_headtohead import (load_commitments,  # noqa: E402
                                    load_task_actions)
from wta.labeling import load_class_artifact  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--out", default="results/t1_r5_strong_embedder.json")
    args = ap.parse_args()

    t0 = time.time()
    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels_debug))
    actions = load_task_actions(Path(args.a0), None)
    actions = {t: r for t, r in actions.items() if t in art}
    print(f"tasks: {len(actions)}; commitments: {len(committed)} "
          f"({time.time() - t0:.0f}s)")

    def make_bge():
        from wta.embed import BgeEmbedder
        return BgeEmbedder()

    def make_minilm():
        from wta.embed import MiniLMEmbedder
        return MiniLMEmbedder()

    out = {}
    for name, make in (("v1_hashed", lambda: None),
                       ("v2_minilm", make_minilm),
                       ("r5_bge_large", make_bge)):
        same, diff = pair_distances(actions, art, committed, make())
        a = auroc(same, diff)
        out[name] = {"auroc": a, "n_same": len(same), "n_diff": len(diff),
                     "same_mean": float(same.mean()),
                     "diff_mean": float(diff.mean())}
        print(f"{name:14s}: AUROC {a:.3f}  same {same.mean():.3f} "
              f"(n={len(same)})  diff {diff.mean():.3f} (n={len(diff)}) "
              f"({time.time() - t0:.0f}s)")

    out["consistency"] = {
        "v1_recorded": 0.555, "v2_recorded": 0.580,
        "v1_reproduced": round(out["v1_hashed"]["auroc"], 3),
        "v2_reproduced": round(out["v2_minilm"]["auroc"], 3),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
