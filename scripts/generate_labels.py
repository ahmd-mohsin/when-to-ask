"""Generate + STORE labels from an A0 collection (the standalone labeler).

    python scripts/generate_labels.py --a0 data/a0_v2_32b --out models/v2_32b

Runs the offline labeling teacher (wta.labeling.build_labels, spec labels.md)
and persists what every downstream stage loads: labels.npz + the full
labels_debug.jsonl audit trail. audit_labels.py is the human-eyeball sibling
(sampled snippets); train_offline.py rebuilds labels only as step one of the
full A1->A2->A3 pipeline. Use this one when the deliverable is the labels.

The tokenizer defaults to 'auto': the collection manifest's model_id, because
token->char maps are built by re-tokenizing the trace and token_idx in the
logs is in the collection model's units (a Qwen3 collection labeled with a
Qwen2.5 tokenizer drifts).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.labeling import build_labels, coverage_table, resolve_tokenizer  # noqa: E402


def label_source_summary(debug_path: Path) -> str:
    """Commitment label_source split (spec labels.md v2): action-sourced
    commitments are the trustworthy ones; a low actions share means the
    v1 whole-trace fallback is doing the work."""
    src, unlabeled = Counter(), 0
    for line in debug_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") != "commitment":
            continue
        if row.get("chosen"):
            src[row.get("label_source", "trace")] += 1
        else:
            unlabeled += 1
    total = sum(src.values())
    parts = ", ".join(f"{k}: {v}" for k, v in sorted(src.items()))
    return (f"commitments: {total} labeled ({parts}), {unlabeled} unlabeled "
            f"of {total + unlabeled} (run, decision) pairs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="models",
                    help="output dir for labels.npz + labels_debug.jsonl")
    ap.add_argument("--tokenizer", default="auto",
                    help="'auto' = collection manifest's model_id")
    ap.add_argument("--layer", type=int, default=None,
                    help="stored layer index (or position) to slice from "
                         "multi-layer logs; default = the collection mid_layer")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_name = resolve_tokenizer(args.a0, args.tokenizer)
    print(f"a0={args.a0}  classes={args.classes}  tokenizer={tokenizer_name}  "
          f"layer={'mid' if args.layer is None else args.layer}")

    debug_path = out / "labels_debug.jsonl"
    ds = build_labels(args.a0, args.classes, tokenizer_name=tokenizer_name,
                      debug_path=debug_path, layer=args.layer)
    ds.save(out / "labels.npz")

    print()
    print(coverage_table(ds))
    print()
    n = len(ds.h)
    dec, cls = int((ds.decision >= 0).sum()), int((ds.cls >= 0).sum())
    forked = sum(1 for c in ds.coverage.values()
                 for v in c["committed_classes"].values() if len(v) >= 2)
    print(f"reads: {n} total, {dec} decision-labeled ({dec / max(n, 1):.0%}), "
          f"{cls} class-labeled ({cls / max(n, 1):.0%})")
    print(f"runs: {len(ds.runs)} across {len(ds.tasks)} tasks; "
          f"forked blockers (>=2 committed classes): {forked}")
    print(label_source_summary(debug_path))
    print(f"\nstored: {out / 'labels.npz'} + {debug_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
