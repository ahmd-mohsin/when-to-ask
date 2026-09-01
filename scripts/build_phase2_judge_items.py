"""028 Amendment G phase 2: build judge work files for the commitments the
lexicon DID label.

Phase 1 judged the 3,361 items the lexicon abstained on. Phase 2 judges the
complementary set -- the 1,595 (run, blocker) pairs the lexicon labelled --
so that every commitment in the T1 pool carries a judge label produced
without reference to lexicon output. Two things follow that phase 1 could not
deliver:

  1. A fully judge-labelled arm, with no lexicon dependency anywhere.
  2. The first HONEST validity number for either labeller. 025's gate scored
     the judge against lexicon labels and counted disagreement as judge error
     (025 Am.A (iv): "a yardstick built from lexicon labels cannot settle a
     dispute about lexicon labels"). Phase 2 produces the same comparison as
     a symmetric AGREEMENT statistic between two independent labellers, which
     is what should have been measured.

Items are built by the SAME `build_judge_items` phase 1 used. That builder
sees only the class artifact, the registry description, and the trace -- the
lexicon's chosen class is structurally absent from the prompt, so the judge is
blind to it exactly as in phase 1.

    python scripts/build_phase2_judge_items.py --out models/v3_32b_judge_phase2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.judge_labels import (build_judge_items, estimate_cost,  # noqa: E402
                              make_batch_requests, write_item_files)


def labeled_pairs(debug_path: str | Path) -> list[tuple[str, str]]:
    """(run_id, blocker) pairs the lexicon teacher DID label -- the exact
    complement of judge_labels.unlabeled_pairs. Line-by-line for the same
    reason: snippets carry U+2028/U+2029 and splitlines() would shear rows."""
    pairs = []
    for line in Path(debug_path).open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") == "commitment" and row.get("chosen"):
            pairs.append((row["run"], row["blocker"]))
    return sorted(set(pairs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--debug", default="models/v3_32b_fixed_debug/labels_debug.jsonl",
                    help="the lexicon trail whose LABELLED pairs get re-judged")
    ap.add_argument("--out", default="models/v3_32b_judge_phase2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = labeled_pairs(args.debug)
    print(f"lexicon-labelled pairs in scope: {len(pairs)}")

    items = build_judge_items(args.a0, args.classes, pairs)
    with (out / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.__dict__, ensure_ascii=False) + "\n")
    files = write_item_files(items, out)
    est = estimate_cost(make_batch_requests(items))
    manifest = {"phase": 2, "n_items": len(items), "n_work_files": len(files),
                "policy": dict(Counter(it.policy for it in items)),
                "debug_source": str(args.debug),
                "scope": "commitments the lexicon DID label (complement of phase 1)",
                "estimate": est, "transport": "session (025 Amendment A)",
                "rater": "claude-fable-5, pinned (028 Am.G.2)"}
    (out / "judge_manifest.json").write_text(json.dumps(manifest, indent=1),
                                             encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
