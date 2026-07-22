"""Human audit of labeling decisions (decisions/005's audit, mechanized).

    python scripts/audit_labels.py --a0 data/a0 --n 30 --out models/label_audit.md

Samples labeled AND unlabeled reads plus every commitment decision, and writes
a markdown file a human can eyeball in minutes: the text the labeler saw next
to the label it produced. This is the tool for "something doesn't add up" --
if the audit reads wrong, fix the artifact/lexicons and re-derive; never
patch labels downstream.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--n", type=int, default=30, help="sampled reads per outcome kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/label_audit.md")
    ap.add_argument("--tokenizer", default="auto",
                    help="'auto' = collection manifest's model_id")
    args = ap.parse_args()

    from wta.labeling import build_labels, coverage_table, resolve_tokenizer

    debug_path = Path(args.out).with_suffix(".debug.jsonl")
    ds = build_labels(args.a0, args.classes,
                      tokenizer_name=resolve_tokenizer(args.a0, args.tokenizer),
                      debug_path=debug_path)

    reads, commits = [], []
    for line in debug_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        (commits if row["kind"] == "commitment" else reads).append(row)

    rng = random.Random(args.seed)
    by_outcome: dict[str, list] = {}
    for r in reads:
        by_outcome.setdefault(r["outcome"], []).append(r)

    lines = ["# Label audit", "", "```", coverage_table(ds), "```", ""]

    lines.append("## Commitment decisions (ALL of them)\n")
    for c in sorted(commits, key=lambda x: (x["blocker"], x["run"])):
        if c.get("chosen"):
            lines.append(f"- **{c['run']} / {c['blocker']}** -> `{c['chosen']}` "
                         f"(scores {c['scores']}), commit snippet: "
                         f"`{(c.get('snippet') or '')[:150]}`")
        else:
            lines.append(f"- {c['run']} / {c['blocker']} -> UNLABELED "
                         f"({c['reason']}; scores {c['scores']})")
    lines.append("")

    for outcome, rows in sorted(by_outcome.items()):
        sample = rng.sample(rows, min(args.n, len(rows)))
        lines.append(f"## Reads: {outcome} ({len(rows)} total, {len(sample)} sampled)\n")
        for r in sample:
            lines.append(f"- **{r['run']}** read {r['read_idx']} (tok {r['token_idx']}) "
                         f"-> decision=`{r['decision']}` phase={r['phase']} "
                         f"anchors={r['anchor_scores']}")
            lines.append(f"  > …{(r['window_snippet'] or '').strip()}…")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"audit -> {out} ({len(commits)} commitment decisions, "
          f"{sum(len(v) for v in by_outcome.values())} reads; "
          f"full trail in {debug_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
