"""Judge-label the lexicon-unlabelable commitments (spec judge_labels.md,
decisions/025 + Amendment A).

    # 1. build items + per-subagent work files from a lexicon debug trail
    python scripts/judge_label_commitments.py --build \
        --a0 data/a0_v3_32b --debug models/v3_32b_1415/labels_debug.jsonl \
        --out models/v3_32b_judge

    # 2. (session transport) Claude Fable 5 subagents consume the work files
    #    and append {custom_id, class, confidence, evidence, reasoning}
    #    records to <out>/session_results.jsonl

    # 3. freeze: acceptance gate (evidence verification) + artifact + audit
    python scripts/judge_label_commitments.py --freeze \
        --a0 data/a0_v3_32b --out models/v3_32b_judge \
        --labels-out data/judge_labels_v3_32b.jsonl

The judge fires only where the lexicon abstained; the frozen artifact feeds
generate_labels.py --judge-labels ... (the SEPARATE labelled arm, 025 §4b).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.judge_labels import (  # noqa: E402
    JudgeItem, build_judge_items, estimate_cost, freeze_results,
    make_batch_requests, session_responses, unlabeled_pairs, write_item_files,
)


def _load_items(out: Path) -> list[JudgeItem]:
    items = []
    for line in (out / "items.jsonl").open(encoding="utf-8"):
        items.append(JudgeItem(**json.loads(line)))
    return items


def _audit_md(records: list[dict], out_path: Path) -> None:
    by = Counter(r["status"] for r in records)
    lines = ["# Judge-label audit (spec judge_labels.md §4)", "",
             "## Status counts", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(by.items())]
    acc = [r for r in records if r["status"] == "accepted"]
    if acc:
        confs = sorted(r["confidence"] for r in acc)
        lines += ["", f"## Accepted: {len(acc)}",
                  f"- confidence min/median/max: {confs[0]:.2f} / "
                  f"{confs[len(confs) // 2]:.2f} / {confs[-1]:.2f}",
                  f"- at conf>=0.7: {sum(c >= 0.7 for c in confs)}   "
                  f"at conf>=0.9: {sum(c >= 0.9 for c in confs)}", "",
                  "## Sampled accepted labels (every ~40th; verify by eye)",
                  ""]
        for r in acc[::max(1, len(acc) // 40)]:
            lines.append(f"- {r['run']} / {r['blocker']} -> {r['class']} "
                         f"(conf {r['confidence']:.2f}, char {r['commit_char']})")
            ev = (r["evidence"] or "").replace("\n", " ")[:160]
            lines.append(f"  evidence: {ev}")
    rej = [r for r in records
           if r["status"] in ("bad_class", "evidence_not_found")]
    if rej:
        lines += ["", f"## Rejected (first 50 of {len(rej)})", ""]
        for r in rej[:50]:
            lines.append(f"- {r['run']} / {r['blocker']}: {r['status']} "
                         f"(class={r['class']})")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--debug", default="models/v3_32b_1415/labels_debug.jsonl",
                    help="lexicon debug trail to enumerate unlabeled pairs from")
    ap.add_argument("--out", default="models/v3_32b_judge")
    ap.add_argument("--labels-out", default="data/judge_labels_v3_32b.jsonl")
    ap.add_argument("--include-ties", action="store_true",
                    help="also judge 'tie between top classes' pairs "
                         "(OFF by default: 025 §4 scopes the no-signature-hit "
                         "items)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.build:
        pairs = unlabeled_pairs(args.debug, include_ties=args.include_ties)
        print(f"unlabeled pairs in scope: {len(pairs)}")
        items = build_judge_items(args.a0, args.classes, pairs)
        with (out / "items.jsonl").open("w", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it.__dict__, ensure_ascii=False) + "\n")
        files = write_item_files(items, out)
        est = estimate_cost(make_batch_requests(items))
        policy = Counter(it.policy for it in items)
        manifest = {"n_items": len(items), "n_work_files": len(files),
                    "policy": dict(policy), "include_ties": args.include_ties,
                    "debug_source": str(args.debug), "estimate": est,
                    "transport": "session (025 Amendment A)"}
        (out / "judge_manifest.json").write_text(
            json.dumps(manifest, indent=1), encoding="utf-8")
        print(json.dumps(manifest, indent=1))
        return 0

    if args.freeze:
        items = _load_items(out)
        results_path = out / "session_results.jsonl"
        records = [json.loads(line) for line in
                   results_path.open(encoding="utf-8")]
        responses = session_responses(records)
        frozen = freeze_results(items, responses, args.a0)
        with Path(args.labels_out).open("w", encoding="utf-8") as fh:
            for r in frozen:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        _audit_md(frozen, out / "judge_labels_audit.md")
        by = Counter(r["status"] for r in frozen)
        print(f"frozen: {args.labels_out}  audit: {out / 'judge_labels_audit.md'}")
        print("status counts:", dict(sorted(by.items())))
        return 0

    print("nothing to do: pass --build or --freeze")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
