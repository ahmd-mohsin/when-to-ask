"""Derive the blocker -> {structural|value} map that decisions/022 §2g freezes
in data/fork_type_annotations.json, from the interpretation-class artifact.

Emits a DRAFT plus per-blocker evidence, because §2g requires the map to be
hand-audited and owner-signed before the sealed pool is unsealed. This script
does the mechanical pass and, crucially, flags what it is unsure about instead
of silently guessing -- review the NEEDS-REVIEW tier by hand.

The distinction (021 §7): a VALUE fork's competing interpretations differ only
in a scalar/parameter value (limit 6 vs 10 vs 100); a STRUCTURAL fork's differ
in behaviour or code shape (remove the class vs keep it). This matters because
value forks were activation-invisible at 7B and 14B while structural forks
separated at raw-h 0.71-0.73, so pooling them buries the real effect.

    python scripts/derive_fork_types.py --out data/fork_type_annotations.draft.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

NUM = re.compile(r"\d+")


def strip_nums(s: str) -> str:
    return NUM.sub("#", s)


def classify(blocker: str, bd: dict) -> tuple[str, str, str]:
    """-> (kind, confidence, evidence). Confidence 'review' means: do not trust
    this one, a human must look."""
    names = [c["name"] for c in bd["classes"]]
    sigs = [s for c in bd["classes"] for s in c.get("signatures", [])]

    named_num = [n for n in names if NUM.search(n)]
    # names that collapse onto each other once digits are masked are the
    # signature of a pure value fork: same concept, different number
    collapsed = Counter(strip_nums(n) for n in names)
    collide = sum(v for v in collapsed.values() if v > 1)
    sig_num = sum(bool(NUM.search(s)) for s in sigs)
    sig_frac = sig_num / len(sigs) if sigs else 0.0

    ev = (f"names={len(names)} numeric_names={len(named_num)} "
          f"digit-collapsed_collisions={collide} sig_numeric={sig_num}/{len(sigs)}"
          f" ({sig_frac:.0%})")

    # strongest signal: >=2 class names differ only by a number
    if collide >= 2:
        return "value", "high", ev + " | >=2 class names differ only by a digit"
    # most names carry numbers and signatures are numeric-heavy
    if len(named_num) >= max(2, len(names) - 1) and sig_frac >= 0.5:
        return "value", "high", ev + " | numeric names + numeric-heavy signatures"
    # numeric-heavy signatures but non-numeric names: ambiguous
    if sig_frac >= 0.5:
        return "value", "review", ev + " | numeric signatures but names are not numeric"
    if len(named_num) >= 2:
        return "value", "review", ev + " | some numeric names, few numeric signatures"
    if sig_frac <= 0.2 and not named_num:
        return "structural", "high", ev + " | no numeric names, few numeric signatures"
    return "structural", "review", ev + " | weak signal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="data/fork_type_annotations.draft.json")
    ap.add_argument("--evidence", default="data/fork_type_annotations.evidence.json")
    args = ap.parse_args()

    art = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    ann: dict[str, str] = {}
    eviden: dict[str, dict] = {}
    tally: Counter = Counter()
    review: list[tuple[str, str, str, str]] = []

    for task, blockers in sorted(art.items()):
        if task == "_provenance":
            continue
        for blocker, bd in sorted(blockers.items()):
            kind, conf, ev = classify(blocker, bd)
            ann[blocker] = kind
            eviden[blocker] = {"task": task, "kind": kind, "confidence": conf,
                               "evidence": ev,
                               "classes": [c["name"] for c in bd["classes"]]}
            tally[(kind, conf)] += 1
            if conf == "review":
                review.append((task, blocker, kind, ev))

    Path(args.out).write_text(json.dumps(ann, indent=1, sort_keys=True) + "\n",
                              encoding="utf-8")
    Path(args.evidence).write_text(json.dumps(eviden, indent=1, sort_keys=True) + "\n",
                                   encoding="utf-8")

    total = sum(tally.values())
    print(f"{total} blockers")
    for (kind, conf), n in sorted(tally.items()):
        print(f"  {kind:11} {conf:6} {n:>4}  ({100*n/total:.0f}%)")
    hi = sum(n for (k, c), n in tally.items() if c == "high")
    print(f"\nhigh-confidence: {hi}/{total} ({100*hi/total:.0f}%) — "
          f"{total-hi} NEED HAND REVIEW before this can be signed")
    print(f"\nwrote {args.out} (DRAFT — not the frozen §2g artifact)")
    print(f"wrote {args.evidence}")
    print("\n--- NEEDS-REVIEW (hand-audit these) ---")
    for task, b, kind, ev in review[:25]:
        print(f"  {task:8} {b[:46]:46} -> {kind:11} {ev}")
    if len(review) > 25:
        print(f"  ... and {len(review)-25} more (see {args.evidence})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
