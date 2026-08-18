"""Are the artifact's signatures RESOLUTION-discriminative, or do they just
match the topic? (decisions/025 Amendment A diagnostic, 2026-08-17.)

Motivating case, proven from raw command text with no model in the loop:
swe_19/contradictory_remove_vs_deprecate_unsafeproxy. Its signature
"class unsafeproxy" (for keep_class_with_identity_preservation) is composed
of the blocker's OWN anchors ("class", "unsafeproxy"), so it also matches
    sed -i '/class UnsafeProxy:/,/^$/d' lib/ansible/utils/unsafe_proxy.py
which DELETES the class. The lexicon labels 15/15 runs "keep" -- a genuine
3-way decision collapses to a unanimous non-fork.

That failure mode does not cost coverage (the run IS labelled); it destroys
FORKS, and a decision with one committed class drops out of gate 5 entirely.
This audit counts how often the pattern occurs, model-free:

  weak win     = winning class scored 1 with margin 1 over the runner-up
  anchor-built = the winning signature contains, or is contained by, one of
                 the blocker's own anchors (topic match, not resolution)
  SUSPECT      = a blocker whose labels are unanimous AND mostly weak
                 AND anchor-built -> a fork the lexicon may be erasing

    python scripts/audit_signature_discrimination.py --a0 data/a0_v3_32b
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.labeling import _hits, _is_mutating, _norm, load_class_artifact  # noqa: E402


def anchor_built(sig: str, anchors: list[str]) -> bool:
    s = _norm(sig)
    return any(_norm(a) in s or s in _norm(a) for a in anchors)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="results/signature_discrimination.json")
    args = ap.parse_args()

    a0 = Path(args.a0)
    art = load_class_artifact(args.classes)

    # cache mutating-action text per run once
    mut: dict[str, dict[str, str]] = {}
    for task_dir in sorted(p for p in a0.iterdir() if p.is_dir()):
        if task_dir.name not in art:
            continue
        per = {}
        for jf in sorted(task_dir.glob("*.json")):
            if jf.name.endswith(".segments.json"):
                continue
            log = json.loads(jf.read_text(encoding="utf-8"))
            texts = [a.get("action_text", "") for a in log.get("actions", [])
                     if _is_mutating(a.get("action_text", ""))]
            per[jf.stem] = _norm("\n".join(texts))
        mut[task_dir.name] = per

    rows, totals = [], Counter()
    for task in sorted(k for k in art if not k.startswith("_")):
        for bid, spec in art[task].items():
            names = [c["name"] for c in spec["classes"]]
            labels, weak, built = Counter(), 0, 0
            for run_id, mn in mut.get(task, {}).items():
                if not mn:
                    continue
                sc = [_hits(mn, c["signatures"]) for c in spec["classes"]]
                order = sorted(range(len(sc)), key=lambda i: -sc[i])
                top, runner = sc[order[0]], sc[order[1]]
                if top < 1 or top <= runner:
                    continue
                win = names[order[0]]
                labels[win] += 1
                if top == 1 and runner == 0:
                    weak += 1
                    hit_sigs = [s for s in spec["classes"][order[0]]["signatures"]
                                if _norm(s) in mn]
                    if hit_sigs and anchor_built(hit_sigs[0], spec["anchors"]):
                        built += 1
            n = sum(labels.values())
            if not n:
                continue
            unanimous = len(labels) == 1
            suspect = bool(unanimous and n >= 4
                           and weak / n >= 0.5 and built / n >= 0.5)
            totals["blockers_with_labels"] += 1
            totals["forked"] += int(len(labels) >= 2)
            totals["unanimous"] += int(unanimous)
            totals["suspect_erased_fork"] += int(suspect)
            rows.append({"task": task, "blocker": bid, "n_labeled": n,
                         "classes": dict(labels), "unanimous": unanimous,
                         "weak_share": round(weak / n, 3),
                         "anchor_built_share": round(built / n, 3),
                         "suspect": suspect})

    rows.sort(key=lambda r: (-int(r["suspect"]), -r["n_labeled"]))
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps({"totals": dict(totals),
                                          "rows": rows}, indent=1),
                              encoding="utf-8")
    print(json.dumps(dict(totals), indent=1))
    print()
    print("SUSPECT blockers (unanimous label resting on weak, anchor-built "
          "signature hits -- a fork the lexicon may be erasing):")
    for r in rows:
        if not r["suspect"]:
            continue
        print(f"  {r['task']}/{r['blocker'][:52]}  n={r['n_labeled']:3d}  "
              f"weak={r['weak_share']:.2f} built={r['anchor_built_share']:.2f} "
              f"-> {list(r['classes'])[0][:34]}")
    print(f"\nfull report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
