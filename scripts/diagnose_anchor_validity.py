"""DIAGNOSTIC: is the judge EVIDENCE-SPAN anchor (028 G.1(1)) a faithful
stand-in for the lexicon's signature anchor?

The interim judge arm (scripts/interim_judge_arm.py) had to anchor its r_vec
window at the judge's verbatim evidence span, because commit_rounds is
undefined for judge-labelled items by construction. If that anchor is noisy,
a null in the judge arm is uninterpretable -- it could be the anchor, not the
labels.

The 025 validation set is the one place where both exist: those 292 items are
lexicon-LABELABLE (so commit_rounds is defined) AND carry judge evidence
spans. This script freezes them through the same acceptance gate, maps each
evidence span through the same turn_of_char logic, and compares the anchored
turn to the lexicon's commit round on the SAME (run, blocker).

Reported as-run. A high agreement rate licenses reading the judge arm's
separation number; a low one means the judge arm's null is confounded and
must be reported as such.

    python scripts/diagnose_anchor_validity.py --out results/diag_anchor_validity.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from interim_judge_arm import segment_offsets, turn_of_char  # noqa: E402
from offline_ask_headtohead import (commit_rounds, load_commitments,  # noqa: E402
                                    load_task_actions)
from wta.judge_labels import JudgeItem, freeze_results, session_responses  # noqa: E402
from wta.labeling import load_class_artifact  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--vdir", default="models/label_judge_validation")
    ap.add_argument("--out", default="results/diag_anchor_validity.json")
    args = ap.parse_args()

    a0, vdir = Path(args.a0), Path(args.vdir)
    art = load_class_artifact(args.classes)
    lex = load_commitments(Path(args.labels_debug))

    items = [JudgeItem(**json.loads(l))
             for l in (vdir / "items.jsonl").open(encoding="utf-8")]
    recs = []
    for f in sorted(glob.glob(str(vdir / "session_results*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    judged = {r["custom_id"] for r in recs}
    items = [it for it in items if it.custom_id in judged]
    frozen = freeze_results(items, session_responses(recs), a0)
    status = Counter(r["status"] for r in frozen)
    print("validation freeze status:", dict(sorted(status.items())))

    actions = load_task_actions(a0, None)
    actions = {t: r for t, r in actions.items() if t in art}
    lex_round = {}
    for task, runs in actions.items():
        for k_, v in commit_rounds(runs, lex, art, task).items():
            if v is not None:
                lex_round[k_] = v

    deltas, same_turn, n_cmp, class_agree = [], 0, 0, 0
    seg_cache = {}
    for r in frozen:
        if r["status"] != "accepted":
            continue
        rid, blk = r["run"], r["blocker"]
        task = rid.split("-s")[0]
        acts = actions.get(task, {}).get(rid)
        lk = lex_round.get((rid, blk))
        if not acts or lk is None:
            continue
        if rid not in seg_cache:
            seg_cache[rid] = segment_offsets(a0, task, rid)
        offs = seg_cache[rid]
        if not offs:
            continue
        jk = turn_of_char(offs, acts, r["commit_char"])
        if jk is None:
            continue
        n_cmp += 1
        deltas.append(jk - lk)
        same_turn += int(jk == lk)
        class_agree += int(r["class"] == lex.get((rid, blk)))

    d = np.array(deltas) if deltas else np.array([0])
    out = {
        "DIAGNOSTIC": True,
        "question": ("does the judge evidence-span anchor (028 G.1(1)) land "
                     "on the same turn the lexicon signature anchor picks, on "
                     "the 025 validation items where both are defined"),
        "validation_freeze_status": dict(sorted(status.items())),
        "n_comparable": n_cmp,
        "same_turn": same_turn,
        "same_turn_rate": (same_turn / n_cmp) if n_cmp else None,
        "within_1_turn_rate": float(np.mean(np.abs(d) <= 1)) if n_cmp else None,
        "delta_turns": {"mean": float(d.mean()), "median": float(np.median(d)),
                        "p10": float(np.percentile(d, 10)),
                        "p90": float(np.percentile(d, 90)),
                        "min": int(d.min()), "max": int(d.max())},
        "class_agreement_with_lexicon": (class_agree / n_cmp) if n_cmp else None,
        "note": ("class agreement here is the 025 disagreement rate, NOT an "
                 "error rate -- blind adjudication of those disagreements "
                 "went 46-0 for the judge (025 Am.A (ii))"),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "question"}, indent=1))
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
