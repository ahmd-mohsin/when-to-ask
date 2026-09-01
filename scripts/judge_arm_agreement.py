"""028 Amendment G.3 read-out 1: SYMMETRIC agreement between the two
labellers on the commitments both of them label.

This is the statistic 025 should have computed and did not. That gate scored
the judge AGAINST lexicon labels and treated every disagreement as judge
error, producing a 0.765 that triggered a STOP -- even though blind
adjudication of all 46 disagreements later went 46-0 FOR the judge
(025 Am.A (ii)), and 025 Am.A (iv) itself concluded "a yardstick built from
lexicon labels cannot settle a dispute about lexicon labels".

Here neither labeller is privileged. Reported: raw agreement, Cohen's kappa
(chance-corrected, per-blocker class inventory), and the marginals that show
which labeller abstains more. Kappa is computed BOTH pooled and macro-averaged
over blockers, because per-blocker class counts differ and a pooled kappa over
heterogeneous label spaces is not interpretable on its own.

Runs against any judge freeze; --judge-labels may be repeated so phase 1 and
phase 2 can be pooled once both exist.

    python scripts/judge_arm_agreement.py \
        --judge-labels data/judge_labels_v3_32b_phase2.jsonl \
        --out results/judge_arm_agreement.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_lexicon(debug_path: str | Path) -> dict:
    out = {}
    for line in Path(debug_path).open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") == "commitment" and row.get("chosen"):
            out[(row["run"], row["blocker"])] = row["chosen"]
    return out


def load_judge(paths: list[str]) -> tuple[dict, Counter]:
    """{(run, blocker): class} for accepted labels, plus a status census."""
    out, status = {}, Counter()
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            continue
        for line in fp.open(encoding="utf-8"):
            r = json.loads(line)
            status[r.get("status")] += 1
            if r.get("status") == "accepted":
                out[(r["run"], r["blocker"])] = r["class"]
    return out, status


def cohen_kappa(a: list, b: list) -> float | None:
    """Chance-corrected agreement over the union of observed categories."""
    if not a:
        return None
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return None                      # degenerate: one class, kappa undefined
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float((m.sum(0) / n) @ (m.sum(1) / n))
    if abs(1.0 - pe) < 1e-12:
        return None
    return float((po - pe) / (1.0 - pe))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--judge-labels", action="append", required=True,
                    help="frozen judge label jsonl; repeat to pool phases")
    ap.add_argument("--out", default="results/judge_arm_agreement.json")
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon)
    judge, status = load_judge(args.judge_labels)
    both = sorted(set(lex) & set(judge))
    print(f"lexicon {len(lex)}; judge accepted {len(judge)}; BOTH commit {len(both)}")
    if not both:
        print("no overlap yet -- phase 2 has not landed")

    a = [lex[k] for k in both]
    b = [judge[k] for k in both]
    raw = float(np.mean([x == y for x, y in zip(a, b)])) if both else None

    per_blocker, kappas = {}, []
    by_blk = defaultdict(list)
    for k in both:
        by_blk[k[1]].append(k)
    for blk, keys in sorted(by_blk.items()):
        aa = [lex[k] for k in keys]
        bb = [judge[k] for k in keys]
        kp = cohen_kappa(aa, bb)
        per_blocker[blk] = {
            "n": len(keys),
            "raw_agreement": float(np.mean([x == y for x, y in zip(aa, bb)])),
            "kappa": kp,
        }
        if kp is not None and len(keys) >= 5:
            kappas.append(kp)

    # where they disagree, what does each side pick -- the audit hook
    disagreements = [{"run": k[0], "blocker": k[1],
                      "lexicon": lex[k], "judge": judge[k]}
                     for k in both if lex[k] != judge[k]]

    out = {
        "READOUT": "028 Am.G.3 (1): symmetric labeller agreement",
        "framing": ("agreement between two INDEPENDENT labellers; neither is "
                    "ground truth. NOT a judge error rate -- see 025 Am.A(iv)"),
        "judge_label_sources": args.judge_labels,
        "judge_freeze_status": dict(sorted(status.items())),
        "n_lexicon_labeled": len(lex),
        "n_judge_accepted": len(judge),
        "n_both_commit": len(both),
        "raw_agreement": raw,
        "kappa_pooled": cohen_kappa(a, b),
        "kappa_macro_over_blockers": (float(np.mean(kappas)) if kappas else None),
        "n_blockers_in_macro": len(kappas),
        "n_disagreements": len(disagreements),
        "per_blocker": per_blocker,
        "disagreements_sample": disagreements[:50],
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in ("n_both_commit", "raw_agreement", "kappa_pooled",
              "kappa_macro_over_blockers", "n_disagreements"):
        print(f"  {k}: {out[k]}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
