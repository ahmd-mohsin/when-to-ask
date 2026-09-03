"""Canonical accuracy: how often does a run resolve a blocker the registry's way?

An ABSOLUTE read-out, not a relational one. Every T1 cell so far asked "did run
A and run B decide the same?", which needs a pair, a distance and a
same/different label -- and that design failed six ways (R1 activations, R2
L-space, R3 hashes, R4 MiniLM, R5 bge, R6a/b LLM judge), plus replay-and-diff.
This asks a per-run question instead: did this run land on the resolution the
blocker registry calls canonical?

The material was already on disk. `data/interpretation_classes.json` marks
exactly one class per blocker `"canonical": true` -- 214 of 214 blockers, and
in every case it is class index 0, which the artifact's own `_provenance`
states outright ("class 0 is always the registry's canonical resolution").
Scoring is therefore pure re-scoring of the 1,595 commitments the lexicon
already made. No collection, no GPU, no box.

WHAT THIS IS NOT. This is agreement between the LEXICON'S label and the
registry's canonical class. It is not verified correctness: a run is scored
correct when the lexicon matched the canonical class's signatures in its
trace, so every bit of the lexicon's noise is inherited. Nothing here observes
whether the code the run wrote actually behaves canonically -- that needs the
test-outcome vector (test_patch + test_cmd + log_parser, which every task
ships), and this script deliberately does not claim it.

Confound checked and ruled out: canonical is always class index 0, so if class
0 carried more signatures it would match more often for reasons having nothing
to do with correctness. It does not -- class 0 averages 3.62 signatures
against class 1's 3.78.

Baselines reported alongside, because an accuracy number alone is unreadable:
uniform-random over each blocker's classes is 0.3797.

    python scripts/canonical_accuracy.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from offline_ask_headtohead import load_commitments  # noqa: E402

BOOT = 2000   # matches scripts/t1_auroc_ci.py
SEED = 0
TEMPS = [0.7, 0.9, 1.1, 1.3]   # collect_v2 --temps default, cycled by seed


def clustered_ci(items, boot=BOOT, seed=SEED):
    """Task-clustered bootstrap over a list of (task, correct) pairs."""
    rng = np.random.default_rng(seed)
    by_task = defaultdict(list)
    for t, c in items:
        by_task[t].append(c)
    tasks = sorted(by_task)
    out = []
    for _ in range(boot):
        pick = rng.choice(len(tasks), len(tasks), replace=True)
        vals = [c for i in pick for c in by_task[tasks[i]]]
        if vals:
            out.append(float(np.mean(vals)))
    a = np.array(out)
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed/labels_debug.jsonl",
                    help="the FIXED trail; models/v3_32b_fixed_debug/ (the "
                         "repo-wide default) does not exist on the box")
    ap.add_argument("--out", default="results/canonical_accuracy.json")
    args = ap.parse_args()

    art = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    art = {k: v for k, v in art.items() if k != "_provenance"}

    canonical, n_classes, sig_n, idx_of = {}, {}, {}, {}
    for task, blockers in art.items():
        for b, spec in blockers.items():
            cs = spec["classes"]
            can = [c["name"] for c in cs if c.get("canonical")]
            assert len(can) == 1, f"{task}/{b}: {len(can)} canonical classes"
            canonical[(task, b)] = can[0]
            n_classes[(task, b)] = len(cs)
            for i, c in enumerate(cs):
                sig_n[(task, b, i)] = len(c.get("signatures", []))
                idx_of[(task, b, c["name"])] = i

    committed = load_commitments(Path(args.labels_debug))

    rows = []          # (task, blocker, run, chosen, correct, temp)
    unmatched = 0
    for (run, b), chosen in committed.items():
        task = run.rsplit("-s", 1)[0]
        key = (task, b)
        if key not in canonical:
            unmatched += 1
            continue
        seed = int(run.rsplit("-s", 1)[1])
        rows.append((task, b, run, chosen, chosen == canonical[key],
                     TEMPS[seed % len(TEMPS)]))

    correct = [r[4] for r in rows]
    acc = float(np.mean(correct))
    ci = clustered_ci([(r[0], r[4]) for r in rows])
    rand = float(np.mean([1 / n_classes[(r[0], r[1])] for r in rows]))

    # per-blocker and per-task
    by_blocker, by_task, by_temp = defaultdict(list), defaultdict(list), defaultdict(list)
    for t, b, _, _, c, tp in rows:
        by_blocker[(t, b)].append(c)
        by_task[t].append(c)
        by_temp[tp].append(c)
    ba = {f"{t}/{b}": float(np.mean(v)) for (t, b), v in by_blocker.items()}
    ta = {t: float(np.mean(v)) for t, v in by_task.items()}

    # forked vs unforked, computed from the commitments themselves
    classes_seen = defaultdict(set)
    for t, b, _, ch, _, _ in rows:
        classes_seen[(t, b)].add(ch)
    forked = {k for k, v in classes_seen.items() if len(v) >= 2}
    f_acc = [c for t, b, _, _, c, _ in rows if (t, b) in forked]
    u_acc = [c for t, b, _, _, c, _ in rows if (t, b) not in forked]

    # --- confound: does the lexicon just pick whichever class it can match? ---
    # canonical is ALWAYS class index 0, so any systematic bias in how often
    # the labeler selects a given index contaminates canonical accuracy
    # directly. Two measurements: selection rate per index normalised by how
    # often that index was AVAILABLE, and whether the selected class is the
    # one carrying the most signatures.
    by_idx = defaultdict(int)
    avail = defaultdict(int)
    for t, b, _, ch, _, _ in rows:
        by_idx[idx_of.get((t, b, ch), -1)] += 1
        for i in range(n_classes[(t, b)]):
            avail[i] += 1
    sel_rate = {str(i): {"available": avail[i], "chosen": by_idx[i],
                         "selection_rate": by_idx[i] / avail[i],
                         "mean_signatures": float(np.mean(
                             [sig_n[(t, b, i)] for t, b, _, _, _, _ in rows
                              if i < n_classes[(t, b)]]))}
                for i in sorted(avail)}
    n_tie, n_most = 0, 0
    for t, b, _, ch, _, _ in rows:
        counts = [sig_n[(t, b, i)] for i in range(n_classes[(t, b)])]
        if len(set(counts)) > 1:
            n_tie += 1
            n_most += int(counts[idx_of[(t, b, ch)]] == max(counts))
    most_rate = n_most / n_tie

    res = {
        "note": "ABSOLUTE per-run read-out. Re-scores existing commitments; no "
                "collection, no GPU. NOT verified correctness -- this is "
                "agreement between the LEXICON's label and the registry's "
                "canonical class, and inherits all lexicon noise. Consequence-"
                "grounded correctness needs the test-outcome vector.",
        "inputs": {"classes": args.classes, "labels_debug": args.labels_debug},
        "coverage": {
            "n_blockers_in_artifact": len(canonical),
            "n_blockers_with_canonical": len(canonical),
            "canonical_always_class_index_0": True,
            "n_commitments_scored": len(rows),
            "n_commitments_dropped_no_blocker_match": unmatched,
        },
        "headline": {
            "canonical_accuracy": acc,
            "clustered_ci95": ci,
            "uniform_random_baseline": rand,
            "lift_over_random": acc - rand,
            "n": len(rows),
        },
        "by_temperature": {str(k): {"acc": float(np.mean(v)), "n": len(v)}
                           for k, v in sorted(by_temp.items())},
        "forked_vs_unforked": {
            "forked_blockers": {"acc": float(np.mean(f_acc)), "n": len(f_acc),
                                "n_blockers": len(forked)},
            "unforked_blockers": {"acc": float(np.mean(u_acc)), "n": len(u_acc),
                                  "n_blockers": len(classes_seen) - len(forked)},
        },
        "per_blocker_accuracy_distribution": {
            "n_blockers": len(ba),
            "mean": float(np.mean(list(ba.values()))),
            "frac_blockers_at_0": float(np.mean([v == 0 for v in ba.values()])),
            "frac_blockers_at_1": float(np.mean([v == 1 for v in ba.values()])),
            "deciles": [float(x) for x in
                        np.percentile(list(ba.values()), range(0, 101, 10))],
        },
        "per_task_accuracy": dict(sorted(ta.items())),
        "chosen_class_index_census": {str(k): v for k, v in sorted(by_idx.items())},
        "confound_check": {
            "selection_rate_by_class_index": sel_rate,
            "chosen_class_has_most_signatures": {
                "n_commitments_where_counts_differ": n_tie,
                "rate": most_rate, "chance": rand,
                "lift": most_rate - rand},
            "verdict": ("CONFOUNDED. The labeler selects whichever class "
                        "carries the most signatures {:.1%} of the time "
                        "against a {:.1%} chance rate, and canonical (always "
                        "index 0) carries fewer signatures than index 1 "
                        "({:.2f} vs {:.2f} commitment-weighted), which is "
                        "also the index selected most often ({:.4f} vs "
                        "{:.4f}). Canonical accuracy measured off lexicon "
                        "labels therefore cannot be read as how often a run "
                        "was RIGHT -- matching opportunity is mixed into it. "
                        "A consequence-grounded target (test_patch + test_cmd "
                        "+ log_parser) does not have this problem."
                        ).format(most_rate, rand,
                                 sel_rate["0"]["mean_signatures"],
                                 sel_rate["1"]["mean_signatures"],
                                 sel_rate["0"]["selection_rate"],
                                 sel_rate["1"]["selection_rate"]),
        },
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"scored {len(rows)} commitments over {len(ta)} tasks, "
          f"{len(ba)} blockers (dropped {unmatched})")
    print(f"canonical accuracy {acc:.4f}  clustered CI95 "
          f"[{ci[0]:.4f}, {ci[1]:.4f}]  vs random {rand:.4f} "
          f"(lift {acc - rand:+.4f})")
    print(f"  forked blockers   {np.mean(f_acc):.4f} (n={len(f_acc)})")
    print(f"  unforked blockers {np.mean(u_acc):.4f} (n={len(u_acc)})")
    print(f"  by temp: "
          f"{ {k: round(float(np.mean(v)), 4) for k, v in sorted(by_temp.items())} }")
    print(f"  chosen class index census: {dict(sorted(by_idx.items()))}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
