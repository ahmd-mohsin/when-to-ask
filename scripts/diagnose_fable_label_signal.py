"""DIAGNOSTIC (not a pre-registered cell): does the T1 separation signal look
different under the FABLE judge labels than under the lexicon labels, on the
only items where both exist?

Context: the 025 judge validation produced 292 Fable judgments (196 accepted)
on lexicon-labelled commitments before its pre-registered STOP. Blind
adjudication of all 46 disagreements went 46-0 for the judge, so the owner is
considering a judge-labelled arm. Before spending anything on production
labelling, this asks the cheapest version of the question: on the SAME runs,
SAME windows, SAME vectors, does relabelling by Fable move the same-vs-diff
AUROC?

Design choices, for comparability:
  * The r_vec windows come from the LEXICON commit rounds in both arms
    (commit_rounds with the lexicon labels). Vectors are therefore identical;
    ONLY the pair labelling (same-class vs different-class) changes.
  * Item set = (run, blocker) with a lexicon label AND an accepted Fable
    label AND a valid action-committed round. This is the intersection, so
    the two arms score the exact same pair universe.
  * Representations: hashed char-3grams (T1 R3) and MiniLM (T1 R4), the two
    cheapest rows, via the unchanged feature_signal_gate machinery.
  * Task-clustered bootstrap CI, 2000 draws, seed 0, mirroring
    scripts/t1_auroc_ci.py's constants.

Known bias, disclosed up front: the 300-item validation sample was drawn from
lexicon-LABELABLE commitments (60% from the known-noisy trace stratum), so
this set is not distributionally representative of production items, and n is
small. This diagnostic can motivate or kill a judge-label arm; it cannot
replace one.

    python scripts/diagnose_fable_label_signal.py \
        --out results/diag_fable_label_signal.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from feature_signal_gate import auroc, behavior_features  # noqa: E402
from offline_ask_headtohead import (commit_rounds, load_commitments,  # noqa: E402
                                    load_task_actions)
from wta.labeling import load_class_artifact  # noqa: E402

N_BOOT = 2000
SEED = 0


def load_fable_labels(vdir: Path) -> dict:
    """{(run_id, blocker): class_name} for accepted (non-abstained) judgments."""
    items = {}
    with (vdir / "items.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            items[d["custom_id"]] = d
    out = {}
    for f in sorted(glob.glob(str(vdir / "session_results*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                c = r.get("class")
                it = items.get(r.get("custom_id"))
                if not c or it is None:
                    continue
                if c not in it["class_names"]:
                    continue  # judge named a class outside the schema
                out[(it["run_id"], it["blocker"])] = c
    return out


def pairs_by_task(vecs_by_cell, labels):
    """[(task, dist, is_diff)] over all within-cell pairs under `labels`."""
    rows = []
    for (task, blocker), items in vecs_by_cell.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (rid_a, va), (rid_b, vb) = items[i], items[j]
                ca, cb = labels[(rid_a, blocker)], labels[(rid_b, blocker)]
                rows.append((task, float(np.linalg.norm(va - vb)), ca != cb))
    return rows


def score(rows):
    same = np.array([d for _, d, x in rows if not x])
    diff = np.array([d for _, d, x in rows if x])
    if not len(same) or not len(diff):
        return {"auroc": None, "n_same": int(len(same)), "n_diff": int(len(diff))}
    a = auroc(same, diff)
    # task-clustered bootstrap (mirrors t1_auroc_ci.py: resample tasks)
    tasks = sorted({t for t, _, _ in rows})
    by_task = defaultdict(list)
    for t, d, x in rows:
        by_task[t].append((d, x))
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(tasks), size=len(tasks), replace=True)
        s, f = [], []
        for k in pick:
            for d, x in by_task[tasks[k]]:
                (f if x else s).append(d)
        if s and f:
            boots.append(auroc(np.array(s), np.array(f)))
    lo, hi = (float(np.percentile(boots, 2.5)),
              float(np.percentile(boots, 97.5))) if boots else (None, None)
    return {"auroc": float(a), "ci95": [lo, hi], "n_same": int(len(same)),
            "n_diff": int(len(diff)), "n_boot_valid": len(boots)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--judge-dir", default="models/label_judge_validation")
    ap.add_argument("--out", default="results/diag_fable_label_signal.json")
    args = ap.parse_args()

    art = load_class_artifact(args.classes)
    lex = load_commitments(Path(args.labels_debug))
    fable = load_fable_labels(Path(args.judge_dir))
    actions = load_task_actions(Path(args.a0), None)
    actions = {t: r for t, r in actions.items() if t in art}
    print(f"lexicon commitments: {len(lex)}; accepted fable labels: {len(fable)}")

    # the intersection item set, windows from the LEXICON rounds
    keep = defaultdict(list)  # (task, blocker) -> [run_id]
    for task, runs in actions.items():
        rounds = commit_rounds(runs, lex, art, task)
        for blocker in art[task]:
            for rid in runs:
                k = rounds.get((rid, blocker))
                if (k is not None and (rid, blocker) in fable
                        and (rid, blocker) in lex):
                    keep[(task, blocker)].append((rid, k))
    keep = {c: v for c, v in keep.items() if len(v) >= 2}
    n_items = sum(len(v) for v in keep.values())
    agree = sum(lex[(r, c[1])] == fable[(r, c[1])]
                for c, v in keep.items() for r, _ in v)
    print(f"cells with >=2 items: {len(keep)}; items {n_items}; "
          f"fable-lexicon agreement on them {agree}/{n_items}")

    def cell_vectors(r_embedder):
        out = {}
        for (task, blocker), v in keep.items():
            feats = {}
            for rid, k in v:
                if rid not in feats:
                    feats[rid] = behavior_features(actions[task][rid],
                                                   r_embedder=r_embedder)
            out[(task, blocker)] = [(rid, feats[rid][k].r_vec)
                                    for rid, k in v
                                    if k < len(feats[rid])]
        return {c: it for c, it in out.items() if len(it) >= 2}

    def make_minilm():
        from wta.embed import MiniLMEmbedder
        return MiniLMEmbedder()

    out = {
        "DIAGNOSTIC": True,
        "question": ("on the items where both a lexicon and an accepted Fable "
                     "judge label exist, does relabelling change the T1 "
                     "same-vs-diff separation? Windows/vectors identical "
                     "(lexicon commit rounds); only pair labels differ."),
        "item_set": {"cells": len(keep), "items": n_items,
                     "fable_lexicon_agreement": agree / n_items if n_items else None},
    }
    forked = {}
    for name, mk in (("hashed_r3", lambda: None), ("minilm_r4", make_minilm)):
        vecs = cell_vectors(mk())
        arm = {}
        for lab_name, labels in (("lexicon", lex), ("fable", fable)):
            rows = pairs_by_task(vecs, labels)
            arm[lab_name] = score(rows)
            forked[lab_name] = len({c for (t, b), items in vecs.items()
                                    for c in [b]
                                    if len({labels[(r, b)] for r, _ in items}) > 1})
        out[name] = arm
        for lab_name in ("lexicon", "fable"):
            s = arm[lab_name]
            ci = s.get("ci95")
            print(f"{name:10s} {lab_name:8s}: AUROC "
                  f"{s['auroc'] if s['auroc'] is None else round(s['auroc'], 3)} "
                  f"CI {ci}  same/diff {s['n_same']}/{s['n_diff']}")
    out["forked_cells"] = forked
    out["disclosed_bias"] = (
        "validation items were sampled from lexicon-LABELABLE commitments "
        "(60% trace stratum); not representative of production items; small n")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
