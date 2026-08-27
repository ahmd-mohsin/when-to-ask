"""DIAGNOSTIC (not a pre-registered cell): is the committed-class label
learnable from the trace at all?

Every T1 row asks whether some representation of a run separates
same-class from different-class commitment pairs. All of them land in the
same .54-.60 band, flat across a 15x parameter increase (R4 MiniLM .580 vs
R5 bge .573) and flat against a full-context LLM judge (R6b .579). A flat
band across methods of wildly different capacity is the signature of a
CEILING ON THE TARGET, not a failure of any one method.

This script tests that directly, by replacing the unfitted distance with a
FITTED supervised classifier and asking whether the label is recoverable at
all -- first from the exact text the r_vec encodes, then from the entire
transcript. Both are upper bounds on what any representation of that text
could achieve.

    python scripts/diagnose_label_learnability.py \
        --a0 /ssd/wta_data/a0_v3_32b_snap1385 \
        --labels models/v3_32b_fixed/labels_debug.jsonl \
        --out results/diag_label_learnability.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from offline_ask_headtohead import (commit_rounds, load_commitments,  # noqa: E402
                                    load_task_actions)
from wta.labeling import load_class_artifact  # noqa: E402


def score_cells(groups, analyzer, ngram, min_runs=6):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    accs, majs, ns = [], [], []
    for items in groups.values():
        y = [c for _, c in items]
        cnt = Counter(y)
        if len(cnt) < 2 or len(y) < min_runs or min(cnt.values()) < 2:
            continue
        X = [t for t, _ in items]
        maj = max(cnt.values()) / len(y)
        try:
            Xv = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                                 min_df=1, sublinear_tf=True,
                                 max_features=20000).fit_transform(X)
            cv = StratifiedKFold(n_splits=min(3, min(cnt.values())),
                                 shuffle=True, random_state=0)
            s = cross_val_score(
                LogisticRegression(max_iter=3000, class_weight="balanced"),
                Xv, y, cv=cv, scoring="accuracy").mean()
        except Exception:
            continue
        accs.append(s); majs.append(maj); ns.append(len(y))
    a, m = np.array(accs), np.array(majs)
    return {
        "n_cells": int(len(a)),
        "mean_runs_per_cell": float(np.mean(ns)) if ns else None,
        "fitted_cv_accuracy": float(a.mean()) if len(a) else None,
        "majority_baseline": float(m.mean()) if len(m) else None,
        "mean_lift": float((a - m).mean()) if len(a) else None,
        "cells_beating_majority": int((a > m).sum()) if len(a) else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="/ssd/wta_data/a0_v3_32b_snap1385")
    ap.add_argument("--labels", default="models/v3_32b_fixed/labels_debug.jsonl")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="results/diag_label_learnability.json")
    args = ap.parse_args()

    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels))
    a0 = Path(args.a0)
    actions = load_task_actions(a0, None)
    actions = {t: r for t, r in actions.items() if t in art}

    # (1) the exact text the r_vec encodes, at the commit round
    win = defaultdict(list)
    for task, runs in actions.items():
        rounds = commit_rounds(runs, committed, art, task)
        for blocker in art[task]:
            for rid, acts in runs.items():
                c = committed.get((rid, blocker))
                k = rounds.get((rid, blocker))
                if c is None or k is None:
                    continue
                ordered = sorted(acts, key=lambda a: a.segment_idx)
                if k >= len(ordered):
                    continue
                a = ordered[k]; obs = a.observables or {}
                win[(task, blocker)].append((" ".join([
                    str(obs.get("subgoal") or ""), a.action_text or "",
                    str(obs.get("error_signature") or "")]).strip(), c))

    # (2) the ENTIRE transcript
    full_txt = {f.stem: f.read_text(encoding="utf-8", errors="replace")
                for f in a0.glob("swe_*/*-s*.txt")}
    full = defaultdict(list)
    for (rid, blocker), c in committed.items():
        t = full_txt.get(rid)
        if t:
            full[(rid.split("-s")[0], blocker)].append((t, c))

    # (3) fork rate by sampling temperature -- epistemic vs aleatoric
    temp = {}
    for f in a0.glob("swe_*/*-s*.json"):
        if f.name.endswith(".segments.json"):
            continue
        try:
            d = json.loads(f.read_text())
            temp[d["run_id"]] = d.get("temperature")
        except Exception:
            pass
    by_tb = defaultdict(list)
    for (rid, blocker), c in committed.items():
        if temp.get(rid) is not None:
            by_tb[(rid.split("-s")[0], blocker)].append((temp[rid], c))
    per_t = defaultdict(lambda: [0, 0])
    for lst in by_tb.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                (t1, c1), (t2, c2) = lst[i], lst[j]
                if t1 != t2:
                    continue
                per_t[t1][1] += 1
                if c1 != c2:
                    per_t[t1][0] += 1

    out = {
        "DIAGNOSTIC": True,
        "question": ("is the committed-class label recoverable from the trace "
                     "by a FITTED model -- i.e. is there a ceiling on the "
                     "target rather than a failure of any one representation"),
        "window_text_the_r_vec_encodes": score_cells(win, "char_wb", (3, 5)),
        "entire_transcript": score_cells(full, "word", (1, 2)),
        "fork_rate_by_temperature": {
            str(t): {"n_pairs": v[1], "n_differing": v[0],
                     "rate": (v[0] / v[1] if v[1] else None)}
            for t, v in sorted(per_t.items())},
        "reference_unfitted_T1_rows": {
            "r3_hashed": 0.555, "r4_minilm": 0.580, "r5_bge": 0.573,
            "r6b_llm_judge": 0.5786},
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in ("window_text_the_r_vec_encodes", "entire_transcript"):
        v = out[k]
        print(f"{k}: fitted {v['fitted_cv_accuracy']:.3f} vs majority "
              f"{v['majority_baseline']:.3f}  lift {v['mean_lift']:+.3f} "
              f"({v['cells_beating_majority']}/{v['n_cells']} cells)")
    print("fork rate by temp:", {k: round(v["rate"], 3)
                                 for k, v in out["fork_rate_by_temperature"].items()
                                 if v["rate"] is not None})
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
