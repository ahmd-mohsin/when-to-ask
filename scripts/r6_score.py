"""R6 scoring (decisions/028 T1): turn collected chunk judgments into the
R6a detection-F1 row and the R6b separation-AUROC row.

Inputs: the frozen item sets (results/r6_items/) and the chunk result files
written by the in-session runner (results/r6_chunks/r6a_chunk_*.json /
r6b_chunk_*.json, each a list of {item_id, ask|different,
unresolved_decision?, confidence}).

Scoring (frozen):
- R6a: task-level fire = ANY judged run of the task has ask=true; detection
  precision/recall/F1 vs census truth over tasks WITH >=1 judged item.
  Run-level accuracy and ask-rate reported alongside. Free-text
  unresolved_decision is carried through for qualitative audit, unscored.
- R6b: score = confidence if different else -confidence (higher = more
  different); AUROC with diff pairs as positives, computed as the exact
  Mann-Whitney U statistic with ties counted 0.5 (pre-run review finding:
  the rank formula used for continuous distances is not tie-aware, and
  these scores take only 10 discrete values, so ties dominate). Accuracy at
  the different/same decision also reported.
- STOP rule (028): the completed fraction is reported; missing items are
  listed, never silently dropped from the denominator of coverage.

    python scripts/r6_score.py --out results/r6_llm_cells.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_chunks(chunk_dir: Path, prefix: str) -> dict:
    """Merge stored chunk files; a duplicate item_id across files is a hard
    error (silent last-wins could mask a mis-stored chunk)."""
    out = {}
    for f in sorted(chunk_dir.glob(f"{prefix}_chunk_*.json")):
        if f.name.startswith("payload_"):
            continue
        for row in json.loads(f.read_text(encoding="utf-8")):
            iid = row["item_id"]
            if iid in out:
                raise SystemExit(f"duplicate judgment for {iid} in {f.name}")
            out[iid] = row
    return out


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Exact Mann-Whitney U with ties counted 0.5 — the scores here take
    only 10 discrete values, so a non-tie-aware rank formula is wrong."""
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", default="results/r6_items")
    ap.add_argument("--chunks-dir", default="results/r6_chunks")
    ap.add_argument("--out", default="results/r6_llm_cells.json")
    args = ap.parse_args()

    items_dir, chunks = Path(args.items_dir), Path(args.chunks_dir)
    out = {}

    # ---------------- R6a ----------------------------------------------
    a_items = json.loads((items_dir / "r6a_items.json")
                         .read_text(encoding="utf-8"))["items"]
    a_res = load_chunks(chunks, "r6a")
    judged = [i for i in a_items if i["item_id"] in a_res]
    missing = [i["item_id"] for i in a_items if i["item_id"] not in a_res]
    by_task: dict[str, dict] = {}
    run_correct, asks = [], []
    for it in judged:
        r = a_res[it["item_id"]]
        ask = bool(r["ask"])
        asks.append(ask)
        run_correct.append(ask == it["truth_needs_ask"])
        t = by_task.setdefault(it["task"],
                               {"truth": it["truth_needs_ask"],
                                "fired": False, "n": 0})
        t["n"] += 1
        t["fired"] = t["fired"] or ask
    fired = {t for t, v in by_task.items() if v["fired"]}
    truth = {t for t, v in by_task.items() if v["truth"]}
    universe = set(by_task)
    tp = len(fired & truth)
    p = tp / len(fired) if fired else 0.0
    r_ = tp / len(truth) if truth else 0.0
    out["r6a"] = {
        "n_items": len(a_items), "n_judged": len(judged),
        "completed_fraction": len(judged) / len(a_items) if a_items else 0,
        "missing_items": missing,
        "n_tasks_covered": len(universe),
        "task_detection": {
            "precision": p, "recall": r_,
            "f1": 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0.0,
            "n_fired": len(fired), "n_true": len(truth),
            "n_tasks": len(universe)},
        "run_level": {
            "accuracy": float(np.mean(run_correct)) if run_correct else None,
            "ask_rate": float(np.mean(asks)) if asks else None},
        "stage1_context": {"divergence_detector_f1": 0.507,
                           "random_budget_matched_f1": 0.582,
                           "always_ask_style_f1": 0.800},
    }

    # ---------------- R6b ----------------------------------------------
    b_items = json.loads((items_dir / "r6b_pairs.json")
                         .read_text(encoding="utf-8"))["items"]
    b_res = load_chunks(chunks, "r6b")
    judged_b = [i for i in b_items if i["item_id"] in b_res]
    missing_b = [i["item_id"] for i in b_items if i["item_id"] not in b_res]
    same_scores, diff_scores, correct = [], [], []
    for it in judged_b:
        r = b_res[it["item_id"]]
        d = bool(r["different"])
        s = float(r["confidence"]) * (1 if d else -1)
        (diff_scores if it["truth"] == "diff" else same_scores).append(s)
        correct.append(d == (it["truth"] == "diff"))
    out["r6b"] = {
        "n_items": len(b_items), "n_judged": len(judged_b),
        "completed_fraction": (len(judged_b) / len(b_items)
                               if b_items else 0),
        "missing_items": missing_b,
        "auroc": (auroc(np.array(same_scores), np.array(diff_scores))
                  if same_scores and diff_scores else None),
        "accuracy": float(np.mean(correct)) if correct else None,
        "n_same_judged": len(same_scores), "n_diff_judged": len(diff_scores),
        "matrix_context": {"r3_hashed_auroc": 0.555,
                           "r4_minilm_auroc": 0.580},
    }

    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk != "missing_items"}
                      for k, v in out.items()}, indent=1))
    p_ = Path(args.out)
    p_.parent.mkdir(parents=True, exist_ok=True)
    p_.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
