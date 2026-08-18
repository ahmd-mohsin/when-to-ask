"""Validation gate for the labelling judge (spec judge_labels.md §6,
decisions/025 §4a + Amendment A(2)).

NOT scripts/validate_judge.py: that harness scores the ask-injection judge
(question -> blocker id) and cannot score a labelling judge. This one scores
the labelling judge blind against lexicon-labelled commitments.

    # 1. freeze the validation set (once; committed to git)
    python scripts/validate_label_judge.py --build-set \
        --debug models/v3_32b_1415/labels_debug.jsonl

    # 2. export per-subagent work files (SAME prompt builder as production)
    python scripts/validate_label_judge.py --export --a0 data/a0_v3_32b \
        --out models/label_judge_validation

    # 3. after the session subagents wrote <out>/session_results.jsonl:
    python scripts/validate_label_judge.py --score --a0 data/a0_v3_32b \
        --out models/label_judge_validation

PRE-REGISTERED PASS BAR (do not tune): accuracy >= 0.90 overall AND >= 0.90
on the actions-sourced stratum (non-abstained items); abstention <= 0.20.
Any miss -> STOP for owner review; no full labelling run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.judge_labels import (  # noqa: E402
    JudgeItem, build_judge_items, freeze_results, session_responses,
    write_item_files,
)

SET_PATH = Path("data/label_judge_validation.json")
SEED = 20260816
N_TOTAL = 300
PASS_ACC = 0.90
PASS_ACC_ACTIONS = 0.90
MAX_ABSTAIN = 0.20


def build_set(debug_path: str, force: bool) -> int:
    if SET_PATH.exists() and not force:
        print(f"REFUSING to overwrite frozen {SET_PATH} (use --force)")
        return 1
    trace, actions = [], []
    for line in Path(debug_path).open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") != "commitment" or not row.get("chosen"):
            continue
        rec = {"run": row["run"], "blocker": row["blocker"],
               "expected": row["chosen"],
               "label_source": row.get("label_source", "trace")}
        (trace if rec["label_source"] == "trace" else actions).append(rec)
    trace.sort(key=lambda r: (r["run"], r["blocker"]))
    actions.sort(key=lambda r: (r["run"], r["blocker"]))
    random.Random(SEED).shuffle(actions)
    items = trace + actions[:max(0, N_TOTAL - len(trace))]
    payload = {"_meta": {
        "spec": "specs/judge_labels.md §6", "seed": SEED,
        "built_from": str(debug_path),
        "composition": {"trace": len(trace),
                        "actions": len(items) - len(trace)},
        "pass_bar": {"accuracy": PASS_ACC,
                     "accuracy_actions": PASS_ACC_ACTIONS,
                     "max_abstention": MAX_ABSTAIN}},
        "items": items}
    SET_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"frozen {SET_PATH}: {len(items)} items "
          f"({len(trace)} trace + {len(items) - len(trace)} actions)")
    return 0


def _load_set() -> dict:
    return json.loads(SET_PATH.read_text(encoding="utf-8"))


def export(a0: str, classes: str, out: Path) -> int:
    vs = _load_set()
    pairs = [(r["run"], r["blocker"]) for r in vs["items"]]
    items = build_judge_items(a0, classes, pairs)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.__dict__, ensure_ascii=False) + "\n")
    files = write_item_files(items, out)
    print(f"exported {len(items)} validation items -> {len(files)} work files "
          f"in {out}")
    return 0


def score(a0: str, out: Path) -> int:
    vs = _load_set()
    expected = {(r["run"], r["blocker"]): r for r in vs["items"]}
    items = [JudgeItem(**json.loads(line)) for line in
             (out / "items.jsonl").open(encoding="utf-8")]
    records = [json.loads(line) for line in
               (out / "session_results.jsonl").open(encoding="utf-8")]
    frozen = freeze_results(items, session_responses(records), a0)

    n = len(frozen)
    abstained = [r for r in frozen if r["status"] == "abstained"]
    judged = [r for r in frozen if r["status"] == "accepted"]
    other = [r for r in frozen if r["status"] not in ("abstained", "accepted")]
    hits = {"all": [0, 0], "actions": [0, 0], "trace": [0, 0]}
    misses = []
    for r in judged:
        exp = expected[(r["run"], r["blocker"])]
        ok = r["class"] == exp["expected"]
        for key in ("all", exp["label_source"]):
            hits[key][0] += int(ok)
            hits[key][1] += 1
        if not ok:
            misses.append({**{k: r[k] for k in
                              ("run", "blocker", "class", "confidence")},
                           "expected": exp["expected"]})

    def rate(k):
        c, t = hits[k]
        return c / t if t else float("nan")

    abst = len(abstained) / n if n else 1.0
    report = {"n": n, "accepted": len(judged), "abstained": len(abstained),
              "other_status": len(other),
              "abstention_rate": round(abst, 4),
              "accuracy": round(rate("all"), 4),
              "accuracy_actions": round(rate("actions"), 4),
              "accuracy_trace": round(rate("trace"), 4),
              "strata_n": {k: hits[k][1] for k in hits},
              "misses": misses,
              "other": [{k: r[k] for k in ("run", "blocker", "status")}
                        for r in other]}
    passed = (rate("all") >= PASS_ACC and rate("actions") >= PASS_ACC_ACTIONS
              and abst <= MAX_ABSTAIN)
    report["pass"] = bool(passed)
    Path("results").mkdir(exist_ok=True)
    Path("results/label_judge_validation.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("misses", "other")}, indent=1))
    if passed:
        print("PASS: judge meets the pre-registered bar "
              "(record the numbers in decisions/025 Amendment A)")
        return 0
    print("STOP: judge below the pre-registered bar -- owner review before "
          "any full labelling run (decisions/025 §4a)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-set", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--debug", default="models/v3_32b_1415/labels_debug.jsonl")
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="models/label_judge_validation")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.build_set:
        return build_set(args.debug, args.force)
    if args.export:
        return export(args.a0, args.classes, Path(args.out))
    if args.score:
        return score(args.a0, Path(args.out))
    print("nothing to do: pass --build-set, --export, or --score")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
