"""Judge-agreement validation (brief §5a; spec eval, decisions/022).

    JUDGE_BASE_URL=http://127.0.0.1:8808 python scripts/validate_judge.py

Runs every pair in data/judge_validation_pairs.json through BOTH judge
paths against the SAME vLLM endpoint:
  (a) ours  -- xtid.harness.judge.ApiJudge (the migrated prompt + flow)
  (b) theirs -- hil_bench.ask_human_server.AskHuman imported from the clone
      (ASK_HUMAN_PROVIDER=self_hosted), the exact code the bridge rows use.

Reports per-pair agreement and each side's accuracy against the frozen
expected labels. Runs on the GPU box BEFORE any ask row is trusted
(AWS_RUNBOOK step 5); a disagreement rate above --max-disagreement exits
non-zero -- STOP and paste the verbatim disagreements to the owner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from xtid.harness.tasks import Blocker  # noqa: E402

PAIRS_FILE = REPO / "data" / "judge_validation_pairs.json"


def load_pairs(path: Path = PAIRS_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["pairs"]


def compare(pairs: list[dict], ask_ours, ask_theirs) -> dict:
    """Pure comparison core (CPU-testable with fake judges).

    ask_ours / ask_theirs: (question, blockers: list[dict]) -> blocker_id|None
    """
    rows, agree, ours_ok, theirs_ok = [], 0, 0, 0
    for p in pairs:
        ours = ask_ours(p["question"], p["blockers"])
        theirs = ask_theirs(p["question"], p["blockers"])
        expected = p["expected"]
        rows.append({"case_id": p["case_id"], "kind": p["kind"],
                     "question": p["question"], "expected": expected,
                     "ours": ours, "theirs": theirs,
                     "agree": ours == theirs})
        agree += int(ours == theirs)
        ours_ok += int(ours == expected)
        theirs_ok += int(theirs == expected)
    n = len(pairs) or 1
    return {"n": len(pairs), "agreement": agree / n,
            "ours_accuracy": ours_ok / n, "theirs_accuracy": theirs_ok / n,
            "rows": rows,
            "disagreements": [r for r in rows if not r["agree"]]}


def _our_asker():
    from xtid.harness.judge import ApiJudge

    judge = ApiJudge()          # JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY

    def ask(question, blocker_dicts):
        blockers = [Blocker(id=b["id"], description=b.get("description", ""),
                            resolution=b.get("resolution", ""),
                            example_questions=list(b.get("example_questions", [])),
                            type=b.get("type")) for b in blocker_dicts]
        return judge.ask(question, blockers).blocker_id

    return ask


def _their_asker():
    base = os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8808")
    os.environ.setdefault("ASK_HUMAN_PROVIDER", "self_hosted")
    os.environ.setdefault("ASK_HUMAN_HOSTING_TYPE", "self_hosted")
    os.environ.setdefault("ASK_HUMAN_SELF_HOSTED_BASE_URL", base)
    sys.path.insert(0, str(REPO / "third_party" / "hil-bench"))
    from hil_bench.ask_human_server import BlockerEntry, AskHuman

    def ask(question, blocker_dicts):
        registry = {b["id"]: BlockerEntry(**b) for b in blocker_dicts}
        ah = AskHuman(registry)
        response = ah.ask_human(question)
        for b in blocker_dicts:                 # resolution -> id
            if response == b.get("resolution"):
                return b["id"]
        return None

    return ask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(PAIRS_FILE))
    ap.add_argument("--max-disagreement", type=float, default=0.1)
    ap.add_argument("--out", default="results/judge_validation.json")
    args = ap.parse_args()

    pairs = load_pairs(Path(args.pairs))
    print(f"{len(pairs)} pairs; endpoint "
          f"{os.environ.get('JUDGE_BASE_URL', 'http://127.0.0.1:8808')}")
    report = compare(pairs, _our_asker(), _their_asker())

    print(f"\nagreement {report['agreement']:.1%}  "
          f"ours-vs-expected {report['ours_accuracy']:.1%}  "
          f"theirs-vs-expected {report['theirs_accuracy']:.1%}")
    for r in report["disagreements"]:
        print(f"\nDISAGREE [{r['case_id']}] ({r['kind']})\n"
              f"  Q: {r['question']}\n  expected={r['expected']} "
              f"ours={r['ours']} theirs={r['theirs']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nreport -> {out}")
    if 1.0 - report["agreement"] > args.max_disagreement:
        print("STOP: disagreement above threshold -- owner review before any "
              "ask row is trusted (decisions/022)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
