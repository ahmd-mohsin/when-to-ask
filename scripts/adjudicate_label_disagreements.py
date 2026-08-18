"""Blind adjudication of judge-vs-lexicon label disagreements (decisions/025
Amendment A diagnostic; method = decisions/024's contested-item adjudication).

The validation gate scores the judge against the LEXICON's labels, but the
lexicon's trace path is itself known-noisy (spec labels.md v2). So a
disagreement is not evidence of a judge error until something independent
says which side is right.

    python scripts/adjudicate_label_disagreements.py --build   # work files
    # (adjudicator subagents write session_results_<k>.jsonl)
    python scripts/adjudicate_label_disagreements.py --score    # tally

Blinding: the two candidate classes are presented as Option A / Option B in a
per-item deterministic order (sha256 of run|blocker), so the adjudicator
cannot tell which came from the lexicon and which from the judge, and cannot
infer it from position. The adjudicator may also answer "neither".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.judge_labels import build_judge_items  # noqa: E402

ADJ_SYSTEM = """\
You are adjudicating a disagreement about which interpretation an autonomous
coding agent's completed run COMMITTED to, for one pre-specified ambiguous
decision. Two independent labellers picked different classes. You decide which
reading the trace actually supports.

Respond with JSON only:
{"verdict": "A" | "B" | "neither",
 "confidence": <0.0-1.0>,
 "evidence": "<verbatim quote copied from the trace>",
 "reasoning": "<one or two sentences>"}

Rules:
- Commitment = what the run's edits, commands, or final work product actually
  implement or assert. Deliberation that merely MENTIONS an option is not
  commitment; weigh mutating actions (file edits, sed/patch/redirection) and
  the final work product above discussion.
- Answer "neither" if the trace supports some third reading, or if it never
  engages this decision, or if the evidence is genuinely balanced.
- Do not assume either option is more likely to be correct. They are presented
  in a randomized order and carry no provenance.
- "evidence" must be copied VERBATIM from the trace.
"""


def _ab_order(run_id: str, blocker: str) -> bool:
    """True => lexicon label is presented as Option A."""
    d = hashlib.sha256(f"adj|{run_id}|{blocker}".encode()).digest()
    return d[0] % 2 == 0


def build(report_path: Path, a0: str, classes: str, out: Path) -> int:
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    misses = rep["misses"]
    pairs = [(m["run"], m["blocker"]) for m in misses]
    items = build_judge_items(a0, classes, pairs)
    by_key = {(m["run"], m["blocker"]): m for m in misses}
    out.mkdir(parents=True, exist_ok=True)

    records, work, files = [], [], []
    used = 0
    for it in items:
        m = by_key[(it.run_id, it.blocker)]
        lex, jdg = m["expected"], m["class"]
        lex_is_a = _ab_order(it.run_id, it.blocker)
        opt_a, opt_b = (lex, jdg) if lex_is_a else (jdg, lex)
        sig = {c: it.signatures[it.class_names.index(c)]
               for c in (opt_a, opt_b)}
        user = "\n".join([
            f"TASK: {it.task}",
            f"DECISION (blocker id): {it.blocker}",
            "DESCRIPTION: " + (it.description or "(none available)"),
            "TOPIC ANCHORS: " + "; ".join(it.anchors),
            "",
            "THE TWO CANDIDATE INTERPRETATIONS:",
            f"Option A -- {opt_a}",
            "   indicators: " + "; ".join(f'"{s}"' for s in sig[opt_a]),
            f"Option B -- {opt_b}",
            "   indicators: " + "; ".join(f'"{s}"' for s in sig[opt_b]),
            "",
            f"AGENT RUN TRACE ({it.run_id}"
            + (", excerpted" if it.policy == "excerpt" else "") + "):",
            "<<<TRACE", it.excerpt, "TRACE>>>",
        ])
        records.append({"custom_id": it.custom_id, "run": it.run_id,
                        "blocker": it.blocker, "lexicon": lex, "judge": jdg,
                        "lexicon_is_A": lex_is_a,
                        "judge_conf": m.get("confidence")})
        size = len(ADJ_SYSTEM) + len(user)
        if work and (used + size > 120_000 or len(work) >= 6):
            p = out / f"adj_work_{len(files):03d}.json"
            p.write_text(json.dumps({"items": work}, ensure_ascii=False),
                         encoding="utf-8")
            files.append(p)
            work, used = [], 0
        work.append({"custom_id": it.custom_id, "system": ADJ_SYSTEM,
                     "user": user})
        used += size
    if work:
        p = out / f"adj_work_{len(files):03d}.json"
        p.write_text(json.dumps({"items": work}, ensure_ascii=False),
                     encoding="utf-8")
        files.append(p)

    with (out / "adj_key.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"built {len(records)} adjudication items -> {len(files)} work files"
          f" in {out}")
    return 0


def score(out: Path) -> int:
    key = {json.loads(l)["custom_id"]: json.loads(l)
           for l in (out / "adj_key.jsonl").open(encoding="utf-8")}
    verdicts = []
    for f in sorted(out.glob("session_results_*.jsonl")):
        for line in f.open(encoding="utf-8"):
            verdicts.append(json.loads(line))
    tally = Counter()
    rows = []
    for v in verdicts:
        k = key.get(v["custom_id"])
        if k is None:
            continue
        vd = str(v.get("verdict", "")).strip().upper()
        if vd == "A":
            winner = "lexicon" if k["lexicon_is_A"] else "judge"
        elif vd == "B":
            winner = "judge" if k["lexicon_is_A"] else "lexicon"
        else:
            winner = "neither"
        tally[winner] += 1
        rows.append({**k, "winner": winner,
                     "adj_conf": v.get("confidence"),
                     "adj_reasoning": v.get("reasoning")})
    n = sum(tally.values())
    report = {"n_adjudicated": n, "tally": dict(tally),
              "share": {k: round(v / n, 3) for k, v in tally.items()} if n else {},
              "rows": rows}
    Path("results").mkdir(exist_ok=True)
    Path("results/label_disagreement_adjudication.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("n_adjudicated", "tally", "share")}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", default="results/label_judge_validation.json")
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--out", default="models/label_disagreement_adj")
    args = ap.parse_args()
    if args.build:
        return build(Path(args.report), args.a0, args.classes, Path(args.out))
    if args.score:
        return score(Path(args.out))
    print("nothing to do: pass --build or --score")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
