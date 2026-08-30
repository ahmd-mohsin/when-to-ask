"""INTERIM DIAGNOSTIC (028 Amendment G phase 1, partial): census + T1
separation on the judge labels produced SO FAR.

Phase 1 is incomplete -- the Fable session limit stopped it partway through
the 3,361-item pool -- and this script reads whatever session_results_*.jsonl
files exist. It is NOT the pre-registered freeze; it exists to answer "are we
on the right track" before spending further usage. Reported as-run, with the
partial-coverage caveat attached to every number.

Two protocol choices are made HERE, for the first time, because Amendment G
did not specify them. Both are recorded in 028 as G.1 and are frozen by this
script's behaviour:

(1) WINDOW ANCHOR FOR JUDGE LABELS. The lexicon arm anchors the r_vec at the
    first mutating action carrying a signature of the committed class
    (commit_rounds). Judge-labelled items have no such action BY
    CONSTRUCTION -- the absence of a signature hit is exactly why the lexicon
    abstained -- so commit_rounds returns None for all of them and the T1
    machinery would silently drop every judge label. Instead the judge
    window is anchored at the judge's own verbatim EVIDENCE SPAN: commit_char
    -> containing segment -> the last action at or before that segment.
    behavior_features' r_vec at that turn is "the latest mutating action's
    vector", the same quantity the lexicon arm reads.
(2) TWO ARMS ARE SCORED SEPARATELY. `judge_only` uses judge labels and judge
    anchors on both sides of every pair. `union` adds lexicon-labelled runs
    (lexicon anchors) to the same cells, which is the coverage gain the whole
    judge arm was for -- but it mixes two anchor methods inside one pair, so
    it is reported as a separate, weaker-evidence arm and never as the
    headline.

    python scripts/interim_judge_arm.py --out results/interim_judge_arm.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from feature_signal_gate import auroc, behavior_features  # noqa: E402
from offline_ask_headtohead import (commit_rounds, load_commitments,  # noqa: E402
                                    load_task_actions)
from wta.judge_labels import JudgeItem, freeze_results, session_responses  # noqa: E402
from wta.labeling import load_class_artifact  # noqa: E402

N_BOOT = 2000
SEED = 0
_SEG_SEP = "\n\n"


def load_partial_results(jdir: Path) -> list[dict]:
    recs = []
    for f in sorted(glob.glob(str(jdir / "session_results_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def segment_offsets(a0: Path, task: str, run_id: str) -> list[int] | None:
    """Char start of each segment in the raw join, or None if no sidecar."""
    sf = a0 / task / f"{run_id}.segments.json"
    if not sf.exists():
        return None
    segs = json.loads(sf.read_text(encoding="utf-8"))
    offs, pos = [], 0
    for s in segs:
        offs.append(pos)
        pos += len(s) + len(_SEG_SEP)
    return offs


def turn_of_char(offs: list[int], acts, char: int) -> int | None:
    """Protocol choice G.1(1): commit_char -> containing segment -> index (in
    segment-ordered action list) of the LAST action at or before it."""
    seg = 0
    for i, o in enumerate(offs):
        if o <= char:
            seg = i
        else:
            break
    k = None
    for i, a in enumerate(acts):           # acts already segment-ordered
        if a.segment_idx <= seg:
            k = i
        else:
            break
    return k


def score(rows):
    """rows = [(task, dist, is_diff)] -> AUROC + task-clustered bootstrap CI."""
    same = np.array([d for _, d, x in rows if not x])
    diff = np.array([d for _, d, x in rows if x])
    if not len(same) or not len(diff):
        return {"auroc": None, "n_same": int(len(same)), "n_diff": int(len(diff))}
    a = auroc(same, diff)
    by_task = defaultdict(list)
    for t, d, x in rows:
        by_task[t].append((d, x))
    tasks = sorted(by_task)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        s, f = [], []
        for k in rng.choice(len(tasks), size=len(tasks), replace=True):
            for d, x in by_task[tasks[k]]:
                (f if x else s).append(d)
        if s and f:
            boots.append(auroc(np.array(s), np.array(f)))
    ci = ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
          if boots else None)
    return {"auroc": float(a), "ci95": ci, "n_same": int(len(same)),
            "n_diff": int(len(diff)), "n_tasks": len(tasks),
            "n_boot_valid": len(boots)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--judge-dir", default="models/v3_32b_judge")
    ap.add_argument("--out", default="results/interim_judge_arm.json")
    args = ap.parse_args()

    jdir, a0 = Path(args.judge_dir), Path(args.a0)
    art = load_class_artifact(args.classes)
    lex = load_commitments(Path(args.labels_debug))

    items = [JudgeItem(**json.loads(l))
             for l in (jdir / "items.jsonl").open(encoding="utf-8")]
    recs = load_partial_results(jdir)
    judged_ids = {r["custom_id"] for r in recs}
    items_judged = [it for it in items if it.custom_id in judged_ids]
    print(f"pool {len(items)} items; judged so far {len(items_judged)} "
          f"({len(items_judged) / len(items):.1%})")

    # the REAL acceptance gate: schema-valid class + verbatim evidence located
    frozen = freeze_results(items_judged, session_responses(recs), a0)
    status = Counter(r["status"] for r in frozen)
    print("status:", dict(sorted(status.items())))

    judge = {(r["run"], r["blocker"]): (r["class"], r["commit_char"])
             for r in frozen if r["status"] == "accepted"}
    print(f"accepted judge labels: {len(judge)}")

    actions = load_task_actions(a0, None)
    actions = {t: r for t, r in actions.items() if t in art}

    # ---- census -----------------------------------------------------------
    cells_j, cells_lex = defaultdict(set), defaultdict(set)
    for (rid, blk), (cls, _) in judge.items():
        cells_j[(rid.split("-s")[0], blk)].add(cls)
    for (rid, blk), cls in lex.items():
        cells_lex[(rid.split("-s")[0], blk)].add(cls)
    runs_per_cell_j = Counter()
    for (rid, blk) in judge:
        runs_per_cell_j[(rid.split("-s")[0], blk)] += 1
    multi_j = {c for c, n in runs_per_cell_j.items() if n >= 2}
    forked_j = {c for c in multi_j if len(cells_j[c]) > 1}
    census = {
        "items_judged": len(items_judged), "items_in_pool": len(items),
        "status_counts": dict(sorted(status.items())),
        "accepted_labels": len(judge),
        "abstention_rate_of_judged": round(
            status.get("abstained", 0) / max(1, len(items_judged)), 4),
        "cells_with_ge2_judge_runs": len(multi_j),
        "forked_cells_judge": len(forked_j),
        "forked_fraction_judge": (round(len(forked_j) / len(multi_j), 4)
                                  if multi_j else None),
    }
    print("census:", json.dumps(census, indent=1))

    # ---- pair pools -------------------------------------------------------
    # judge side: evidence-anchored turns (protocol choice G.1(1))
    j_turn = {}
    seg_cache = {}
    for (rid, blk), (cls, cchar) in judge.items():
        task = rid.split("-s")[0]
        acts = actions.get(task, {}).get(rid)
        if not acts:
            continue
        if rid not in seg_cache:
            seg_cache[rid] = segment_offsets(a0, task, rid)
        offs = seg_cache[rid]
        if not offs:
            continue
        k = turn_of_char(offs, acts, cchar)
        if k is not None:
            j_turn[(rid, blk)] = k

    lex_rounds = {}
    for task, runs in actions.items():
        for (rid, blk), k in commit_rounds(runs, lex, art, task).items():
            if k is not None:
                lex_rounds[(rid, blk)] = k

    def build(r_embedder, use_lex):
        """-> [(task, dist, is_diff)] over within-cell pairs."""
        cache = {}

        def feats(task, rid):
            if rid not in cache:
                cache[rid] = behavior_features(actions[task][rid],
                                               r_embedder=r_embedder)
            return cache[rid]

        cells = defaultdict(list)   # (task, blocker) -> [(rid, cls, vec)]
        for (rid, blk), k in j_turn.items():
            task = rid.split("-s")[0]
            f = feats(task, rid)
            if k < len(f) and f[k].weight > 0:      # a mutating action exists
                cells[(task, blk)].append((rid, judge[(rid, blk)][0], f[k].r_vec))
        if use_lex:
            for (rid, blk), k in lex_rounds.items():
                task = rid.split("-s")[0]
                if rid not in actions.get(task, {}):
                    continue
                f = feats(task, rid)
                if k < len(f) and f[k].weight > 0:
                    cells[(task, blk)].append((rid, lex[(rid, blk)], f[k].r_vec))
        rows = []
        for (task, blk), v in cells.items():
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    rows.append((task,
                                 float(np.linalg.norm(v[i][2] - v[j][2])),
                                 v[i][1] != v[j][1]))
        return rows

    def make_minilm():
        from wta.embed import MiniLMEmbedder
        return MiniLMEmbedder()

    out = {"INTERIM": True, "PARTIAL_PHASE1": True,
           "question": ("census + T1 separation under the Fable judge labels "
                        "produced so far (phase 1 incomplete)"),
           "protocol_choices_made_here": {
               "judge_window_anchor": ("evidence span -> segment -> last "
                                       "action at/before it (028 G.1(1)); "
                                       "commit_rounds is undefined for judge "
                                       "items by construction"),
               "arms": "judge_only (headline) and union (mixed anchors, weaker)"},
           "census": census}
    for name, mk in (("hashed_r3", lambda: None), ("minilm_r4", make_minilm)):
        emb = mk()
        arm = {}
        for arm_name, use_lex in (("judge_only", False), ("union", True)):
            arm[arm_name] = score(build(emb, use_lex))
            s = arm[arm_name]
            au = None if s["auroc"] is None else round(s["auroc"], 3)
            print(f"{name:10s} {arm_name:10s}: AUROC {au} "
                  f"CI {s.get('ci95')} same/diff {s['n_same']}/{s['n_diff']} "
                  f"tasks {s.get('n_tasks')}")
        out[name] = arm
    out["reference_lexicon_T1_rows"] = {
        "r3_hashed": 0.555, "r4_minilm": 0.580, "r5_bge": 0.573,
        "r6b_llm_judge": 0.5786}
    out["caveat"] = (
        "PARTIAL: phase 1 stopped at the Fable usage limit, so this is a "
        "prefix of the pool in work-file order, not a random sample. Numbers "
        "are interim and are superseded by the frozen full-pool cell.")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
