"""DIAGNOSTIC (NOT a pre-registered 028 cell): re-anchor r_vec PER BLOCKER.

Motivation. `wta.divergence.behavior_features` sets `r_vec` to the run's
LATEST mutating action (spec B2, sticky vote). Every blocker in a run whose
commitment resolves to the same action therefore gets a BYTE-IDENTICAL
representation under a DIFFERENT label. Measured 2026-08-30: 705 of the 1,595
labelled commitments (44.2%) share a `commit_char` with another blocker in the
same run; group sizes run up to 5 blockers on one anchor.

That makes the pair-separation statistic ill-posed for nearly half the frozen
pool -- independently of whether any interpretation signal exists. This script
re-anchors per blocker and re-runs the identical statistic.

Anchor rule (blocker_anchor): among the run's mutating actions at or before the
commit round, pick the one whose normalized text scores the most `_hits`
against THAT blocker's own vocabulary (its `anchors` plus every class
`signatures` list). Ties -> earliest. Zero hits anywhere -> fall back to the
published run-level sticky vector, and count it.

Restricting to actions at/before the commit round keeps this apples-to-apples
with the published row, which anchors at that round; it never looks past the
commitment.

Arms, for hashed and MiniLM:
  run_anchor      the published construction -- a reproduction guard
                  (must return .555 / .580)
  blocker_anchor  the re-anchored construction
Each reported with a task-clustered bootstrap CI (2000 draws, seed 0 -- the
same estimator and constants as scripts/t1_auroc_ci.py), plus the aliasing
rate before and after, and the exclusive/aliased subgroup split.

Reported as-run. Writes a fresh results/diag_per_blocker_anchor.json.

    python scripts/diagnose_per_blocker_anchor.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from offline_ask_headtohead import (commit_rounds,  # noqa: E402
                                    load_commitments, load_task_actions)
from t1_auroc_ci import BOOT, SEED, auroc_from  # noqa: E402
from wta.divergence import (R_DIM, _char3grams,  # noqa: E402
                            behavior_features, signed_hash_vec)
from wta.labeling import _hits, _is_mutating, _norm, _norm_map  # noqa: E402
from wta.labeling import load_class_artifact  # noqa: E402

SPAN_HALF = 300  # chars either side of the blocker's signature match


def span_text(raw: str, terms: list[str]) -> str | None:
    """The +/-SPAN_HALF raw-char window centred on where THIS blocker's
    vocabulary matches inside one action. Positions are found in normalized
    text and mapped back through _norm_map, so the window is never built by
    comparing two coordinate systems (the decisions/026 defect)."""
    norm, idx = _norm_map(raw)
    pos = []
    for t in terms:
        tn = _norm(t)
        if not tn:
            continue
        start = norm.find(tn)
        while start != -1:
            pos.append(start)
            start = norm.find(tn, start + 1)
    if not pos:
        return None
    centre = idx[int(np.median(pos))] if idx else 0
    return raw[max(0, centre - SPAN_HALF):centre + SPAN_HALF]


def vec_from_text(text: str, obs: dict, emb) -> np.ndarray:
    """r_vec for an arbitrary action-text substitute, built exactly as
    wta.divergence.behavior_features builds it (same fields, same order)."""
    if emb is not None:
        body = " ".join([str(obs.get("subgoal") or ""), text,
                         str(obs.get("error_signature") or "")])
        return emb(body)
    body = " ".join([_norm(text), " ".join(obs.get("region") or []),
                     str(obs.get("error_signature") or "")])
    return signed_hash_vec(_char3grams(body), R_DIM, seed=0)


def blocker_vocab(spec) -> list[str]:
    """That blocker's own lexicon: its cue anchors plus every class signature."""
    terms = list(spec.get("anchors") or [])
    for c in spec.get("classes") or []:
        terms.extend(c.get("signatures") or [])
    return terms


def clustered_ci(rows, boot=BOOT, seed=SEED):
    """Task-clustered bootstrap, identical estimator/constants to t1_auroc_ci."""
    if not rows:
        return None
    tasks = np.array([r[0] for r in rows])
    labels = np.array([r[2] for r in rows], dtype=int)
    dists = np.array([r[3] for r in rows], dtype=float)
    point = auroc_from(labels, dists)
    uniq = sorted(set(tasks.tolist()))
    idx = {t: np.where(tasks == t)[0] for t in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([idx[uniq[i]] for i in pick])
        a = auroc_from(labels[sel], dists[sel])
        if not np.isnan(a):
            draws.append(a)
    lo, hi = (float(np.percentile(draws, 2.5)),
              float(np.percentile(draws, 97.5))) if draws else (None, None)
    return {"auroc": float(point), "ci95": [lo, hi],
            "n_same": int((labels == 0).sum()), "n_diff": int((labels == 1).sum()),
            "n_tasks": len(uniq), "n_boot_valid": len(draws)}


def pair_rows(vec_by_cell):
    """[(task, blocker, label, distance)] from {(task,blocker): {class: [vecs]}}."""
    rows = []
    for (task, blocker), vecs in vec_by_cell.items():
        classes = sorted(vecs)
        for i, ca in enumerate(classes):
            va = vecs[ca]
            for x in range(len(va)):
                for y in range(x + 1, len(va)):
                    rows.append((task, blocker, 0,
                                 float(np.linalg.norm(va[x][1] - va[y][1]))))
            for cb in classes[i + 1:]:
                for u in va:
                    for v in vecs[cb]:
                        rows.append((task, blocker, 1,
                                     float(np.linalg.norm(u[1] - v[1]))))
    return rows


def aliasing_rate(vec_by_cell):
    """Fraction of (run, blocker) vectors byte-identical to another blocker's
    vector in the SAME run -- the ill-posedness this script targets."""
    by_run = defaultdict(list)
    for (task, blocker), vecs in vec_by_cell.items():
        for cls, items in vecs.items():
            for rid, v in items:
                by_run[(task, rid)].append((blocker, v))
    aliased = total = 0
    for _, items in by_run.items():
        for i, (bi, vi) in enumerate(items):
            total += 1
            if any(bj != bi and np.array_equal(vi, vj)
                   for bj, vj in items):
                aliased += 1
    return {"aliased": aliased, "total": total,
            "rate": (aliased / total) if total else None}


def build(actions, art, committed, emb, mode):
    """mode='run'  -> published sticky latest-mutating-action anchor
       mode='blocker' -> per-blocker signature-matched anchor
       mode='span'    -> that anchor, narrowed to the blocker's own text window"""
    out, fallbacks, moved, considered = {}, 0, 0, 0
    for task, runs in actions.items():
        feats = {rid: behavior_features(a, r_embedder=emb)
                 for rid, a in runs.items()}
        rounds = commit_rounds(runs, committed, art, task)
        ordered = {rid: sorted(a, key=lambda x: x.segment_idx)
                   for rid, a in runs.items()}
        for blocker, spec in art[task].items():
            terms = blocker_vocab(spec) if mode != "run" else []
            vecs = defaultdict(list)
            for rid in runs:
                c = committed.get((rid, blocker))
                k = rounds.get((rid, blocker))
                if c is None or k is None or k >= len(feats[rid]):
                    continue
                run_vec = feats[rid][k].r_vec
                if mode == "run":
                    vecs[c].append((rid, run_vec))
                    continue
                considered += 1
                # mutating actions at or before the commit round
                cand = [a for a in ordered[rid][:k + 1]
                        if _is_mutating(a.action_text or "")]
                best, best_score = None, 0
                for a in cand:
                    s = _hits(_norm(a.action_text or ""), terms)
                    if s > best_score:
                        best, best_score = a, s
                if best is None:
                    fallbacks += 1
                    vecs[c].append((rid, run_vec))
                    continue
                if mode == "span":
                    w = span_text(best.action_text or "", terms)
                    if w is None:
                        fallbacks += 1
                        vecs[c].append((rid, run_vec))
                        continue
                    v = vec_from_text(w, best.observables or {}, emb)
                else:
                    v = behavior_features([best], r_embedder=emb)[0].r_vec
                if not np.array_equal(v, run_vec):
                    moved += 1
                vecs[c].append((rid, v))
            if vecs:
                out[(task, blocker)] = dict(vecs)
    meta = {"n_cells": len(out)}
    if mode != "run":
        meta.update({"considered": considered, "fell_back_no_signature_hit": fallbacks,
                     "fallback_rate": (fallbacks / considered) if considered else None,
                     "anchor_moved_vs_run_anchor": moved,
                     "moved_rate": (moved / considered) if considered else None})
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out", default="results/diag_per_blocker_anchor.json")
    args = ap.parse_args()

    t0 = time.time()
    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels_debug))
    actions = load_task_actions(Path(args.a0), None)
    actions = {t: r for t, r in actions.items() if t in art}
    print(f"tasks {len(actions)}  ({time.time() - t0:.0f}s)", flush=True)

    res = {"note": "DIAGNOSTIC, not a pre-registered 028 cell. Re-anchors "
                   "r_vec per blocker; 44.2% of labelled commitments share a "
                   "commit_char with another blocker in the same run, so under "
                   "the published run-level anchor those blockers carry "
                   "byte-identical vectors under different labels.",
           "anchor_rule": "among mutating actions at/before the commit round, "
                          "argmax _hits against that blocker's anchors + all "
                          "class signatures; ties -> earliest; zero hits -> "
                          "fall back to the published run-level vector",
           "reference_published": {"r3_hashed": 0.555, "r4_minilm": 0.580},
           "arms": {}}

    for rep, make in (("hashed", lambda: None),
                      ("minilm", lambda: __import__(
                          "wta.embed", fromlist=["MiniLMEmbedder"]
                      ).MiniLMEmbedder())):
        emb = make()
        print(f"\n=== {rep} ===", flush=True)
        for mode in ("run", "blocker", "span"):
            cells, meta = build(actions, art, committed, emb, mode)
            rows = pair_rows(cells)
            ci = clustered_ci(rows, boot=args.boot)
            alias = aliasing_rate(cells)
            key = f"{mode}_anchor"
            res["arms"].setdefault(key, {})[rep] = {
                **(ci or {}), "aliasing": alias, "meta": meta}
            print(f"{key:15s} AUROC {ci['auroc']:.4f} "
                  f"CI [{ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]  "
                  f"same {ci['n_same']} diff {ci['n_diff']} "
                  f"tasks {ci['n_tasks']}  "
                  f"aliased {alias['rate']:.3f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            if mode != "run":
                print(f"                moved {meta['moved_rate']:.3f}  "
                      f"fellback {meta['fallback_rate']:.3f}", flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {p}  ({res['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
