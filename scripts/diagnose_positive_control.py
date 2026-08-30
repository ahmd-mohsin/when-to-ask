"""DIAGNOSTIC (NOT a pre-registered 028 cell): does the T1 pair-separation
machinery work at all?

Every T1 row (R3/R4/R5) scores the SAME statistic: Euclidean distance between
two runs' r_vec at their commitment round, ranked by AUROC. Every row lands in
.54-.60. Two worlds are consistent with that:

  (W1) the ruler works, and the interpretation signal genuinely is not there;
  (W2) the ruler is broken, and .54-.60 is what it returns for ANY label.

Nothing in 026/027/028 distinguishes them, because the statistic has never been
run against a target that MUST be recoverable. This script does that, reusing
`feature_signal_gate.auroc` and the same r_vec/Euclidean construction so the
comparison is apples-to-apples.

Four arms:

  A ceiling_task    class = task_id. Runs on different HiL-Bench tasks touch
                    different repos entirely. A working ruler must score ~1.0.
                    Failure here is catastrophic and invalidates all of T1.

  B ceiling_files   within a task, class = the set of files touched at the
                    commit round. An unambiguous behavioural difference, but a
                    much finer one than arm A.

  C planted_2x2     fully synthetic, in the style of
                    harness/contract/test_divergence.py's planted fork. Crosses
                    {same, different} interpretive CHOICE with {same, different}
                    shell IDIOM. Yields the signal-to-nuisance ratio directly:
                    if (same choice, diff idiom) distance exceeds (diff choice,
                    same idiom) distance, idiom variance dominates meaning.

  D label_ladder    the REAL interpretation label, stratified by the lexicon's
                    own decision margin (winner score minus runner-up in
                    labels_debug `scores`). If AUROC does not rise with label
                    confidence, label noise is not what is holding T1 down.

Reported as-run. Writes a fresh results/diag_positive_control.json.

    python scripts/diagnose_positive_control.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from feature_signal_gate import auroc  # noqa: E402
from offline_ask_headtohead import (commit_rounds,  # noqa: E402
                                    load_commitments, load_task_actions)
from wta.divergence import behavior_features  # noqa: E402
from wta.labeling import _is_mutating, load_class_artifact  # noqa: E402

MAX_DIFF_PAIRS = 200_000  # deterministic subsample cap for the cross-task arm


def _dist(u, v):
    return float(np.linalg.norm(u - v))


def _pairs_by_class(vecs_by_class, rng=None, cap=None):
    """same[] / diff[] Euclidean distances, exactly as pair_distances does."""
    same, diff = [], []
    classes = sorted(vecs_by_class)
    for i, ca in enumerate(classes):
        va = vecs_by_class[ca]
        for x in range(len(va)):
            for y in range(x + 1, len(va)):
                same.append(_dist(va[x], va[y]))
        for cb in classes[i + 1:]:
            for u in va:
                for v in vecs_by_class[cb]:
                    diff.append(_dist(u, v))
    if cap is not None and len(diff) > cap and rng is not None:
        idx = rng.choice(len(diff), size=cap, replace=False)
        diff = [diff[i] for i in idx]
    return np.array(same), np.array(diff)


def _summary(same, diff):
    return {"auroc": auroc(same, diff) if len(same) and len(diff) else None,
            "n_same": int(len(same)), "n_diff": int(len(diff)),
            "same_mean": float(same.mean()) if len(same) else None,
            "diff_mean": float(diff.mean()) if len(diff) else None}


# --------------------------------------------------------------------------
# Arm C: synthetic planted forks. Two interpretive resolutions of one blocker
# ("what should the function do on an unknown platform?"), each written four
# mechanically different ways -- the idioms actually observed in data/a0_v3_32b.
# --------------------------------------------------------------------------
CHOICE_A = ("return None", "        return None")
CHOICE_B = ("raise ValueError", "        raise ValueError('unsupported platform')")

SUBGOAL = "handle the unknown-platform branch in get_distribution"


def _synth(choice_body: str, idiom: int) -> str:
    """One mutating action expressing `choice_body` in idiom #idiom."""
    f = "lib/ansible/module_utils/common/sys_info.py"
    if idiom == 0:
        return f"sed -i 's|# UNKNOWN|{choice_body.strip()}|' {f}"
    if idiom == 1:
        return (f"cat << EOF > {f}\ndef get_distribution():\n"
                f"    system = platform.system()\n"
                f"    if system not in SUPPORTED:\n{choice_body}\n"
                f"    return system\nEOF")
    if idiom == 2:
        # NB: the obvious fourth idiom, `python -c "...open(f,'w').write(s)"`,
        # is NOT seen as mutating by wta.labeling._is_mutating, whose token
        # list is ("sed -i", ">", ">>", "tee ", "patch ", "git apply",
        # "perl -i"). Python-mediated writes are invisible to the labeler --
        # recorded as `labeler_blind_spot` below, and avoided here so all four
        # arms of the 2x2 are actually populated.
        return (f"printf '%s\\n' \"{choice_body.strip()}\" | tee -a {f}")
    return (f"find . -name \"sys_info.py\" -exec sed -i "
            f"'/SUPPORTED/r /dev/stdin' {{}} <<'EOF'\n{choice_body}\nEOF")


def _synth_vec(choice_body, idiom, embedder):
    from wta.logging_schema import ActionEvent
    obs = {"files": ["lib/ansible/module_utils/common/sys_info.py"],
           "subgoal": SUBGOAL, "region": [], "error_signature": "exit 0"}
    a = ActionEvent(token_idx=5, action_text=_synth(choice_body, idiom),
                    observables=dict(obs), segment_idx=0)
    assert _is_mutating(a.action_text), f"idiom {idiom} not seen as mutating"
    return behavior_features([a], r_embedder=embedder)[0].r_vec


def arm_planted(embedder):
    idioms = range(4)
    vec = {(c, i): _synth_vec(body, i, embedder)
           for c, (_, body) in (("A", CHOICE_A), ("B", CHOICE_B))
           for i in idioms}
    cells = defaultdict(list)
    for (ca, ia), (cb, ib) in itertools.combinations(vec, 2):
        key = (("same_choice" if ca == cb else "diff_choice"),
               ("same_idiom" if ia == ib else "diff_idiom"))
        cells[key].append(_dist(vec[(ca, ia)], vec[(cb, ib)]))
    out = {f"{a}__{b}": {"mean": float(np.mean(v)), "n": len(v)}
           for (a, b), v in sorted(cells.items())}
    signal = out["diff_choice__same_idiom"]["mean"]
    nuisance = out["same_choice__diff_idiom"]["mean"]
    out["signal_to_nuisance"] = (float(signal / nuisance) if nuisance > 0
                                 else None)
    # AUROC for "different choice", idiom held constant vs idiom varying
    same_i = {c: [vec[(c, i)] for i in idioms] for c in ("A", "B")}
    s, d = _pairs_by_class(same_i)
    out["auroc_choice_idiom_varying"] = auroc(s, d)
    for i in idioms:
        s2, d2 = _pairs_by_class({c: [vec[(c, i)]] for c in ("A", "B")})
        out[f"dist_choice_within_idiom_{i}"] = float(d2[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--out", default="results/diag_positive_control.json")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(0)
    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels_debug))
    actions = load_task_actions(Path(args.a0), None)
    actions = {t: r for t, r in actions.items() if t in art}
    print(f"tasks {len(actions)}  ({time.time() - t0:.0f}s)", flush=True)

    results = {"note": "DIAGNOSTIC, not a pre-registered 028 cell. "
                       "Validates the T1 pair-separation machinery against "
                       "targets that must be recoverable.",
               "labeler_blind_spot": {
                   "finding": "wta.labeling._is_mutating matches only "
                              "('sed -i', '>', '>>', 'tee ', 'patch ', "
                              "'git apply', 'perl -i'). A Python-mediated "
                              "write such as "
                              "python -c \"...open(f,'w').write(s)\" mutates "
                              "the repo but is NOT counted as a mutating "
                              "action, so it can neither anchor a commitment "
                              "nor contribute to r_vec.",
                   "incidental": True},
               "arms": {}}

    for rep_name, make in (("hashed", lambda: None),
                           ("minilm", lambda: __import__(
                               "wta.embed", fromlist=["MiniLMEmbedder"]
                           ).MiniLMEmbedder())):
        emb = make()
        print(f"\n=== representation: {rep_name} ===", flush=True)
        feats = {t: {r: behavior_features(a, r_embedder=emb)
                     for r, a in runs.items()} for t, runs in actions.items()}

        # ---- last-mutating-action vector per run (sticky r_vec) ----
        last = {}
        for task, runs in feats.items():
            for rid, fs in runs.items():
                if fs and fs[-1].weight == 1.0:
                    last[(task, rid)] = fs[-1].r_vec

        # ---- ARM A: ceiling, same-vs-different TASK ----
        by_task = defaultdict(list)
        for (task, _), v in last.items():
            by_task[task].append(v)
        s, d = _pairs_by_class(by_task, rng=rng, cap=MAX_DIFF_PAIRS)
        a_res = _summary(s, d)
        a_res["n_runs"] = len(last)
        a_res["n_tasks"] = len(by_task)
        results["arms"].setdefault("A_ceiling_task", {})[rep_name] = a_res
        print(f"A ceiling_task    AUROC {a_res['auroc']:.3f}  "
              f"(same {a_res['n_same']}, diff {a_res['n_diff']})", flush=True)

        # ---- ARM B: within task, same-vs-different FILE SET ----
        s_all, d_all, n_cells = [], [], 0
        for task, runs in actions.items():
            groups = defaultdict(list)
            for rid, acts in runs.items():
                mut = [a for a in sorted(acts, key=lambda x: x.segment_idx)
                       if _is_mutating(a.action_text or "")]
                if not mut:
                    continue
                files = frozenset(
                    f for a in mut for f in ((a.observables or {}).get("files") or []))
                if files and (task, rid) in last:
                    groups[files].append(last[(task, rid)])
            if len(groups) >= 2:
                s2, d2 = _pairs_by_class(groups)
                if len(s2) and len(d2):
                    s_all.append(s2)
                    d_all.append(d2)
                    n_cells += 1
        s, d = np.concatenate(s_all), np.concatenate(d_all)
        b_res = _summary(s, d)
        b_res["n_tasks_contributing"] = n_cells
        results["arms"].setdefault("B_ceiling_files", {})[rep_name] = b_res
        print(f"B ceiling_files   AUROC {b_res['auroc']:.3f}  "
              f"(same {b_res['n_same']}, diff {b_res['n_diff']}, "
              f"{n_cells} tasks)", flush=True)

        # ---- ARM C: planted synthetic 2x2 ----
        c_res = arm_planted(emb)
        results["arms"].setdefault("C_planted_2x2", {})[rep_name] = c_res
        print(f"C planted  diff-choice/same-idiom {c_res['diff_choice__same_idiom']['mean']:.4f}"
              f"  same-choice/diff-idiom {c_res['same_choice__diff_idiom']['mean']:.4f}"
              f"  ratio {c_res['signal_to_nuisance']:.3f}", flush=True)
        print(f"           AUROC(choice | idiom varying) "
              f"{c_res['auroc_choice_idiom_varying']:.3f}", flush=True)

        # ---- ARM D: real label, stratified by lexicon decision margin ----
        margins = {}
        for line in open(args.labels_debug, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "commitment" or r.get("chosen") is None:
                continue
            sc = sorted((r.get("scores") or {}).values(), reverse=True)
            if sc:
                margins[(r["run"], r["blocker"])] = sc[0] - (sc[1] if len(sc) > 1 else 0)

        buckets = defaultdict(lambda: ([], []))
        for task, runs in actions.items():
            rounds = commit_rounds(runs, committed, art, task)
            for blocker in art[task]:
                vecs, mg = defaultdict(list), []
                for rid in runs:
                    c = committed.get((rid, blocker))
                    k = rounds.get((rid, blocker))
                    if c is not None and k is not None and k < len(feats[task][rid]):
                        vecs[c].append(feats[task][rid][k].r_vec)
                        mg.append(margins.get((rid, blocker), 0))
                if len(vecs) < 1 or not mg:
                    continue
                med = float(np.median(mg))
                lbl = ("margin_hi" if med >= 5 else
                       "margin_mid" if med >= 2 else "margin_lo")
                s2, d2 = _pairs_by_class(vecs)
                buckets[lbl][0].append(s2)
                buckets[lbl][1].append(d2)

        d_res = {}
        for lbl, (ss, dd) in sorted(buckets.items()):
            s3 = np.concatenate([x for x in ss if len(x)]) if any(len(x) for x in ss) else np.array([])
            d3 = np.concatenate([x for x in dd if len(x)]) if any(len(x) for x in dd) else np.array([])
            d_res[lbl] = _summary(s3, d3)
            a = d_res[lbl]["auroc"]
            print(f"D {lbl:11s}   AUROC "
                  f"{'n/a' if a is None else f'{a:.3f}'}  "
                  f"(same {d_res[lbl]['n_same']}, diff {d_res[lbl]['n_diff']})",
                  flush=True)
        results["arms"].setdefault("D_label_margin_ladder", {})[rep_name] = d_res

    results["elapsed_s"] = round(time.time() - t0, 1)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {p}  ({results['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
