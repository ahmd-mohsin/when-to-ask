"""DIAGNOSTIC: what actually limits the replay pilot's arm A.

Third pass over the replay-and-diff pilot, re-scoring the diffs it already
cached in --work-dir. Nothing is re-replayed and no pilot number is restated:
results/replay_diff_pilot.json stands as-run at arm A = 0.581, and
results/diag_replay_empty_diffs.json explains 0.581 -> 0.817 (26 no-op runs
scored as exact matches by jaccard_distance(empty, empty) = 0.0).

0.817 is still under the pilot's 0.95 ceiling bar. This asks what is holding it
there. Same-task pairs sit at mean distance 0.880 -- two runs on the SAME repo
share almost no changed lines -- and that is what caps the separation.

Three candidate explanations, tested in order:

  1. THE METRIC punishes size asymmetry. Diffs run 1..1797 lines (median 110),
     and same-task pairs differ in size by a median 2.9x (p90 22x). Jaccard is
     harsh on that: A subset B with |A|=5, |B|=200 scores distance 0.975.
     -> RULED OUT. Containment (1 - |A&B|/min) and Dice move arm A by <0.006
        (0.8170 / 0.8220 / 0.8170). Size asymmetry is real and irrelevant.

  2. NOISE inflates each diff. -> RULED OUT by inspection. The per-task shared
     core is syntactic boilerplate (`+try:`, `+else:`, `+import json`) plus
     DELETIONS -- runs editing the same region delete the same original lines.
     The run-unique remainder is real solution content. No line is common to
     ALL runs of any task (in_ALL = 0 for all five), and 37-83% of a task's
     distinct lines are seen exactly once.

  3. THE POOL IS A MIXTURE of two populations. -> THIS ONE.
     47 of 89 non-empty runs contain NO deletion lines at all: they only add
     new content (the `mkdir -p && touch` shape from the previous diagnostic,
     grown up -- new files, appended blocks) and never modify existing code.
     The other 42 modify code. Split them and arm A separates:

         modifying (has deletions)   n=42   arm A 0.967
         additive-only               n=47   arm A 0.821

     The channel is NOT the story -- on those same 42 runs, deletions-only
     scores 0.880 and additions-only 0.937, both BELOW the full 0.967. It is
     which runs, not which lines.

WHAT THIS DOES NOT ESTABLISH. The 0.967 clears the 0.95 bar as a point
estimate and does not survive its own uncertainty: task-clustered CI95
[0.782, 0.994] against additive-only [0.656, 0.972]. The intervals overlap
across nearly their whole range, so this pilot CANNOT distinguish the two
populations. With 5 tasks the cluster bootstrap has 5 independent units and
the interval is correspondingly wide. Treat "modifying runs are separable" as
a hypothesis worth pre-registering, NOT as a result -- and note the subgroup
is chosen post hoc on a property of the outcome, which is exactly the move
that manufactures findings if it is not pre-registered before the next look.

The cheap decisive move is more TASKS, not more runs: clusters, not pairs, set
the width. 5 tasks cost 16.9 GB and 22 min, so ~20 tasks is ~68 GB and ~90 min
-- far short of the 174 GB / 60-task commitment, and enough to tell 0.967 from
0.821 if the split is real.

    python scripts/diagnose_replay_dispersion.py --work-dir /ssd3/wta-replay
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from feature_signal_gate import auroc  # noqa: E402
from replay_diff_pilot import normalize_diff  # noqa: E402

BOOT = 2000  # matches scripts/t1_auroc_ci.py
SEED = 0


def jaccard(a, b):
    return 1 - len(a & b) / len(a | b)


def containment(a, b):
    return 1 - len(a & b) / min(len(a), len(b))


def dice(a, b):
    return 1 - 2 * len(a & b) / (len(a) + len(b))


def arm_a(keys, norm, task_of, distfn=jaccard):
    same, diff = [], []
    ks = sorted(keys)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            d = distfn(norm[a], norm[b])
            (same if task_of[a] == task_of[b] else diff).append(d)
    s, d = np.array(same), np.array(diff)
    if not len(s) or not len(d):
        return None
    return {"auroc": auroc(s, d), "n_same": len(s), "n_cross": len(d),
            "same_mean": float(s.mean()), "cross_mean": float(d.mean()),
            "same_disjoint_rate": float((s == 1.0).mean())}


def clustered_ci(keys, norm, task_of, boot=BOOT, seed=SEED):
    """Task-clustered bootstrap: resample TASKS with replacement (t1_auroc_ci)."""
    rng = np.random.default_rng(seed)
    by_task = {}
    for k in keys:
        by_task.setdefault(task_of[k], []).append(k)
    tasks = sorted(by_task)
    out = []
    for _ in range(boot):
        drawn = [tasks[i] for i in rng.choice(len(tasks), len(tasks), replace=True)]
        same, cross = [], []
        for ii, t1 in enumerate(drawn):
            ks = by_task[t1]
            for x in range(len(ks)):
                for y in range(x + 1, len(ks)):
                    same.append(jaccard(norm[ks[x]], norm[ks[y]]))
            for t2 in drawn[ii + 1:]:
                for a in by_task[t1]:
                    for b in by_task[t2]:
                        if a == b:
                            continue
                        (same if t1 == t2 else cross).append(
                            jaccard(norm[a], norm[b]))
        if same and cross:
            out.append(auroc(np.array(same), np.array(cross)))
    a = np.array(out)
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], len(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="/ssd3/wta-replay")
    ap.add_argument("--out", default="results/diag_replay_dispersion.json")
    args = ap.parse_args()

    norm, task_of = {}, {}
    for f in sorted(Path(args.work_dir).glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if "diff" not in rec:
            continue
        v = normalize_diff(rec.get("diff") or "")
        if v:  # empty diffs are the previous diagnostic's subject
            norm[f.stem] = v
            task_of[f.stem] = f.stem.rsplit("-s", 1)[0]

    sizes = sorted(len(v) for v in norm.values())
    only = {"add": lambda v: frozenset(x for x in v if x[0] == "+"),
            "del": lambda v: frozenset(x for x in v if x[0] == "-")}

    # 1. metric
    metrics = {name: arm_a(list(norm), norm, task_of, fn)["auroc"]
               for name, fn in [("jaccard", jaccard),
                                ("containment", containment), ("dice", dice)]}

    # 2. structure
    structure = {}
    for t in sorted(set(task_of.values())):
        rs = [r for r in norm if task_of[r] == t]
        cnt = Counter()
        for r in rs:
            cnt.update(norm[r])
        structure[t] = {
            "n_runs": len(rs), "distinct_lines": len(cnt),
            "in_all_runs": sum(1 for c in cnt.values() if c == len(rs)),
            "seen_once": sum(1 for c in cnt.values() if c == 1),
            "seen_once_frac": round(
                sum(1 for c in cnt.values() if c == 1) / len(cnt), 3)}

    # 3. mixture
    modifying = [k for k in norm if only["del"](norm[k])]
    additive = [k for k in norm if not only["del"](norm[k])]
    groups = {}
    for name, ks in [("modifying_has_deletions", modifying),
                     ("additive_only", additive),
                     ("all_non_empty", list(norm))]:
        r = arm_a(ks, norm, task_of)
        ci, nb = clustered_ci(ks, norm, task_of)
        r["n_runs"] = len(ks)
        r["clustered_ci95"] = ci
        r["boot_draws_used"] = nb
        r["per_task_n"] = {t: sum(1 for k in ks if task_of[k] == t)
                           for t in sorted(set(task_of.values()))}
        groups[name] = r

    # channel control, on the modifying subgroup only
    channel = {}
    for name, fn in [("both", lambda v: v), ("add_only", only["add"]),
                     ("del_only", only["del"])]:
        proj = {k: fn(norm[k]) for k in modifying}
        keys = [k for k in modifying if proj[k]]
        channel[name] = arm_a(keys, proj, task_of)

    res = {
        "note": "DIAGNOSTIC on the replay-and-diff PILOT. Re-scores cached "
                "diffs; results/replay_diff_pilot.json stands as-run.",
        "question": "Arm A is 0.817 once no-op runs are excluded, still under "
                    "the 0.95 bar. Same-task pairs sit at mean distance 0.880. "
                    "What holds them apart?",
        "n_runs_non_empty": len(norm),
        "diff_size": {"min": sizes[0], "p25": sizes[len(sizes) // 4],
                      "median": st.median(sizes),
                      "p75": sizes[3 * len(sizes) // 4], "max": sizes[-1]},
        "h1_metric": {
            "arm_a_by_distance": metrics,
            "verdict": "RULED OUT: jaccard/containment/dice agree within "
                       "0.006, so size asymmetry is not what caps arm A."},
        "h2_noise": {
            "per_task": structure,
            "verdict": "RULED OUT: no line is common to all runs of any task "
                       "(in_all_runs = 0 everywhere); the shared core is "
                       "syntactic boilerplate plus deletions of the same "
                       "original lines, and the unique remainder is real "
                       "solution content."},
        "h3_mixture": {
            "groups": groups,
            "channel_control_on_modifying_subgroup": channel,
            "verdict": "THIS ONE, as a hypothesis only. 47 of 89 runs never "
                       "modify existing code (no deletion lines). Splitting "
                       "gives arm A 0.967 (modifying) vs 0.821 (additive- "
                       "only). The channel is not the story: on those same 42 "
                       "runs del-only scores 0.880 and add-only 0.937, both "
                       "below the full 0.967."},
        "honest_reading": (
            "The 0.967 does NOT convert the pilot into a GO. Its task-"
            "clustered CI95 is {} against additive-only {} -- overlapping "
            "across nearly their whole range, so 5 tasks cannot distinguish "
            "the two populations. The subgroup is also chosen post hoc on a "
            "property of the outcome. This is a hypothesis to pre-register, "
            "not a result."
        ).format(groups["modifying_has_deletions"]["clustered_ci95"],
                 groups["additive_only"]["clustered_ci95"]),
        "recommended_next_step": (
            "Widen the pilot in TASKS, not runs: the cluster bootstrap has one "
            "independent unit per task, so 5 tasks is what makes the interval "
            "wide. 5 tasks cost 16.9 GB / 22 min, so ~20 tasks is ~68 GB / "
            "~90 min -- far short of the 174 GB, 60-task commitment, and "
            "enough to separate 0.967 from 0.821 if the split is real. "
            "Pre-register the modifying/additive-only split BEFORE that run."),
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"H1 metric   : {metrics}")
    print(f"H2 in_all_runs per task: "
          f"{ {t: s['in_all_runs'] for t, s in structure.items()} }")
    for k, g in groups.items():
        print(f"H3 {k:26} n={g['n_runs']:3} armA={g['auroc']:.4f} "
              f"CI95 {[round(x, 3) for x in g['clustered_ci95']]}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
