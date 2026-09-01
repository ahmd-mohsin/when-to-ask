"""PILOT (NOT a pre-registered 028 cell): can replay-and-diff be an instrument?

The T1 pair-separation statistic fails its positive control
(results/diag_positive_control.json): it scores .788/.849 on "are these two
runs even on different repositories", .685 on "did they edit different files",
and .589/.474 on a fork planted by construction. Label noise is ruled out (the
judge arm, frozen at 99.0%, moves it by <.03) and anchor ill-posedness is ruled
out (results/diag_per_blocker_anchor.json: de-aliasing .499 -> .064 LOWERS the
number). What remains is the representation itself: one shell command, embedded.

Replay-and-diff replaces it. Two runs decided the same thing iff, after
replaying their recorded commands in the task image, their final normalized
`git diff` matches. That removes shell idiom, the anchor, the lexicon and the
judge -- all four by construction.

The full version is expensive (60 tasks, 174 GB of images, ~38.5k execs). This
PILOT runs a few tasks first and answers two questions, in order:

  Q1 REPLAY FIDELITY -- the gate. Does replaying `action_text` in
     `segment_idx` order actually reproduce the run? Every ActionEvent
     recorded an `error_signature` of the form "exit N"; replay must
     reproduce those exit codes. If fidelity is low the diffs describe a run
     that never happened and the whole approach is void, however good its
     separation looks. NO-GO here means stop, and do not spend the box time.

  Q2 POSITIVE CONTROL -- the same three arms diag_positive_control.py used, so
     the answer is directly comparable:
       A cross-task    runs on different repos must be ~1.0
       B file-set      runs touching different files, within a task
       C interpretation the real lexicon label (the .555/.580 cell)
     Scored two ways: exact match of the normalized diff (the "iff"
     formulation) and 1 - Jaccard over normalized changed-line sets (a graded
     distance, so AUROC is comparable to the published T1 rows).

Sealed-pool safety: the task list is derived by
restore_hilbench_images.eligible_tasks, which is verbatim the walk in
collect_v2.main(). swe_60+ is excluded by construction, not by a filter that
could be mis-set.

Resumable: per-run diffs are cached under --work-dir and reused.

    python scripts/replay_diff_pilot.py --n-tasks 5
    python scripts/replay_diff_pilot.py --n-tasks 5 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from feature_signal_gate import auroc  # noqa: E402
from offline_ask_headtohead import (load_commitments,  # noqa: E402
                                    load_task_actions)
from restore_hilbench_images import eligible_tasks  # noqa: E402
from wta.agent_env import DockerTaskEnv  # noqa: E402
from wta.labeling import load_class_artifact  # noqa: E402

_DIFF_NOISE = re.compile(r"^(diff --git |index |--- |\+\+\+ |@@ |similarity |"
                         r"rename |new file mode |deleted file mode |"
                         r"old mode |new mode |Binary files )")
_EXIT = re.compile(r"exit\s+(-?\d+)")


def normalize_diff(diff: str) -> frozenset[str]:
    """The set of normalized changed lines -- idiom-invariant by construction.

    Positional noise (hunk headers, blob hashes, file modes) is dropped, and
    what survives is WHAT changed, not where or in what order. Two runs that
    wrote the same code via `sed -i`, a heredoc and `tee` collapse to the same
    set; two runs that wrote different code do not.
    """
    out = set()
    for line in diff.splitlines():
        if _DIFF_NOISE.match(line):
            continue
        if not line or line[0] not in "+-":
            continue
        body = line[1:].strip()
        if body:
            out.add(f"{line[0]}{' '.join(body.split())}")
    return frozenset(out)


def jaccard_distance(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def recorded_exit(action) -> int | None:
    m = _EXIT.search(str((action.observables or {}).get("error_signature") or ""))
    return int(m.group(1)) if m else None


def replay_run(image: str, run_id: str, actions, exec_timeout: int) -> dict:
    """Replay one run's commands in a fresh container; return its final diff
    plus the per-action exit-code agreement that gates the whole approach."""
    ordered = sorted(actions, key=lambda a: a.segment_idx)
    matched = compared = 0
    exits = []
    with DockerTaskEnv(image, name=f"wta-replay-{run_id}",
                       exec_timeout=exec_timeout) as env:
        code, _ = env.execute("git rev-parse --is-inside-work-tree")
        is_git = code == 0
        if is_git:
            env.execute("git config --global --add safe.directory /app; "
                        "git add -A; git stash list >/dev/null 2>&1 || true")
        for a in ordered:
            code, _ = env.execute(a.action_text or "")
            exits.append(code)
            rec = recorded_exit(a)
            if rec is not None:
                compared += 1
                matched += int(rec == code)
        if is_git:
            _, diff = env.execute(
                "git -c core.fileMode=false add -A >/dev/null 2>&1; "
                "git -c core.fileMode=false diff --cached --no-color")
            mode = "git"
        else:
            _, diff = env.execute(
                "find . -type f -newer /etc/hostname -not -path './.git/*' "
                "| sort | xargs -r sha1sum")
            mode = "find-sha1"
    return {"run_id": run_id, "diff": diff, "mode": mode,
            "n_actions": len(ordered), "exit_codes": exits,
            "exit_matched": matched, "exit_compared": compared,
            "fidelity": (matched / compared) if compared else None}


def arm(vec_by_class, distfn):
    same, diff = [], []
    classes = sorted(vec_by_class)
    for i, ca in enumerate(classes):
        va = vec_by_class[ca]
        for x in range(len(va)):
            for y in range(x + 1, len(va)):
                same.append(distfn(va[x], va[y]))
        for cb in classes[i + 1:]:
            for u in va:
                for v in vec_by_class[cb]:
                    diff.append(distfn(u, v))
    s, d = np.array(same), np.array(diff)
    if not len(s) or not len(d):
        return None
    return {"auroc_jaccard": auroc(s, d), "n_same": len(s), "n_diff": len(d),
            "same_mean": float(s.mean()), "diff_mean": float(d.mean()),
            "exact_match_rate_same": float((s == 0).mean()),
            "exact_match_rate_diff": float((d == 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="data/a0_v3_32b")
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--classes", default="data/interpretation_classes.json")
    ap.add_argument("--labels-debug",
                    default="models/v3_32b_fixed_debug/labels_debug.jsonl")
    ap.add_argument("--n-tasks", type=int, default=5,
                    help="pilot tasks, taken from the SAME 60 collect_v2 derives")
    ap.add_argument("--exec-timeout", type=int, default=120)
    ap.add_argument("--work-dir", default="/opt/dlami/nvme/wta-replay")
    ap.add_argument("--fidelity-gate", type=float, default=0.80,
                    help="mean per-run exit-code agreement below which the "
                         "approach is declared NO-GO")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="results/replay_diff_pilot.json")
    args = ap.parse_args()

    t0 = time.time()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    art = load_class_artifact(args.classes)
    committed = load_commitments(Path(args.labels_debug))
    # sealed pool excluded BY CONSTRUCTION: same walk as collect_v2.main()
    eligible = eligible_tasks(Path(args.tasks_dir), Path(args.classes), 60)
    pilot = eligible[:args.n_tasks]
    images = {d.name: (d / "shared" / "image_ref.txt").read_text(
        encoding="utf-8").strip() for d in pilot}
    print(f"pilot tasks ({len(pilot)} of {len(eligible)} eligible): "
          f"{[d.name for d in pilot]}")

    actions = load_task_actions(Path(args.a0), None)
    actions = {t: r for t, r in actions.items() if t in images}
    n_runs = sum(len(r) for r in actions.values())
    n_execs = sum(len(a) for r in actions.values() for a in r.values())
    print(f"{n_runs} runs, {n_execs} execs to replay")
    if args.dry_run:
        for t, r in sorted(actions.items()):
            print(f"  {t}: {len(r)} runs, image {images[t]}")
        return 0

    # ---- replay (resumable) ----
    replays = {}
    for task, runs in sorted(actions.items()):
        for rid, acts in sorted(runs.items()):
            cache = work / f"{rid}.json"
            if cache.exists():
                replays[(task, rid)] = json.loads(cache.read_text(encoding="utf-8"))
                continue
            try:
                rec = replay_run(images[task], rid, acts, args.exec_timeout)
            except Exception as e:  # a dead image must not kill the pilot
                rec = {"run_id": rid, "error": f"{type(e).__name__}: {e}"}
            cache.write_text(json.dumps(rec), encoding="utf-8")
            replays[(task, rid)] = rec
            f = rec.get("fidelity")
            print(f"  {rid}: fidelity {'n/a' if f is None else f'{f:.2f}'} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    ok = {k: v for k, v in replays.items() if "error" not in v}
    fids = [v["fidelity"] for v in ok.values() if v.get("fidelity") is not None]
    fidelity = float(np.mean(fids)) if fids else None
    res = {"note": "PILOT, not a pre-registered 028 cell. Gates the full "
                   "replay-and-diff run before spending box time.",
           "pilot_tasks": [d.name for d in pilot],
           "n_runs_attempted": len(replays), "n_runs_replayed": len(ok),
           "n_runs_failed": len(replays) - len(ok),
           "Q1_replay_fidelity": {
               "mean_exit_code_agreement": fidelity,
               "median": float(np.median(fids)) if fids else None,
               "p25": float(np.percentile(fids, 25)) if fids else None,
               "frac_runs_above_0.9": (float(np.mean(np.array(fids) >= 0.9))
                                       if fids else None),
               "gate": args.fidelity_gate,
               "verdict": ("NO-GO: replay does not reproduce the recorded runs; "
                           "diffs describe runs that never happened"
                           if fidelity is None or fidelity < args.fidelity_gate
                           else "GO on fidelity")},
           "diff_modes": {}}
    for v in ok.values():
        res["diff_modes"][v.get("mode", "?")] = res["diff_modes"].get(
            v.get("mode", "?"), 0) + 1

    norm = {k: normalize_diff(v.get("diff") or "") for k, v in ok.items()}
    res["diff_sizes"] = {
        "median_changed_lines": float(np.median([len(x) for x in norm.values()]))
        if norm else None,
        "n_empty_diffs": int(sum(1 for x in norm.values() if not x))}

    # ---- Q2: the positive control, same three arms ----
    arms = {}
    by_task = defaultdict(list)
    for (task, _), v in norm.items():
        by_task[task].append(v)
    arms["A_cross_task"] = arm(by_task, jaccard_distance)

    s_all, d_all = [], []
    for task, runs in actions.items():
        groups = defaultdict(list)
        for rid in runs:
            if (task, rid) not in norm:
                continue
            files = frozenset(
                f for a in runs[rid] for f in ((a.observables or {}).get("files") or []))
            if files:
                groups[files].append(norm[(task, rid)])
        if len(groups) >= 2:
            r = arm(groups, jaccard_distance)
            if r:
                s_all.append(r)
    arms["B_file_set"] = ({"n_tasks_contributing": len(s_all),
                           "auroc_jaccard_mean": float(np.mean(
                               [r["auroc_jaccard"] for r in s_all]))}
                          if s_all else None)

    cells = []
    for task, runs in actions.items():
        for blocker in art.get(task, {}):
            vecs = defaultdict(list)
            for rid in runs:
                c = committed.get((rid, blocker))
                if c is not None and (task, rid) in norm:
                    vecs[c].append(norm[(task, rid)])
            if len(vecs) >= 2:
                r = arm(vecs, jaccard_distance)
                if r:
                    cells.append(r)
    arms["C_interpretation"] = ({"n_cells": len(cells),
                                 "auroc_jaccard_mean": float(np.mean(
                                     [r["auroc_jaccard"] for r in cells])),
                                 "reference_published_r3_r4": [0.555, 0.580]}
                                if cells else None)
    res["Q2_positive_control"] = arms
    a = arms.get("A_cross_task")
    res["verdict"] = (
        res["Q1_replay_fidelity"]["verdict"] if
        (fidelity is None or fidelity < args.fidelity_gate) else
        (f"GO: fidelity {fidelity:.2f}, cross-task AUROC "
         f"{a['auroc_jaccard']:.3f}" if a and a["auroc_jaccard"] >= 0.95 else
         f"NO-GO on the ceiling: cross-task AUROC "
         f"{a['auroc_jaccard']:.3f} < 0.95 — replay-and-diff does not resolve "
         f"either" if a else "INCONCLUSIVE: no cross-task pairs"))

    res["elapsed_s"] = round(time.time() - t0, 1)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nQ1 fidelity: {fidelity}")
    print(f"VERDICT: {res['verdict']}")
    print(f"wrote {p}  ({res['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
