"""Consequence-grounded fork labels: replay a run, then run the task's own tests.

The lexicon-based absolute read-out is confounded
(results/canonical_accuracy.json: the labeler picks the class with the most
signatures 52.5% of the time against a 38.0% chance rate, and canonical is
always class index 0). This replaces the label entirely. Every hil-bench task
ships `test_patch`, `test_cmd`, `log_parser` and a `FAIL_TO_PASS` list in
`shared/metadata.json`, and the runner + parser live inside the image. So:
replay the run, apply the task's test patch, run the task's tests, and read
the FAIL_TO_PASS pass/fail vector. That label owes nothing to the lexicon, to
an anchor, or to a judge -- it is what the run's code actually does.

Q: does the F2P vector VARY across a task's runs? If it does, there is a fork
label grounded in consequence. If it is constant, that is publishable too: the
benchmark's own tests do not discriminate the blocker.

THE NO-OP CONTROL, which the diff pilot lacked. 22.6% of pilot runs changed
nothing and 8.1% of commitments sit on such runs, and a run that wrote nothing
produces an unchanged test vector for reasons that have nothing to do with the
blocker. Two guards, both recorded rather than assumed:

  1. `repo_changed` per run, from the same normalized git diff the pilot used.
  2. A per-task ZERO-ACTION BASELINE: patch + tests on a pristine container
     with no replay at all. Any run whose vector equals that baseline achieved
     nothing measurable, whether or not it wrote bytes. Entropy is reported
     over all runs AND over consequential runs only.

Deviations from the shipped `test_cmd`, both deliberate and recorded:

  - `test_cmd` invokes `run_script.sh` with NO arguments, which runs the FULL
    suite (`ansible-test units` + `sanity`). That is infeasible 67 times over.
    The runner accepts explicit test paths, so it is given the unique files
    named by FAIL_TO_PASS -- one invocation per file, same runner, same
    parser.
  - Files touched by `test_patch` are restored from HEAD BEFORE the patch is
    applied, so a run that edited the tests cannot fake a pass.

SCOPE: python tasks only. The `sweap_json` parser mis-parses jest output on
the two JS tasks (swe_1, swe_12): it emits concatenated test names under a
`src/app/...` prefix that never matches the `applications/drive/src/app/...`
names in FAIL_TO_PASS, and marks tests PASSED whose lines carry jest's failure
glyph. Verified against a pristine swe_1 container: 0 of 4 F2P names matched
and all 5 reported tests were "PASSED" with failure markers in the text. The
three python tasks matched 6/6, 7/7 and 5/5 F2P names, all FAILED at baseline
as they must be pre-fix. The JS tasks are excluded and the defect recorded.

    python scripts/test_outcome_vector.py --work-dir /ssd3/wta-testvec
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from offline_ask_headtohead import load_task_actions  # noqa: E402
from replay_diff_pilot import normalize_diff  # noqa: E402
from wta.agent_env import DockerTaskEnv  # noqa: E402

PY_TASKS = ["swe_0", "swe_10", "swe_11"]
JS_EXCLUDED = {"swe_1": "sweap_json mis-parses jest output",
               "swe_12": "sweap_json mis-parses jest output"}
_TOUCHED = re.compile(r"^diff --git a/(\S+)", re.M)


def load_meta(tasks_dir: Path, task: str) -> dict:
    return json.loads((tasks_dir / task / "shared" / "metadata.json")
                      .read_text(encoding="utf-8"))


def prep_patch(meta: dict, dest: Path) -> list[str]:
    """Write test_patch with a trailing newline (git apply rejects it without
    one) and return the paths it touches."""
    p = meta.get("test_patch") or ""
    if p and not p.endswith("\n"):
        p += "\n"
    dest.write_text(p, encoding="utf-8")
    return sorted(set(_TOUCHED.findall(p)))


def f2p_files(meta: dict) -> tuple[list[str], list[str]]:
    f2p = meta["swe_bench_metadata"]["FAIL_TO_PASS"]
    files = sorted({x.split("::")[0].split("|")[0].strip() for x in f2p})
    return f2p, files


def run_tests(env: DockerTaskEnv, container: str, patch_host: Path,
              touched: list[str], files: list[str]) -> dict:
    """Restore test files, apply test_patch, run the task's tests, parse."""
    subprocess.run(["docker", "cp", str(patch_host), f"{container}:/tmp/tp.patch"],
                   capture_output=True, check=False)
    if touched:
        # delete any agent-authored version FIRST, then restore HEAD's if the
        # file existed there. A test file created by test_patch is absent from
        # HEAD, so checkout alone would leave the agent's copy in place and it
        # could fake a pass.
        q = " ".join(f"'{t}'" for t in touched)
        env.execute(f"rm -f {q}; git checkout HEAD -- {q} 2>/dev/null; true")
    code, _ = env.execute("git apply /tmp/tp.patch")
    if code != 0:
        return {"error": "test_patch did not apply"}
    args = " ".join(f"'{f}'" for f in files)
    env.execute(f"bash /root/run_script.sh {args} "
                f">/tmp/o.log 2>/tmp/e.log; "
                f"python /root/parser.py /tmp/o.log /tmp/e.log /tmp/out.json")
    code, out = env.execute("cat /tmp/out.json")
    if code != 0:
        return {"error": "no parser output"}
    try:
        return {"tests": {t["name"]: t["status"]
                          for t in json.loads(out)["tests"]}}
    except Exception as e:
        return {"error": f"unparseable: {type(e).__name__}"}


def vector(parsed: dict, f2p: list[str]) -> tuple[list | None, str]:
    """(vector, status). Tests that ran but never collected the F2P names is
    NOT missing data -- it is the run failing to produce importable code, so
    every FAIL_TO_PASS test is un-passed and the vector is all zeros. Verified
    on swe_10-s10: the replayed edits left an IndentationError in
    lib/ansible/module_utils/facts/network/linux.py and pytest errored during
    collection. Flagged separately from a clean all-fail so the two are never
    conflated."""
    if "tests" not in parsed:
        return None, parsed.get("error", "no_output")
    got = parsed["tests"]
    if any(n in got for n in f2p):
        return [1 if got.get(n) == "PASSED" else 0 for n in f2p], "ok"
    return [0] * len(f2p), "collection_error"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="/ssd/wta_data/a0_v3_32b")
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--work-dir", default="/ssd3/wta-testvec")
    ap.add_argument("--exec-timeout", type=int, default=120)
    ap.add_argument("--run-budget", type=float, default=300.0)
    ap.add_argument("--test-timeout", type=int, default=900)
    ap.add_argument("--out", default="results/test_outcome_vector.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    tasks_dir = Path(args.tasks_dir)
    actions = load_task_actions(Path(args.a0), None)

    plan = {}
    for t in PY_TASKS:
        meta = load_meta(tasks_dir, t)
        f2p, files = f2p_files(meta)
        touched = prep_patch(meta, work / f"{t}.patch")
        img = (tasks_dir / t / "shared" / "image_ref.txt").read_text().strip()
        plan[t] = {"image": img, "f2p": f2p, "files": files,
                   "touched": touched, "runs": sorted(actions.get(t, {}))}
        print(f"{t}: {len(plan[t]['runs'])} runs, {len(f2p)} F2P tests, "
              f"image {img}")
    if args.dry_run:
        return 0

    # ---- per-task zero-action baseline (the no-op control) ----
    baselines = {}
    for t, p in plan.items():
        cache = work / f"__baseline__{t}.json"
        if cache.exists():
            baselines[t] = json.loads(cache.read_text(encoding="utf-8"))
            continue
        name = f"wta-tv-base-{t}"
        with DockerTaskEnv(p["image"], name=name,
                           exec_timeout=args.test_timeout) as env:
            parsed = run_tests(env, name, work / f"{t}.patch",
                               p["touched"], p["files"])
        vec, st = vector(parsed, p["f2p"])
        rec = {"vector": vec, "status": st}
        cache.write_text(json.dumps(rec), encoding="utf-8")
        baselines[t] = rec
        print(f"  baseline {t}: {rec.get('vector')} ({time.time()-t0:.0f}s)")

    # ---- per-task GROUND-TRUTH positive control ----
    # Without this the whole read-out is uninterpretable: if the pipeline
    # cannot make FAIL_TO_PASS pass even from the task's own reference
    # solution, an all-zero agent vector says nothing about the agent.
    gt = {}
    for t, p in plan.items():
        cache = work / f"__groundtruth__{t}.json"
        if cache.exists():
            gt[t] = json.loads(cache.read_text(encoding="utf-8"))
            continue
        sol = tasks_dir / t / "baseline" / "solution" / "ground_truth.patch"
        gp = sol.read_text(encoding="utf-8")
        if not gp.endswith("\n"):
            gp += "\n"
        (work / f"{t}.gt.patch").write_text(gp, encoding="utf-8")
        name = f"wta-tv-gt-{t}"
        with DockerTaskEnv(p["image"], name=name,
                           exec_timeout=args.test_timeout) as env:
            subprocess.run(["docker", "cp", str(work / f"{t}.gt.patch"),
                            f"{name}:/tmp/gt.patch"], capture_output=True)
            code, _ = env.execute("git apply /tmp/gt.patch")
            parsed = run_tests(env, name, work / f"{t}.patch",
                               p["touched"], p["files"])
        v, st = vector(parsed, p["f2p"])
        rec = {"vector": v, "status": st, "apply_rc": code,
               "all_pass": v == [1] * len(p["f2p"])}
        cache.write_text(json.dumps(rec), encoding="utf-8")
        gt[t] = rec
        print(f"  ground-truth {t}: {v} all_pass={rec['all_pass']}")

    # ---- per-run replay + tests ----
    recs = {}
    for t, p in plan.items():
        for rid in p["runs"]:
            cache = work / f"{rid}.json"
            if cache.exists():
                recs[rid] = json.loads(cache.read_text(encoding="utf-8"))
                continue
            acts = sorted(actions[t][rid], key=lambda a: a.segment_idx)
            name = f"wta-tv-{rid}"
            rec = {"run": rid, "task": t, "n_actions": len(acts)}
            try:
                tr = time.time()
                with DockerTaskEnv(p["image"], name=name,
                                   exec_timeout=args.exec_timeout) as env:
                    env.execute("{ git config --global --add safe.directory "
                                "/app; git add -A; } >/dev/null 2>&1")
                    stopped = None
                    for i, a in enumerate(acts):
                        if time.time() - tr > args.run_budget:
                            stopped = {"after": i, "of": len(acts)}
                            break
                        env.execute("{ " + (a.action_text or "")
                                    + "\n} >/dev/null 2>&1")
                    rec["abandoned"] = stopped
                    _, diff = env.execute(
                        "git -c core.fileMode=false add -A >/dev/null 2>&1; "
                        "git -c core.fileMode=false diff --cached --no-color")
                    rec["repo_changed"] = bool(normalize_diff(diff or ""))
                    env.exec_timeout = args.test_timeout
                    parsed = run_tests(env, name, work / f"{t}.patch",
                                       p["touched"], p["files"])
                rec["vector"], rec["status"] = vector(parsed, p["f2p"])
                rec["wall_s"] = round(time.time() - tr, 1)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            cache.write_text(json.dumps(rec), encoding="utf-8")
            recs[rid] = rec
            print(f"  {rid}: vec={rec.get('vector')} "
                  f"changed={rec.get('repo_changed')} ({time.time()-t0:.0f}s)",
                  flush=True)

    # ---- analysis ----
    per_task = {}
    for t, p in plan.items():
        rs = [r for r in recs.values()
              if r["task"] == t and r.get("vector") is not None]
        base = baselines[t].get("vector")
        vecs = [tuple(r["vector"]) for r in rs]
        cons = [tuple(r["vector"]) for r in rs
                if tuple(r["vector"]) != tuple(base or [])]
        solved = [v for v in vecs if all(v)]
        per_task[t] = {
            "n_runs_scored": len(rs),
            "n_runs_failed": sum(1 for r in recs.values()
                                 if r["task"] == t and r.get("vector") is None),
            "n_f2p": len(p["f2p"]),
            "baseline_vector": base,
            "ground_truth_vector": gt[t]["vector"],
            "distinct_vectors": len(set(vecs)),
            "vector_counts": {"".join(map(str, k)): v
                              for k, v in Counter(vecs).most_common()},
            "varies": len(set(vecs)) > 1,
            "n_fully_solved": len(solved),
            "n_equal_to_baseline": sum(1 for v in vecs
                                       if v == tuple(base or [])),
            "n_repo_unchanged": sum(1 for r in rs if not r.get("repo_changed")),
            "n_collection_error": sum(1 for r in rs
                                      if r.get("status") == "collection_error"),
            "distinct_vectors_consequential_only": len(set(cons)),
            "varies_consequential_only": len(set(cons)) > 1,
        }

    all_scored = [r for r in recs.values() if r.get("vector") is not None]
    noop = [r for r in all_scored if not r.get("repo_changed")]
    noop_eq_base = sum(
        1 for r in noop
        if tuple(r["vector"]) == tuple(baselines[r["task"]].get("vector") or []))
    res = {
        "note": "Consequence-grounded fork labels from the task's OWN tests. "
                "Owes nothing to the lexicon, anchors or a judge.",
        "inputs": {"a0": args.a0, "tasks_dir": args.tasks_dir},
        "deviations": {
            "selected_tests_not_full_suite":
                "shipped test_cmd runs run_script.sh with no args (full suite); "
                "given the FAIL_TO_PASS files instead -- same runner, same parser",
            "test_files_restored_from_HEAD":
                "files touched by test_patch are checked out from HEAD before "
                "applying it, so a run that edited tests cannot fake a pass",
        },
        "excluded_tasks": JS_EXCLUDED,
        "js_parser_defect": (
            "sweap_json mis-parses jest: on a pristine swe_1 container it "
            "emitted 5 concatenated test names under a src/app/... prefix "
            "(FAIL_TO_PASS uses applications/drive/src/app/...), 0 of 4 F2P "
            "names matched, and every test was reported PASSED although the "
            "captured lines carry jest's failure glyph."),
        "no_op_control": {
            "n_scored": len(all_scored),
            "n_repo_unchanged": len(noop),
            "n_repo_unchanged_matching_baseline": noop_eq_base,
            "interpretation":
                "a run that changed nothing should reproduce the zero-action "
                "baseline exactly; any that does not indicates test flakiness "
                "or replay nondeterminism, not a decision",
        },
        "ground_truth_control": {
            t: gt[t] for t in plan},
        "instrument_validated": all(gt[t]["all_pass"] for t in plan),
        "per_task": per_task,
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for t, v in per_task.items():
        print(f"{t}: {v['n_runs_scored']} runs, {v['distinct_vectors']} "
              f"distinct vectors, varies={v['varies']}, "
              f"solved={v['n_fully_solved']}, baseline={v['baseline_vector']}")
    print(f"wrote {p} ({res['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
