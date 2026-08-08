"""Phase-4 eval runner (GPU box): our-loop arms over hil-bench tasks.

    python scripts/run_eval.py --artifacts models/<gates_dir> \
        --arms no_ask,full_info,model_initiated,detector \
        --tasks swe_60..swe_99 --out data/eval --shard 0 --num-shards 4

Copies collect_v2.py's orchestration conventions: manifest with env_info +
effective_generation_config, resume (skip a pass whose committed.json
exists), --shard/--num-shards over tasks, run ids {task}-p{pass}-s{seed}.
Protocol pins (decisions/022): nudge asserted OFF, n_runs refuses null
outside --smoke, single-trajectory arms at temperature 1.0, thresholds only
from --artifacts. RUNNING the sealed pool waits for A4 gates + owner
sign-off (AWS_RUNBOOK step 5) -- this script does not check that for you.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

SINGLE_ARMS = ("no_ask", "full_info", "model_initiated")
N_ARMS = ("detector", "output_divergence", "probe", "verbalized", "random")

# collect_v2's DELIBERATION_NUDGE forbids asking -- self-sabotage for ask
# arms. Assert its fingerprint never reaches an eval instruction (022 §2e).
_NUDGE_FINGERPRINT = "Do not ask for clarification"

GROUNDING = ("\n\nYou are working inside the repository this task refers to. "
             "Explore it with shell commands as needed.")

_RANGE = re.compile(r"^([A-Za-z_]+)(\d+)\.\.(?:[A-Za-z_]+)?(\d+)$")


def resolve_n_runs(cfg_n_runs, smoke: bool) -> int:
    """decisions/022 §1: n_runs ships null; refuse outside --smoke."""
    if cfg_n_runs is None:
        if not smoke:
            raise SystemExit(
                "configs/eval.yaml n_runs is null: the eval N is pre-registered "
                "in decisions/ after the R1 pilot (decisions/022). Set it there "
                "first, or pass --smoke for a non-sealed dry run (N=4).")
        return 4
    return int(cfg_n_runs)


def parse_task_selector(spec: str) -> list[str]:
    """'swe_0,swe_4' or 'swe_60..swe_99' (inclusive) -> ordered id list."""
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        m = _RANGE.match(part)
        if m:
            prefix, a, b = m.group(1), int(m.group(2)), int(m.group(3))
            out.extend(f"{prefix}{i}" for i in range(a, b + 1))
        elif part:
            out.append(part)
    return out


def check_instruction(instruction: str) -> str:
    if _NUDGE_FINGERPRINT in instruction:
        raise ValueError("collection nudge found in an eval instruction "
                         "(decisions/022 §2e)")
    return instruction


def build_arm_specs(names, *, n_runs, ladder, single_temp, cfg, args,
                    runtime=None, layer_pos=None):
    """ArmSpec + policy factory per requested arm name."""
    from wta.eval.orchestrator import ArmSpec
    from wta.eval.policies import (
        DetectorPolicy, OutputDivergencePolicy, ProbePolicy, RandomPolicy,
        VerbalizedPolicy,
    )

    ladder = tuple(ladder)
    specs = {}
    for name in names:
        if name == "no_ask":
            specs[name] = ArmSpec(name, 1, temperatures=(single_temp,))
        elif name == "full_info":
            specs[name] = ArmSpec(name, 1, full_info=True,
                                  temperatures=(single_temp,))
        elif name == "model_initiated":
            specs[name] = ArmSpec(name, 1, ask_affordance=True,
                                  temperatures=(single_temp,))
        elif name == "detector":
            if runtime is None:
                raise SystemExit("detector arm needs --artifacts")
            specs[name] = ArmSpec(name, n_runs, temperatures=ladder,
                                  policy_factory=lambda: DetectorPolicy(runtime))
        elif name == "output_divergence":
            specs[name] = ArmSpec(name, n_runs, temperatures=ladder,
                                  policy_factory=OutputDivergencePolicy)
        elif name == "probe":
            if not args.b3_probe:
                raise SystemExit("probe arm needs --b3-probe <path>")
            import pickle
            probe = pickle.loads(Path(args.b3_probe).read_bytes())
            thr, agg = cfg["b3"]["threshold"], cfg["b3"]["agg"]
            specs[name] = ArmSpec(
                name, n_runs, temperatures=ladder,
                policy_factory=lambda: ProbePolicy(probe, thr, layer_pos, agg))
        elif name == "verbalized":
            b2 = cfg["b2"]
            specs[name] = ArmSpec(
                name, n_runs, temperatures=ladder,
                policy_factory=lambda: VerbalizedPolicy(b2["threshold"],
                                                        b2["every"]))
        elif name == "random":
            budget = args.b4_budget if args.b4_budget is not None \
                else cfg["b4"]["budget"]
            if budget is None:
                raise SystemExit(
                    "random arm needs --b4-budget (the detector arm's measured "
                    "asks/task; B4 runs LAST -- decisions/022 §2i)")
            specs[name] = ArmSpec(
                name, n_runs, temperatures=ladder,
                policy_factory=lambda: RandomPolicy(float(budget),
                                                    cfg["max_steps"]))
        else:
            raise SystemExit(f"unknown arm {name!r} "
                             f"(known: {SINGLE_ARMS + N_ARMS})")
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--tasks", required=True,
                    help="'swe_60..swe_99' or comma list; sealed pool only "
                         "after gates + sign-off")
    ap.add_argument("--arms", default="no_ask,full_info,model_initiated,detector")
    ap.add_argument("--artifacts", default=None,
                    help="offline model dir (a1/a2/a3/gate_report) -- the ONLY "
                         "source of detector thresholds")
    ap.add_argument("--model-id", default="Qwen/Qwen3-32B")
    ap.add_argument("--enable-thinking", action="store_true", default=False)
    ap.add_argument("--judge", choices=("api", "mock"), default="api",
                    help="api = JUDGE_BASE_URL (:8808 local vLLM); mock only "
                         "for plumbing dry runs, never for reported numbers")
    ap.add_argument("--b3-probe", default=None)
    ap.add_argument("--b4-budget", type=float, default=None)
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="non-sealed dry run: allows n_runs=null (N=4)")
    ap.add_argument("--scratch-dir", default=None)
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    n_runs = resolve_n_runs(cfg.get("n_runs"), args.smoke)
    passes = args.passes or int(cfg["passes"])
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]

    # heavy imports only past arg validation
    from collect_a0 import env_info, log_event
    from extract_task_context import image_available, try_load_archive

    from wta.agent_env import DockerTaskEnv
    from wta.agent_loop import AgentLoopConfig
    from wta.eval.artifacts import DetectorRuntime, load_artifacts
    from wta.eval.orchestrator import run_task_pass
    from wta.hf_reader import HFStreamReader
    from wta.logging_schema import save_run_log
    from wta.reads import DEFAULT_VALUE_PATTERN
    from xtid.harness.ask_f1 import AskSession
    from xtid.harness.judge import build_judge
    from xtid.harness.tasks import load_hil_bench_tasks

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.num_shards == 1 else f".s{args.shard}"
    events = out_root / f"events{suffix}.jsonl"
    manifest_path = out_root / f"eval_manifest{suffix}.json"

    reader = HFStreamReader(
        args.model_id, mid_layer=cfg["mid_layer"], layers=cfg["layers"],
        cadence=cfg["cadence"], value_pattern=DEFAULT_VALUE_PATTERN,
        enable_thinking=args.enable_thinking,
        top_p=cfg["top_p"], top_k=cfg["top_k"], min_p=cfg["min_p"])

    runtime = None
    layer_pos = None
    if args.artifacts:
        art = load_artifacts(args.artifacts)
        # the A2 was trained at mid_layer; find its position in the captured stack
        layer_pos = reader.layer_indices.index(reader.mid_layer) \
            if reader.layer_indices and reader.mid_layer in reader.layer_indices else None
        runtime = DetectorRuntime(art, layer_pos=layer_pos, sealed=not args.smoke)

    judge = build_judge({"kind": args.judge})
    specs = build_arm_specs(arm_names, n_runs=n_runs,
                            ladder=cfg["temps_ladder"],
                            single_temp=cfg["single_temperature"],
                            cfg=cfg, args=args, runtime=runtime,
                            layer_pos=layer_pos)

    wanted = parse_task_selector(args.tasks)
    all_tasks = {t.instance_id: t
                 for t in load_hil_bench_tasks(domain="swe",
                                               root=Path(args.tasks_dir).parent)}
    by_dir = {Path(t.meta["task_dir"]).name: t for t in all_tasks.values()}
    tasks = [by_dir[w] for w in wanted if w in by_dir]
    missing = [w for w in wanted if w not in by_dir]
    if missing:
        print(f"WARNING: {len(missing)} selected tasks not found: {missing[:5]}")
    my_tasks = tasks[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(my_tasks)} of "
          f"{len(tasks)} tasks; arms={arm_names}; N={n_runs}; passes={passes}")

    manifest = {"args": vars(args), "config": cfg, "env": env_info(),
                "n_runs": n_runs,
                "generation": reader.effective_generation_config(),
                "artifacts": args.artifacts, "results": {}}
    log_event(events, event="eval_start", args=vars(args))

    loop_cfg = AgentLoopConfig(max_steps=cfg["max_steps"],
                               max_new_tokens_per_turn=cfg["max_new_tokens"])

    for task in my_tasks:
        task_dir = Path(task.meta["task_dir"])
        task_key = task_dir.name
        image = (task_dir / "shared" / "image_ref.txt").read_text(
            encoding="utf-8").strip()
        if not image_available(image):
            load_log: list[str] = []
            try_load_archive(task_dir, load_log, scratch_dir=args.scratch_dir)
            if not image_available(image):
                manifest["results"][task_key] = "SKIPPED: image unavailable"
                log_event(events, event="task_skipped", task=task_key)
                continue

        task.statement = check_instruction(task.statement + GROUNDING)

        for arm_name, spec in specs.items():
            for pass_idx in range(passes):
                pass_dir = out_root / task_key / arm_name / f"pass_{pass_idx}"
                if (pass_dir / "committed.json").exists():
                    continue  # resume
                pass_dir.mkdir(parents=True, exist_ok=True)
                ask_session = AskSession()
                envs = []

                def env_factory(run_id, _image=image, _envs=envs):
                    env = DockerTaskEnv(_image, name=f"wta-eval-{run_id}",
                                        exec_timeout=cfg["exec_timeout"])
                    env.start()
                    _envs.append(env)
                    return env

                t0 = time.time()
                try:
                    res = run_task_pass(
                        spec, task, session=reader, env_factory=env_factory,
                        judge=judge, ask_session=ask_session, cfg=loop_cfg,
                        pass_idx=pass_idx, seed_base=pass_idx * 100,
                        model_id=args.model_id, mid_layer=reader.mid_layer,
                        layers=reader.layer_indices)
                except Exception as e:
                    log_event(events, event="pass_error", task=task_key,
                              arm=arm_name, p=pass_idx,
                              error=f"{type(e).__name__}: {e}")
                    print(f"{task_key}/{arm_name}/p{pass_idx}: ERROR {e}")
                    continue
                finally:
                    for env in envs:
                        try:
                            env.stop()
                        except Exception:
                            pass

                for run_id, rr in res.runs.items():
                    save_run_log(rr.log, pass_dir)
                    (pass_dir / f"{run_id}.segments.json").write_text(
                        json.dumps(rr.segments), encoding="utf-8")
                (pass_dir / "ask_log.json").write_text(json.dumps({
                    "events": res.ask_events, "stats": res.policy_stats,
                    "instance_key": res.instance_key,
                    "session": {iid: {"n_blockers": lg.n_blockers,
                                      "questions": lg.questions}
                                for iid, lg in ask_session.logs.items()},
                }, indent=1), encoding="utf-8")
                (pass_dir / "compute.json").write_text(
                    json.dumps(res.compute, indent=1), encoding="utf-8")
                (pass_dir / "committed.json").write_text(json.dumps({
                    "committed_run_id": res.committed_run_id,
                    "seeds": res.seeds,
                    "finished": {r: res.runs[r].finished for r in res.runs},
                    "prediction": res.prediction,
                }, indent=1), encoding="utf-8")

                dt = time.time() - t0
                manifest["results"].setdefault(task_key, {}).setdefault(
                    arm_name, {})[f"pass_{pass_idx}"] = {
                    "runs": len(res.runs), "asks": len(res.ask_events),
                    "committed": res.committed_run_id, "seconds": round(dt, 1)}
                log_event(events, event="pass_done", task=task_key,
                          arm=arm_name, p=pass_idx, asks=len(res.ask_events),
                          secs=round(dt, 1))
                print(f"{task_key}/{arm_name}/p{pass_idx}: "
                      f"{res.compute['turns']} turns, "
                      f"{len(res.ask_events)} asks, {dt:.0f}s")
                manifest_path.write_text(json.dumps(manifest, indent=1),
                                         encoding="utf-8")

    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log_event(events, event="eval_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
