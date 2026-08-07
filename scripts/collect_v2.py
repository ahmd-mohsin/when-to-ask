"""A0 v2 collection (AWS GPU box): REAL agent trajectories with tool calls.

    python scripts/collect_v2.py --n-tasks 20 --scratch-dir /opt/dlami/nvme/wta-scratch

The end-state collector (decisions/017): for each task, N seeded agent runs
inside the task's own docker container (observe -> think -> act, one shell
command per turn), reading mid-layer residuals DURING each turn's generation
at cadence + cue + value positions, across 4 layers, logging every action as
an offline label observable. Everything the previous collectors learned is
baked in: grounded containers, multi-layer capture (decisions/014), value
reads (decisions/016), resumability, manifest + events diagnostics (ADR 012).

Outputs per run: <run_id>.npz (R, L, H) + <run_id>.json (reads/actions meta)
+ <run_id>.segments.json (per-turn generated text -- labeling maps reads
through these) + <run_id>.txt (joined, human-readable).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_a0 import env_info, log_event  # noqa: E402
from extract_task_context import image_available, try_load_archive  # noqa: E402

from wta.agent_env import DockerTaskEnv  # noqa: E402
from wta.agent_loop import AgentLoopConfig, run_agent  # noqa: E402
from wta.hf_reader import HFStreamReader  # noqa: E402
from wta.logging_schema import save_run_log  # noqa: E402
from wta.reads import DEFAULT_VALUE_PATTERN  # noqa: E402


# The method doc's A0 says to force runs to actually disagree ("vary seed and
# temperature, and optionally a light persona nudge"); v1.5 had a nudge and the
# v2 collector dropped it (decisions/021 §1). INTERPRETATION-NEUTRAL by
# construction: it asks the model to commit to *a* reading of any under-spec,
# and must never name or hint at a specific interpretation class -- naming one
# would contaminate the very labels this data trains.
DELIBERATION_NUDGE = (
    "Where the task leaves a choice open, state the alternatives you see in "
    "your THOUGHT, pick one, and proceed with it. Do not ask for "
    "clarification; commit to your reading and implement it."
)


def artifact_task_ids(classes_path) -> set[str]:
    """Task ids covered by an interpretation-class artifact (train pool)."""
    art = json.loads(Path(classes_path).read_text(encoding="utf-8"))
    return {k for k in art if not k.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--classes", default=None,
                    help="path to interpretation_classes.json: collect ONLY "
                         "tasks with a class artifact (the train pool). "
                         "Without this, tasks are taken in sorted-dir order, "
                         "which interleaves numbering (swe_60 before swe_7) "
                         "and would touch the sealed test pool.")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--n-runs", type=int, default=8)
    ap.add_argument("--mid-layer", type=float, default=0.5)
    # decisions/021 §6: hook capture makes extra layers ~free (KB/step) and
    # select-at-load means the sweep costs zero extra GPU runs. Span 0.2-0.85;
    # the old 0.4-0.7 band was never swept on real agent data.
    ap.add_argument("--layers", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85")
    # decisions/021 §3: the read selector is rebuilt per TURN, so cadence K
    # means a turn shorter than K tokens yields ZERO reads. At K=32 that gave
    # median 8 reads/run and 26% zero-read runs. Spec A0's {16,32,64} sweep
    # was never run; 8 is the agent-loop default until it is.
    ap.add_argument("--cadence", type=int, default=8)
    ap.add_argument("--no-value-reads", action="store_true",
                    help="disable value-triggered reads (ON by default in v2, "
                         "decisions/016)")
    ap.add_argument("--enable-thinking", action="store_true", default=False,
                    help="Qwen3 hybrid thinking mode. Default OFF and pinned "
                         "explicitly (decisions/019 addendum: the paper is "
                         "silent, the hil-bench repo's own Qwen configs use "
                         "the non-thinking Instruct variant). Recorded in the "
                         "manifest. No-op for Qwen2.5 templates.")
    # decisions/021 §2: at 15 steps, 81% of the 32B runs were CENSORED at the
    # cap and half the forked blockers had zero finished runs -- late decisions
    # were never observed. 1024 tokens also truncated a run mid-bash-block.
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--exec-timeout", type=int, default=120)
    # decisions/021 §1: sampling diversity. generate() previously passed only
    # temperature, inheriting the model's generation_config (Qwen3 ships
    # top_k=20), which caps interpretation diversity no matter the temperature.
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--temps", default="0.7,0.9,1.1,1.3",
                    help="temperature ladder, cycled over seeds INDEPENDENTLY "
                         "of the seed value (decisions/021 §1: the old TEMPS "
                         "were tied to seed%%3, confounding the two levers)")
    ap.add_argument("--no-nudge", action="store_true",
                    help="ablate the deliberation nudge. The method doc's A0 "
                         "prescribes a nudge to make runs actually disagree; "
                         "v1.5 had one and the v2 collector dropped it "
                         "(decisions/021 §1). Interpretation-NEUTRAL by "
                         "construction: it must never name a class.")
    ap.add_argument("--scratch-dir", default=None)
    ap.add_argument("--out", default="data/a0_v2")
    # Data-parallel across GPUs: Qwen3-32B bf16 (~65GB) fits on ONE 96GB card,
    # so N cards = N independent workers, each owning a disjoint slice of tasks
    # (near-linear scaling; no tensor sharding). Launch one process per GPU with
    # CUDA_VISIBLE_DEVICES=<i> and --shard <i> --num-shards <N>, all writing to
    # the SAME --out (per-task subdirs never collide) -- see AWS_RUNBOOK 2d.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    # Parallel shards share --out (per-task subdirs never collide) but must NOT
    # share these two append/rewrite files -- suffix them per shard and merge
    # at analysis time.
    suffix = "" if args.num_shards == 1 else f".s{args.shard}"
    events = out_root / f"events{suffix}.jsonl"
    manifest_path = out_root / f"collection_manifest{suffix}.json"

    layer_specs = [float(x) if "." in x else int(x) for x in args.layers.split(",")] \
        if args.layers and args.layers.lower() != "none" else None
    reader = HFStreamReader(
        args.model_id, mid_layer=args.mid_layer, layers=layer_specs,
        cadence=args.cadence,
        value_pattern=None if args.no_value_reads else DEFAULT_VALUE_PATTERN,
        enable_thinking=args.enable_thinking,
        top_p=args.top_p, top_k=args.top_k, min_p=args.min_p)
    temps = tuple(float(t) for t in args.temps.split(","))
    manifest = {"args": vars(args), "env": env_info(),
                "reader": {"n_layers": reader.n_layers, "hidden_dim": reader.hidden_dim,
                           "mid_layer": reader.mid_layer,
                           "layer_indices": reader.layer_indices},
                # decisions/021 R0: record what the model would have defaulted
                # to alongside what we override -- never assume the config.
                "generation": reader.effective_generation_config(),
                "temps": list(temps),
                "tasks": {}}
    print(f"generation config: {reader.effective_generation_config()}")
    log_event(events, event="v2_collection_start", args=vars(args))

    tasks_dir = Path(args.tasks_dir)
    class_tasks = artifact_task_ids(args.classes) if args.classes else None
    # Resolve the eligible task list ONCE, then slice this shard's stride out
    # of it. --n-tasks is the GLOBAL count (truncate before sharding), so the
    # same command with different --shard covers the same 60 tasks between
    # workers rather than 60 each.
    eligible = []
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        if class_tasks is not None and task_dir.name not in class_tasks:
            continue
        if not (task_dir / "baseline" / "instruction.md").exists():
            continue
        if not (task_dir / "shared" / "image_ref.txt").exists():
            continue
        eligible.append(task_dir)
        if len(eligible) >= args.n_tasks:
            break
    my_tasks = eligible[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(my_tasks)} of "
          f"{len(eligible)} eligible tasks")

    for task_dir in my_tasks:
        instr_f = task_dir / "baseline" / "instruction.md"
        ref_f = task_dir / "shared" / "image_ref.txt"
        task_id = task_dir.name
        image = ref_f.read_text(encoding="utf-8").strip()
        out_dir = out_root / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        t_rec = manifest["tasks"].setdefault(task_id, {"image": image, "runs": {}})

        # image must be loadable (same ladder as the context extractor)
        if not image_available(image):
            load_log: list[str] = []
            try_load_archive(task_dir, load_log, scratch_dir=args.scratch_dir)
            if not image_available(image):
                t_rec["status"] = "SKIPPED: image unavailable"
                t_rec["image_log"] = load_log[-3:]
                log_event(events, event="task_skipped", task=task_id)
                print(f"{task_id}: SKIPPED (image unavailable)")
                continue

        instruction = instr_f.read_text(encoding="utf-8", errors="replace")
        instruction += ("\n\nYou are working inside the repository this task "
                        "refers to. Explore it with shell commands as needed.")
        if not args.no_nudge:
            instruction += "\n\n" + DELIBERATION_NUDGE

        for seed in range(args.n_runs):
            run_id = f"{task_id}-s{seed}"
            if (out_dir / f"{run_id}.json").exists():
                t_rec["runs"][run_id] = {"status": "already-present (resumed)"}
                continue
            cfg = AgentLoopConfig(max_steps=args.max_steps,
                                  max_new_tokens_per_turn=args.max_new_tokens,
                                  temperature=temps[seed % len(temps)])
            log_event(events, event="run_start", run=run_id, temp=cfg.temperature)
            t0 = time.time()
            try:
                with DockerTaskEnv(image, name=f"wta-{run_id}",
                                   exec_timeout=args.exec_timeout) as env:
                    res = run_agent(reader, env, instruction, run_id=run_id,
                                    task_id=task_id, seed=seed, cfg=cfg,
                                    model_id=args.model_id,
                                    mid_layer=reader.mid_layer,
                                    layers=reader.layer_indices)
            except Exception as e:
                log_event(events, event="run_error", run=run_id,
                          error=f"{type(e).__name__}: {e}")
                t_rec["runs"][run_id] = {"status": f"ERROR: {type(e).__name__}: {e}"}
                print(f"{run_id}: ERROR {e}")
                continue
            dt = time.time() - t0
            save_run_log(res.log, out_dir)
            (out_dir / f"{run_id}.segments.json").write_text(
                json.dumps(res.segments), encoding="utf-8")
            (out_dir / f"{run_id}.txt").write_text("\n\n".join(res.segments),
                                                   encoding="utf-8")
            trig = {t: sum(1 for r in res.log.reads if r.trigger == t)
                    for t in ("cadence", "cue", "value")}
            t_rec["runs"][run_id] = {
                "status": "ok", "steps": res.n_steps, "finished": res.finished,
                "stop_reason": res.stop_reason, "reads": len(res.log.reads),
                "reads_by_trigger": trig, "actions": len(res.log.actions),
                "seconds": round(dt, 1),
            }
            log_event(events, event="run_done", run=run_id, steps=res.n_steps,
                      finished=res.finished, reads=len(res.log.reads),
                      secs=round(dt, 1))
            print(f"{run_id}: {res.n_steps} steps ({res.stop_reason}), "
                  f"{len(res.log.reads)} reads {trig}, "
                  f"{len(res.log.actions)} actions, {dt:.0f}s")
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    n_ok = sum(1 for t in manifest["tasks"].values()
               for r in t["runs"].values() if r.get("status") == "ok")
    n_fin = sum(1 for t in manifest["tasks"].values()
                for r in t["runs"].values() if r.get("finished"))
    print(f"\nshard {args.shard}/{args.num_shards}: {len(my_tasks)} tasks; "
          f"{n_ok} runs ok, {n_fin} reached TASK_DONE. Manifest: {manifest_path}")
    log_event(events, event="v2_collection_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
