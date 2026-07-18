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

from collect_a0 import TEMPS, env_info, log_event  # noqa: E402
from extract_task_context import image_available, try_load_archive  # noqa: E402

from wta.agent_env import DockerTaskEnv  # noqa: E402
from wta.agent_loop import AgentLoopConfig, run_agent  # noqa: E402
from wta.hf_reader import HFStreamReader  # noqa: E402
from wta.logging_schema import save_run_log  # noqa: E402
from wta.reads import DEFAULT_VALUE_PATTERN  # noqa: E402


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
    ap.add_argument("--layers", default="0.4,0.5,0.6,0.7")
    ap.add_argument("--cadence", type=int, default=32)
    ap.add_argument("--no-value-reads", action="store_true",
                    help="disable value-triggered reads (ON by default in v2, "
                         "decisions/016)")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--exec-timeout", type=int, default=120)
    ap.add_argument("--scratch-dir", default=None)
    ap.add_argument("--out", default="data/a0_v2")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    events = out_root / "events.jsonl"

    layer_specs = [float(x) if "." in x else int(x) for x in args.layers.split(",")] \
        if args.layers and args.layers.lower() != "none" else None
    reader = HFStreamReader(
        args.model_id, mid_layer=args.mid_layer, layers=layer_specs,
        cadence=args.cadence,
        value_pattern=None if args.no_value_reads else DEFAULT_VALUE_PATTERN)
    manifest = {"args": vars(args), "env": env_info(),
                "reader": {"n_layers": reader.n_layers, "hidden_dim": reader.hidden_dim,
                           "mid_layer": reader.mid_layer,
                           "layer_indices": reader.layer_indices},
                "tasks": {}}
    log_event(events, event="v2_collection_start", args=vars(args))

    tasks_dir = Path(args.tasks_dir)
    class_tasks = artifact_task_ids(args.classes) if args.classes else None
    done = 0
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        if done >= args.n_tasks:
            break
        if class_tasks is not None and task_dir.name not in class_tasks:
            continue
        instr_f = task_dir / "baseline" / "instruction.md"
        ref_f = task_dir / "shared" / "image_ref.txt"
        if not instr_f.exists() or not ref_f.exists():
            continue
        done += 1
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

        for seed in range(args.n_runs):
            run_id = f"{task_id}-s{seed}"
            if (out_dir / f"{run_id}.json").exists():
                t_rec["runs"][run_id] = {"status": "already-present (resumed)"}
                continue
            cfg = AgentLoopConfig(max_steps=args.max_steps,
                                  max_new_tokens_per_turn=args.max_new_tokens,
                                  temperature=TEMPS[seed % len(TEMPS)])
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
        (out_root / "collection_manifest.json").write_text(
            json.dumps(manifest, indent=1), encoding="utf-8")

    n_ok = sum(1 for t in manifest["tasks"].values()
               for r in t["runs"].values() if r.get("status") == "ok")
    n_fin = sum(1 for t in manifest["tasks"].values()
                for r in t["runs"].values() if r.get("finished"))
    print(f"\n{done} tasks; {n_ok} runs ok, {n_fin} reached TASK_DONE. "
          f"Manifest: {out_root/'collection_manifest.json'}")
    log_event(events, event="v2_collection_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
