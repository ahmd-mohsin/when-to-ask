"""A0 collection v1.5 (AWS box): N grounded runs per task, per-read logging.

    python scripts/extract_task_context.py --n-tasks 20     # once, first
    python scripts/collect_a0.py --n-tasks 20

v1.5 over v1 (ADR 012): repo-grounded prompts (pre-patch source files via
--context-dir; leak analysis in extract_task_context.py), longer generations
(A3 needs sequences that settle), a mild deliberation nudge (recorded), a
code-block answer signature, and diagnostics designed so that when a number
looks wrong later we can trace exactly what produced it:

  data/a0/collection_manifest.json   args, versions, GPU, prompt hash + grounding
                                     mode per task, per-run timing/finish reason
  data/a0/<task>/prompt.txt          the exact prompt used (one per task)
  data/a0/<task>/<run>.{npz,json,txt} as before (schema unchanged)
  data/a0/events.jsonl               append-only event log (start/finish/error
                                     per run with timestamps and token counts)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.collect_utils import (  # noqa: E402
    answer_signature, build_prompt, load_context_files, sha256_text,
)
from wta.hf_reader import HFStreamReader  # noqa: E402
from wta.logging_schema import save_run_log  # noqa: E402

TEMPS = (0.7, 0.85, 1.0)  # cycled over seeds (decisions/008)


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        info["repo_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:
        info["repo_commit"] = None
    return info


def log_event(path: Path, **kw) -> None:
    kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")


def iter_tasks(tasks_dir: Path, limit: int):
    count = 0
    for task_dir in sorted(tasks_dir.iterdir()):
        instr = task_dir / "baseline" / "instruction.md"
        if not instr.exists():
            continue
        yield task_dir.name, instr.read_text(encoding="utf-8", errors="replace")
        count += 1
        if count >= limit:
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="third_party/hil-bench/harbor_swe")
    ap.add_argument("--context-dir", default="data/task_context",
                    help="output of extract_task_context.py; missing task dirs "
                         "fall back to instruction-only (recorded)")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--n-runs", type=int, default=8)
    ap.add_argument("--mid-layer", type=float, default=0.5)
    ap.add_argument("--cadence", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--no-nudge", action="store_true")
    ap.add_argument("--layers", default="0.4,0.5,0.6,0.7",
                    help="comma list of mid-layer specs (fraction of depth or "
                         "explicit index) captured per read for the layer sweep "
                         "(decisions/014); '' or 'none' -> single --mid-layer")
    ap.add_argument("--value-reads", action="store_true",
                    help="also read the moment a multi-digit literal is emitted "
                         "(decisions/016 value-fork experiment; cooldown-limited)")
    ap.add_argument("--out", default="data/a0")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    events = out_root / "events.jsonl"

    layer_specs = None
    if args.layers and args.layers.lower() != "none":
        layer_specs = [float(x) if "." in x else int(x)
                       for x in args.layers.split(",")]
    from wta.reads import DEFAULT_VALUE_PATTERN

    reader = HFStreamReader(args.model_id, mid_layer=args.mid_layer,
                            layers=layer_specs, cadence=args.cadence,
                            value_pattern=(DEFAULT_VALUE_PATTERN
                                           if args.value_reads else None))
    manifest = {
        "args": vars(args), "env": env_info(),
        "reader": {"n_layers": reader.n_layers, "hidden_dim": reader.hidden_dim,
                   "mid_layer": reader.mid_layer,
                   "layer_indices": reader.layer_indices,
                   "layer_specs": layer_specs},
        "tasks": {},
    }
    log_event(events, event="collection_start", args=vars(args))

    for task_id, instruction in iter_tasks(Path(args.tasks_dir), args.n_tasks):
        out_dir = out_root / task_id
        out_dir.mkdir(parents=True, exist_ok=True)

        ctx = load_context_files(Path(args.context_dir) / task_id)
        prompt = build_prompt(instruction, ctx, nudge=not args.no_nudge)
        (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        t_rec = manifest["tasks"].setdefault(task_id, {
            "grounding": "context" if ctx else "instruction-only",
            "context_files": [p for p, _ in ctx],
            "prompt_sha": sha256_text(prompt), "prompt_chars": len(prompt),
            "runs": {},
        })

        signatures = set()
        for seed in range(args.n_runs):
            run_id = f"{task_id}-s{seed}"
            if (out_dir / f"{run_id}.json").exists():
                t_rec["runs"][run_id] = {"status": "already-present (resumed)"}
                # Count the resumed run's signature too, else a task whose runs
                # were ALL collected before a stop reports 0 distinct signatures
                # (a reporting artifact, not low diversity -- seen for swe_0 in
                # the 2026-07-07 full collection).
                tf = out_dir / f"{run_id}.txt"
                if tf.exists():
                    signatures.add(answer_signature(
                        tf.read_text(encoding="utf-8", errors="replace")))
                continue
            temperature = TEMPS[seed % len(TEMPS)]
            log_event(events, event="run_start", run=run_id, temp=temperature)
            t0 = time.time()
            try:
                log, text = reader.run(prompt, run_id=run_id, task_id=task_id,
                                       seed=seed, temperature=temperature,
                                       max_new_tokens=args.max_new_tokens)
            except Exception as e:
                log_event(events, event="run_error", run=run_id,
                          error=f"{type(e).__name__}: {e}")
                t_rec["runs"][run_id] = {"status": f"ERROR: {type(e).__name__}: {e}"}
                print(f"{run_id}: ERROR {e}")
                continue
            dt = time.time() - t0
            save_run_log(log, out_dir)
            (out_dir / f"{run_id}.txt").write_text(text, encoding="utf-8")

            n_tok = (log.reads[-1].token_idx + 1) if log.reads else 0
            finish = "length" if n_tok >= args.max_new_tokens - args.cadence else "eos"
            cues = sum(1 for r in log.reads if r.trigger == "cue")
            sig = answer_signature(text)
            signatures.add(sig)
            t_rec["runs"][run_id] = {
                "status": "ok", "reads": len(log.reads), "cue_reads": cues,
                "approx_gen_tokens": n_tok, "finish_reason": finish,
                "seconds": round(dt, 1), "tok_per_s": round(n_tok / max(dt, 1e-9), 1),
                "text_chars": len(text), "answer_sig": sig,
                "text_sha": sha256_text(text),
            }
            log_event(events, event="run_done", run=run_id, reads=len(log.reads),
                      cues=cues, secs=round(dt, 1), finish=finish)
            print(f"{run_id}: {len(log.reads)} reads ({cues} cue), "
                  f"{n_tok} tok in {dt:.0f}s [{finish}]")

        t_rec["distinct_answer_signatures"] = len(signatures)
        print(f"{task_id} [{t_rec['grounding']}]: "
              f"{len(signatures)}/{args.n_runs} distinct answer signatures")
        (out_root / "collection_manifest.json").write_text(
            json.dumps(manifest, indent=1), encoding="utf-8")

    n_div = sum(1 for t in manifest["tasks"].values()
                if t.get("distinct_answer_signatures", 0) >= 2)
    n_grounded = sum(1 for t in manifest["tasks"].values()
                     if t["grounding"] == "context")
    print(f"\n{len(manifest['tasks'])} tasks ({n_grounded} grounded); "
          f"{n_div} with >= 2 distinct signatures "
          f"(escalation below 30% -- decisions/008). Manifest: "
          f"{out_root / 'collection_manifest.json'}")
    log_event(events, event="collection_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
