"""A0 collection v1 (AWS box): N runs per task, per-read mid-layer logging.

    python scripts/collect_a0.py --tasks-dir third_party/hil-bench/harbor_swe \
        --n-tasks 20 --out data/a0

v1 scope (decisions/004, AWS runbook): single-generation reasoning traces per
task instruction -- enough for A1 (should-ask direction), A2 training data,
A3 calibration, and gates 1-5/7 on reasoning spans. v2 (full multi-step agent
loop through the HiL-Bench executor) extends the same RunLog schema via
segment_idx; nothing downstream changes.

Diversity is measured, not assumed (decisions/008): the script reports how
many distinct final answers each task produced across its N runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wta.hf_reader import HFStreamReader  # noqa: E402
from wta.logging_schema import save_run_log  # noqa: E402

TEMPS = (0.7, 0.85, 1.0)  # cycled over seeds (decisions/008)

PROMPT_TEMPLATE = """You are a coding agent. Work on this task:

{instruction}

Think through how you would implement this step by step, reasoning about any
choices the task leaves open, then state the exact change you would make."""


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
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--n-runs", type=int, default=8)
    ap.add_argument("--mid-layer", type=float, default=0.5)
    ap.add_argument("--cadence", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--out", default="data/a0")
    args = ap.parse_args()

    reader = HFStreamReader(args.model_id, mid_layer=args.mid_layer,
                            cadence=args.cadence)
    diversity = {}
    for task_id, instruction in iter_tasks(Path(args.tasks_dir), args.n_tasks):
        out_dir = Path(args.out) / task_id
        answers = set()
        for seed in range(args.n_runs):
            run_id = f"{task_id}-s{seed}"
            if (out_dir / f"{run_id}.json").exists():
                continue  # resumable
            log, text = reader.run(
                PROMPT_TEMPLATE.format(instruction=instruction.strip()),
                run_id=run_id, task_id=task_id, seed=seed,
                temperature=TEMPS[seed % len(TEMPS)],
                max_new_tokens=args.max_new_tokens,
            )
            save_run_log(log, out_dir)
            (out_dir / f"{run_id}.txt").write_text(text, encoding="utf-8")
            answers.add(text.strip()[-200:])  # crude answer signature
            print(f"{run_id}: {len(log.reads)} reads")
        diversity[task_id] = len(answers)
        print(f"{task_id}: {diversity[task_id]} distinct answer signatures / {args.n_runs} runs")

    report = Path(args.out) / "diversity_report.json"
    report.write_text(json.dumps(diversity, indent=1), encoding="utf-8")
    frac_diverse = sum(v >= 2 for v in diversity.values()) / max(len(diversity), 1)
    print(f"\nDiversity: {frac_diverse:.0%} of tasks produced >= 2 distinct answers "
          f"(escalation rule fires below 30% -- decisions/008)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
