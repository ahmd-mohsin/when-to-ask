#!/usr/bin/env python
"""De-risk experiment entry point (brief S7).

Runs N trajectories per task, records every signal at each aligned decision point, trains
the B3 probe out-of-fold, and prints the C1 table: per-regime AUROC for internal vs.
output (B1) vs. probe (B3), plus internal-vs-output lead-time on the fork slice.

  python scripts/run_derisk.py --config configs/smoke.yaml    # CPU, fake model + mock judge
  python scripts/run_derisk.py --config configs/derisk.yaml   # GPU, real backbone + API judge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtid.agent.multi import run_tasks
from xtid.analysis.lead_time import lead_time_analysis
from xtid.analysis.separation import c1_verdict, separation_table
from xtid.backbone.model import build_model
from xtid.harness.executor import build_executor
from xtid.harness.judge import build_judge
from xtid.harness.tasks import load_tasks
from xtid.recording.recorder import build_records, save_records


def _fmt(x) -> str:
    return "  n/a" if x is None else f"{x:5.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    exp = cfg.get("experiment", {})
    seed = exp.get("seed", 0)
    n_traj = exp.get("n_trajectories", 5)

    print(f"== xtid de-risk: {exp.get('name', '?')} (N={n_traj}, seed={seed}) ==")
    model = build_model(cfg["backbone"])
    build_judge(cfg.get("judge", {"kind": "mock"}))  # constructed/validated; used in closed loop
    tasks_cfg = dict(cfg.get("tasks", {}))
    tasks_cfg.setdefault("seed", seed)
    tasks = load_tasks(tasks_cfg)
    executor = build_executor(tasks_cfg.get("source", "synthetic"))
    print(f"   backbone={cfg['backbone'].get('kind')} hidden_dim={model.hidden_dim} "
          f"mid_layer={model.mid_layer} | tasks={len(tasks)} ({tasks_cfg.get('source')})")

    agent_cfg = cfg.get("agent", {})
    trajs = run_tasks(
        model, tasks, n_trajectories=n_traj,
        temperature=agent_cfg.get("temperature", 0.8),
        diverse_prompting=agent_cfg.get("diverse_prompting", False),
    )
    run = build_records(
        trajs, tasks, executor,
        scheme=cfg.get("signals", {}).get("alignment", "step_index_then_anchor"),
        seed=seed, config=cfg,
    )
    print(f"   decision points recorded: {len(run.records)}")

    table = separation_table(run)
    print("\n-- Separation AUROC (higher = better should-ask vs proceed) --")
    print(f"{'signal':34s} {'overall':>8} {'fork|clr':>9} {'cwrong|clr':>11}")
    for sig, row in table.items():
        marker = "  <- internal (primary)" if sig.endswith("mean_pairwise_cosine") else (
            "  <- B1 output" if sig == "output_divergence_b1" else (
                "  <- B3 probe" if sig == "probe_b3" else ""))
        print(f"{sig:34s} {_fmt(row['overall']):>8} {_fmt(row['fork_vs_clear']):>9} "
              f"{_fmt(row['confident_wrong_vs_clear']):>11}{marker}")

    v = c1_verdict(table)
    lt = lead_time_analysis(run)
    print("\n-- C1 verdict (fork slice) --")
    print(f"   internal fork AUROC = {_fmt(v['internal_fork_auroc'])}   "
          f"output(B1) fork AUROC = {_fmt(v['output_b1_fork_auroc'])}   "
          f"=> internal beats B1 on fork: {v['internal_beats_b1_on_fork']}")
    print(f"   blind spot: probe(B3) confident-wrong AUROC = {_fmt(v['probe_confident_wrong_auroc'])} "
          f"vs internal {_fmt(v['internal_confident_wrong_auroc'])} "
          f"=> probe covers blind spot: {v['probe_covers_blind_spot']}")
    print("\n-- Lead-time (internal vs output, fork tasks) --")
    print(f"   median lead = {lt['median_lead']}  fraction positive = {_fmt(lt['fraction_positive_lead'])}  "
          f"(both-cross={lt['n_both_cross']}, internal-only={lt['internal_only_count']}, "
          f"neither={lt['neither_count']}, fork tasks={lt['n_fork_tasks']})")

    out_dir = exp.get("out_dir", "logs/derisk")
    save_records(run, out_dir)
    print(f"\n   records written to {out_dir}/records.json")


if __name__ == "__main__":
    main()
