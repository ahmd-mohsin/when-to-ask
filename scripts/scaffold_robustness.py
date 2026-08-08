"""Scaffold-robustness check (decisions/021 §8 item 4; spec eval-bridge).

    python scripts/scaffold_robustness.py --traj results/bridge \
        --our data/eval --out results/scaffold_robustness.json

Claim tested: fork structure is not an artifact of our one-bash-block
scaffold. Parses the bridge rows' SWE-Agent ``.traj`` trajectories, runs OUR
composite-observable extractors (``wta.agent_loop``) over each step's
command + thought, and reports per task, side by side with the same
statistics from our-loop trajectories on the same tasks:

  * steps and distinct composite junctures (file+region+error signature),
  * cross-pass juncture overlap (Jaccard),
  * cross-pass behavioural fork signatures: junctures visited by 2+ passes
    with DIFFERENT subgoal prefixes -- divergent readings at a shared point.

Behavioural only -- bridge runs go through vLLM, no activations exist on
that side (stated scope limit, spec eval-bridge).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wta.agent_loop import (  # noqa: E402
    extract_file_observables, extract_region, extract_subgoal,
)
from xtid.harness.passk import extract_public_trajectory_steps  # noqa: E402


def steps_from_traj_file(traj_path: Path) -> list[dict]:
    payload = json.loads(traj_path.read_text(encoding="utf-8"))
    return extract_public_trajectory_steps(payload)


def steps_from_our_pass(pass_dir: Path, run_id: str) -> list[dict]:
    """Adapt an our-loop run (RunLog json + segments) to the same step shape."""
    log = json.loads((pass_dir / f"{run_id}.json").read_text(encoding="utf-8"))
    segs_f = pass_dir / f"{run_id}.segments.json"
    segments = json.loads(segs_f.read_text(encoding="utf-8")) if segs_f.exists() else []
    steps = []
    for a in log.get("actions", []):
        seg = a.get("segment_idx", 0)
        steps.append({"act": a.get("action_text", ""),
                      "thought": segments[seg] if seg < len(segments) else "",
                      "obs": ""})
    return steps


def junctures(steps: list[dict]) -> dict[str, list[str]]:
    """composite juncture key -> subgoal prefixes seen there.

    Key = files + region (the composite label's stable half); the subgoal is
    the divergence-bearing half (what reading the model verbalized)."""
    out: dict[str, list[str]] = defaultdict(list)
    for step in steps:
        act = step.get("act", "") or ""
        if not act.strip():
            continue
        files = extract_file_observables(act)
        if not files:
            continue
        key = "|".join(files) + "@" + ",".join(extract_region(act))
        thought = step.get("thought", "") or step.get("response", "") or ""
        out[key].append(extract_subgoal(thought, limit=120).lower())
    return dict(out)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def task_stats(passes: dict[str, list[dict]]) -> dict:
    """passes: pass_label -> steps. Cross-pass juncture stats."""
    per_pass = {label: junctures(steps) for label, steps in passes.items()}
    keysets = {label: set(j) for label, j in per_pass.items()}
    labels = sorted(keysets)
    overlaps = [_jaccard(keysets[a], keysets[b])
                for i, a in enumerate(labels) for b in labels[i + 1:]]
    # shared junctures where 2+ passes verbalized different subgoals
    fork_signatures = []
    all_keys = set().union(*keysets.values()) if keysets else set()
    for key in all_keys:
        readings = {}
        for label, j in per_pass.items():
            if key in j and j[key]:
                readings[label] = j[key][0][:80]
        if len(readings) >= 2 and len(set(readings.values())) >= 2:
            fork_signatures.append({"juncture": key, "readings": readings})
    return {
        "n_passes": len(passes),
        "steps_per_pass": {la: len(st) for la, st in passes.items()},
        "junctures_per_pass": {la: len(ks) for la, ks in keysets.items()},
        "mean_cross_pass_overlap": (sum(overlaps) / len(overlaps)
                                    if overlaps else 0.0),
        "n_fork_signatures": len(fork_signatures),
        "fork_signatures": fork_signatures[:10],
    }


def collect_bridge(traj_root: Path) -> dict[str, dict[str, list[dict]]]:
    """task -> pass_label -> steps, from every *.traj under the root. The
    task id is recovered from the instance id embedded in the path."""
    by_task: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for traj in sorted(traj_root.rglob("*.traj")):
        steps = steps_from_traj_file(traj)
        if not steps:
            continue
        stem = traj.parent.name          # SWE-Agent: <instance_id>/<id>.traj
        task = stem.split("__")[0]
        by_task[task][stem] = steps
    return dict(by_task)


def collect_ours(eval_root: Path, arm: str = "no_ask") -> dict[str, dict[str, list[dict]]]:
    by_task: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for committed in sorted(eval_root.glob(f"*/{arm}/pass_*/committed.json")):
        pass_dir = committed.parent
        task = pass_dir.parent.parent.name
        info = json.loads(committed.read_text(encoding="utf-8"))
        run_id = info.get("committed_run_id")
        if run_id and (pass_dir / f"{run_id}.json").exists():
            by_task[task][pass_dir.name] = steps_from_our_pass(pass_dir, run_id)
    return dict(by_task)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True,
                    help="bridge output root (searched for *.traj)")
    ap.add_argument("--our", default=None,
                    help="run_eval.py output root (comparison side)")
    ap.add_argument("--our-arm", default="no_ask")
    ap.add_argument("--out", default="results/scaffold_robustness.json")
    args = ap.parse_args()

    report = {"bridge": {}, "ours": {}}
    for task, passes in collect_bridge(Path(args.traj)).items():
        report["bridge"][task] = task_stats(passes)
    if args.our:
        for task, passes in collect_ours(Path(args.our), args.our_arm).items():
            report["ours"][task] = task_stats(passes)

    both = sorted(set(report["bridge"]) & set(report["ours"]))
    report["comparison"] = {
        t: {"bridge_forks": report["bridge"][t]["n_fork_signatures"],
            "our_forks": report["ours"][t]["n_fork_signatures"],
            "bridge_overlap": report["bridge"][t]["mean_cross_pass_overlap"],
            "our_overlap": report["ours"][t]["mean_cross_pass_overlap"]}
        for t in both}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    n_b = len(report["bridge"])
    n_forks = sum(s["n_fork_signatures"] for s in report["bridge"].values())
    print(f"bridge: {n_b} tasks, {n_forks} cross-pass fork signatures; "
          f"comparison rows: {len(both)}\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
