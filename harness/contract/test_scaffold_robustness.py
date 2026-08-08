"""Contract: scaffold-robustness analyzer (spec eval-bridge, decisions/022;
decisions/021 §8 item 4).

Builds SWE-Agent-shaped .traj fixtures in tmp and pins: .traj parsing via the
ported extract_public_trajectory_steps; juncture extraction through OUR
composite extractors; cross-pass fork-signature detection (divergent subgoals
at a shared file+region juncture); the ours-side adapter over run_eval.py's
on-disk layout; and the comparison table shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from scaffold_robustness import (  # noqa: E402
    collect_bridge, collect_ours, junctures, steps_from_traj_file, task_stats,
)


def _traj_payload(steps):
    return {"trajectory": [
        {"action": act, "observation": "ok", "thought": thought}
        for act, thought in steps]}


def _write_traj(root: Path, task: str, pass_label: str, steps):
    d = root / f"{task}__qwen__baseline__{pass_label}"
    d.mkdir(parents=True)
    (d / "run.traj").write_text(json.dumps(_traj_payload(steps)),
                                encoding="utf-8")


def test_traj_parsing_and_junctures(tmp_path):
    _write_traj(tmp_path, "swe_0", "pass_0", [
        ("ls", "look around"),
        ("sed -i '5,9s/a/b/' lib/x.py", "keep the shim in place"),
    ])
    traj = next(tmp_path.rglob("*.traj"))
    steps = steps_from_traj_file(traj)
    assert len(steps) == 2 and steps[1]["act"].startswith("sed")
    j = junctures(steps)
    (key,) = j.keys()                        # ls has no file observable
    assert key == "lib/x.py@5-9"
    assert j[key] == ["keep the shim in place"]


def test_cross_pass_fork_signature_detected(tmp_path):
    # two passes touch the SAME file+region with DIFFERENT verbalized readings
    _write_traj(tmp_path, "swe_0", "pass_0", [
        ("sed -i '5,9s/a/b/' lib/x.py", "keep the compatibility shim"),
    ])
    _write_traj(tmp_path, "swe_0", "pass_1", [
        ("sed -i '5,9s/a/c/' lib/x.py", "drop the shim, use stdlib"),
    ])
    _write_traj(tmp_path, "swe_0", "pass_2", [
        ("sed -i '5,9s/a/b/' lib/x.py", "keep the compatibility shim"),
    ])
    by_task = collect_bridge(tmp_path)
    assert set(by_task) == {"swe_0"} and len(by_task["swe_0"]) == 3
    stats = task_stats(by_task["swe_0"])
    assert stats["n_passes"] == 3
    assert stats["n_fork_signatures"] == 1
    (sig,) = stats["fork_signatures"]
    assert sig["juncture"] == "lib/x.py@5-9"
    assert len(set(sig["readings"].values())) == 2
    assert stats["mean_cross_pass_overlap"] == 1.0   # same juncture everywhere


def test_no_fork_when_readings_agree(tmp_path):
    for p in range(2):
        _write_traj(tmp_path, "swe_1", f"pass_{p}", [
            ("sed -i '5,9s/a/b/' lib/x.py", "keep the compatibility shim"),
        ])
    stats = task_stats(collect_bridge(tmp_path)["swe_1"])
    assert stats["n_fork_signatures"] == 0


def test_ours_side_adapter(tmp_path):
    pass_dir = tmp_path / "swe_0" / "no_ask" / "pass_0"
    pass_dir.mkdir(parents=True)
    run_id = "swe_0-p0-s0"
    (pass_dir / "committed.json").write_text(json.dumps(
        {"committed_run_id": run_id}))
    (pass_dir / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id,
        "actions": [{"action_text": "sed -i '5,9s/a/b/' lib/x.py",
                     "segment_idx": 0, "token_idx": 5, "observables": {}}]}))
    (pass_dir / f"{run_id}.segments.json").write_text(json.dumps(
        ["THOUGHT: keep the shim.\n```bash\nsed -i '5,9s/a/b/' lib/x.py\n```"]))
    by_task = collect_ours(tmp_path, arm="no_ask")
    assert set(by_task) == {"swe_0"}
    j = junctures(by_task["swe_0"]["pass_0"])
    assert "lib/x.py@5-9" in j
    assert j["lib/x.py@5-9"] == ["thought: keep the shim."]
