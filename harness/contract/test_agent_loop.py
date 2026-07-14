"""Contract: v2 agent loop + segment-aware labeling (decisions/017).

Docker-free and model-free: a scripted FakeSession produces the turns and
reads; a FakeEnv records executed commands. The labeling test builds a real
two-segment run on disk and checks reads are labeled through the RIGHT
segment's text."""

import json

import numpy as np
import pytest

from wta.agent_loop import (
    AgentLoopConfig, extract_file_observables, parse_action, run_agent,
    truncate_obs,
)
from wta.logging_schema import ReadRecord, save_run_log

H = 16


class FakeSession:
    """Yields scripted turn texts; emits 2 cadence reads per turn."""

    def __init__(self, turns):
        self.turns = turns

    def generate_segment(self, messages, *, seed, temperature, max_new_tokens,
                         segment_idx):
        text = self.turns[segment_idx]
        rng = np.random.default_rng(seed * 10 + segment_idx)
        reads = [ReadRecord(token_idx=t, trigger="cadence", cue=None,
                            h=rng.standard_normal(H).astype(np.float16),
                            segment_idx=segment_idx) for t in (7, 15)]
        return reads, text


class FakeEnv:
    def __init__(self):
        self.commands = []

    def execute(self, cmd):
        self.commands.append(cmd)
        return 0, f"ok output for: {cmd[:30]}"


TURNS = [
    "THOUGHT: look around first.\n```bash\nls lib/ansible/module_utils\n```",
    "THOUGHT: edit the file.\n```bash\nsed -i 's/a/b/' lib/ansible/module_utils/common/sys_info.py\n```",
    "THOUGHT: all good, submitting.\n```bash\necho TASK_DONE\n```",
]


def test_agent_loop_end_to_end():
    env = FakeEnv()
    res = run_agent(FakeSession(TURNS), env, "Fix the bug.", run_id="r0",
                    task_id="t0", seed=0, cfg=AgentLoopConfig(max_steps=10),
                    model_id="fake", mid_layer=4, layers=None)
    assert res.finished and res.stop_reason == "submit_marker"
    assert res.n_steps == 3 and len(res.segments) == 3
    # marker command is logged as an action but NOT executed
    assert env.commands == [parse_action(TURNS[0]), parse_action(TURNS[1])]
    # reads carry per-segment indices; ordering valid
    assert [(r.segment_idx, r.token_idx) for r in res.log.reads] == [
        (0, 7), (0, 15), (1, 7), (1, 15), (2, 7), (2, 15)]
    res.log.validate()
    # actions: one per turn, with file observables from the command
    assert len(res.log.actions) == 3
    assert res.log.actions[1].observables["files"] == [
        "lib/ansible/module_utils/common/sys_info.py"]
    assert res.log.actions[1].segment_idx == 1


def test_agent_loop_handles_missing_block_and_max_steps():
    env = FakeEnv()
    res = run_agent(FakeSession(["no block here"] * 4), env, "Task.",
                    run_id="r1", task_id="t0", seed=0,
                    cfg=AgentLoopConfig(max_steps=4), model_id="fake",
                    mid_layer=4)
    assert not res.finished and res.stop_reason == "max_steps"
    assert env.commands == [] and res.log.actions == []
    assert res.n_steps == 4  # kept prompting, never crashed


def test_parse_and_observables_and_truncation():
    assert parse_action("x\n```bash\nls\n```\n") == "ls"
    assert parse_action("```bash\na\n```\n...\n```bash\nb\n```") == "b"  # last wins
    assert parse_action("no block") is None
    assert extract_file_observables("cat a/b.py && vim c.go") == ["a/b.py", "c.go"]
    long = "A" * 5000
    t = truncate_obs(long, 1500, 500)
    assert len(t) < 2200 and t.startswith("A" * 100) and t.endswith("A" * 100)
    assert "truncated" in t


def test_segment_aware_labeling(tmp_path):
    """Reads in segment 1 must be labeled via segment 1's text, not segment 0's."""
    from wta.labeling import build_labels

    # artifact: one decision; anchors appear ONLY in segment 1
    art = {"taskX": {"blockerA": {
        "anchors": ["retry policy", "transient errors", "backoff decision"],
        "classes": [
            {"name": "canonical", "canonical": True, "signatures": ["retry_all()"]},
            {"name": "alt", "signatures": ["retry_transient()"]},
        ]}}}
    art_f = tmp_path / "classes.json"
    art_f.write_text(json.dumps(art), encoding="utf-8")

    seg0 = "Let me look at the repository structure first. " * 8  # no anchors
    seg1 = ("Now the retry policy question: for transient errors the backoff "
            "decision matters. I'll commit to retry_all() here. " * 4)
    task_dir = tmp_path / "a0" / "taskX"
    task_dir.mkdir(parents=True)

    from wta.logging_schema import RunLog
    rng = np.random.default_rng(0)
    reads = ([ReadRecord(token_idx=t, trigger="cadence", cue=None,
                         h=rng.standard_normal(H).astype(np.float16),
                         segment_idx=0) for t in (5, 25)]
             + [ReadRecord(token_idx=t, trigger="cadence", cue=None,
                           h=rng.standard_normal(H).astype(np.float16),
                           segment_idx=1) for t in (5, 40)])
    log = RunLog(run_id="taskX-s0", task_id="taskX", seed=0, temperature=0.7,
                 model_id="fake", mid_layer=4, reads=reads)
    save_run_log(log, task_dir)
    (task_dir / "taskX-s0.segments.json").write_text(json.dumps([seg0, seg1]),
                                                     encoding="utf-8")
    (task_dir / "taskX-s0.txt").write_text("\n\n".join([seg0, seg1]),
                                           encoding="utf-8")

    ds = build_labels(tmp_path / "a0", art_f, window_chars=150)
    assert len(ds.h) == 4
    # segment-0 reads: no anchors near them -> unlabeled decision
    assert ds.decision[0] == -1 and ds.decision[1] == -1
    # segment-1 reads: anchors surround them -> labeled with the decision,
    # and the run committed to the canonical class (signature in seg1)
    assert ds.decision[2] == 0 and ds.decision[3] == 0
    assert (ds.cls[2] >= 0) or (ds.cls[3] >= 0)