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

def test_seq_by_run_decision_generation_order(tmp_path):
    """v2 token_idx restarts per segment: sequences fed to A3/gate7 must be in
    generation (row) order, NOT sorted by token_idx, which interleaves turns."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_full_gates import seq_by_run_decision
    from wta.labeling import build_labels
    from wta.logging_schema import RunLog

    anchors_text = ("the retry policy question: for transient errors the "
                    "backoff decision matters here. ")
    art = {"taskX": {"blockerA": {
        "anchors": ["retry policy", "transient errors", "backoff decision"],
        "classes": [
            {"name": "canonical", "canonical": True, "signatures": ["retry_all()"]},
            {"name": "alt", "signatures": ["retry_transient()"]},
        ]}}}
    art_f = tmp_path / "classes.json"
    art_f.write_text(json.dumps(art), encoding="utf-8")

    seg0 = anchors_text * 6                                   # no commitment yet
    seg1 = (anchors_text + "I'll commit to retry_all(). ") * 4
    task_dir = tmp_path / "a0" / "taskX"
    task_dir.mkdir(parents=True)

    # generation order 0,1,2,3 stamped into h[:,0]; token-sorted order would
    # scramble to [2, 0, 3, 1] because segment 1 restarts token_idx
    reads = []
    for gen_i, (seg, tok) in enumerate([(0, 31), (0, 63), (1, 5), (1, 40)]):
        h = np.zeros(H, dtype=np.float16)
        h[0], h[1] = gen_i, 1.0
        reads.append(ReadRecord(token_idx=tok, trigger="cadence", cue=None,
                                h=h, segment_idx=seg))
    log = RunLog(run_id="taskX-s0", task_id="taskX", seed=0, temperature=0.7,
                 model_id="fake", mid_layer=4, reads=reads)
    save_run_log(log, task_dir)
    (task_dir / "taskX-s0.segments.json").write_text(json.dumps([seg0, seg1]),
                                                     encoding="utf-8")
    (task_dir / "taskX-s0.txt").write_text("\n\n".join([seg0, seg1]),
                                           encoding="utf-8")

    ds = build_labels(tmp_path / "a0", art_f, window_chars=150)
    assert (ds.decision == 0).sum() == 4, "fixture: all reads must be labeled"

    class IdModel:
        def encode_lean(self, x):
            return x

    groups = seq_by_run_decision(ds, np.ones(len(ds.h), bool), IdModel())
    (r_seq, _cls, _run) = groups[0][0]
    assert [float(v) for v in r_seq[:, 0]] == [0.0, 1.0, 2.0, 3.0]


def test_real_action_reads_maps_first_matching_action(tmp_path):
    """real_action_reads: the FIRST ActionEvent containing the committed
    class's signature, located against the (run, decision) read sequence by
    (segment_idx, token_idx); unmatched commitment -> -1."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_full_gates import real_action_reads
    from wta.labeling import build_labels
    from wta.logging_schema import ActionEvent, RunLog

    anchors_text = ("the retry policy question: for transient errors the "
                    "backoff decision matters here. ")
    art = {"taskX": {"blockerA": {
        "anchors": ["retry policy", "transient errors", "backoff decision"],
        "classes": [
            {"name": "canonical", "canonical": True, "signatures": ["retry_all()"]},
            {"name": "alt", "signatures": ["retry_transient()"]},
        ]}}}
    art_f = tmp_path / "classes.json"
    art_f.write_text(json.dumps(art), encoding="utf-8")

    seg0 = anchors_text * 6
    seg1 = (anchors_text + "I'll commit to retry_all(). ") * 4
    task_dir = tmp_path / "a0" / "taskX"
    task_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    reads = [ReadRecord(token_idx=t, trigger="cadence", cue=None,
                        h=rng.standard_normal(H).astype(np.float16),
                        segment_idx=s)
             for s, t in [(0, 31), (0, 63), (1, 5), (1, 40)]]
    # the matching edit happens in segment 1 after token 5 and before 40:
    # reads (0,31), (0,63), (1,5) precede it -> action_read index 2
    actions = [
        ActionEvent(token_idx=90, action_text="cat notes.txt", segment_idx=0),
        ActionEvent(token_idx=20, action_text="echo 'retry_all()' >> impl.py",
                    segment_idx=1),
        ActionEvent(token_idx=60, action_text="echo 'retry_all()' >> impl.py",
                    segment_idx=1),
    ]
    log = RunLog(run_id="taskX-s0", task_id="taskX", seed=0, temperature=0.7,
                 model_id="fake", mid_layer=4, reads=reads, actions=actions)
    save_run_log(log, task_dir)
    (task_dir / "taskX-s0.segments.json").write_text(json.dumps([seg0, seg1]),
                                                     encoding="utf-8")
    (task_dir / "taskX-s0.txt").write_text("\n\n".join([seg0, seg1]),
                                           encoding="utf-8")

    ds = build_labels(tmp_path / "a0", art_f, window_chars=150)
    ar = real_action_reads(ds, tmp_path / "a0", art_f)
    assert ar == {(0, 0): 2}


def _value_fork_fixture(tmp_path, actions):
    """Two-segment run where PROSE favors class A (timeout = 30, mentioned
    3x while deliberating) but the given actions may write class B
    (timeout = 60). Returns (a0_dir, artifact_path, debug_path)."""
    from wta.logging_schema import RunLog

    anchors_text = ("the request timeout policy question: which timeout "
                    "duration applies to the retry request here. ")
    art = {"taskX": {"blockerA": {
        "anchors": ["timeout policy", "timeout duration", "retry request"],
        "classes": [
            {"name": "thirty", "canonical": True, "signatures": ["timeout = 30"]},
            {"name": "sixty", "signatures": ["timeout = 60"]},
        ]}}}
    art_f = tmp_path / "classes.json"
    art_f.write_text(json.dumps(art), encoding="utf-8")

    seg0 = (anchors_text
            + "maybe timeout = 30? docs hint timeout = 30; or timeout = 60. "
            + "I lean towards timeout = 30 but let me check. " + anchors_text)
    seg1 = (anchors_text + "decided; applying the edit now.\n```bash\n"
            + (actions[0].action_text if actions else "ls") + "\n```\n"
            + anchors_text)
    task_dir = tmp_path / "a0" / "taskX"
    task_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    reads = [ReadRecord(token_idx=t, trigger="cadence", cue=None,
                        h=rng.standard_normal(H).astype(np.float16),
                        segment_idx=s)
             for s, t in [(0, 9), (0, 30), (1, 9), (1, 30)]]
    log = RunLog(run_id="taskX-s0", task_id="taskX", seed=0, temperature=0.7,
                 model_id="fake", mid_layer=4, reads=reads, actions=actions)
    save_run_log(log, task_dir)
    (task_dir / "taskX-s0.segments.json").write_text(json.dumps([seg0, seg1]),
                                                     encoding="utf-8")
    (task_dir / "taskX-s0.txt").write_text("\n\n".join([seg0, seg1]),
                                           encoding="utf-8")
    return tmp_path / "a0", art_f, tmp_path / "labels_debug.jsonl"


def _commitments(debug_path):
    out = []
    for line in debug_path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("kind") == "commitment":
            out.append(e)
    return out


def test_mutating_action_overrides_prose_mentions(tmp_path):
    """spec labels.md v2: the written edit (timeout = 60) must win over the
    deliberation mentions (timeout = 30 x3), and reads BEFORE the edit stay
    should_ask even though the winning value was mentioned earlier."""
    from wta.labeling import build_labels
    from wta.logging_schema import ActionEvent

    a0, art_f, dbg = _value_fork_fixture(tmp_path, [
        ActionEvent(token_idx=0, action_text="sed -i 's/x/timeout = 60/' cfg.py",
                    segment_idx=1),
    ])
    ds = build_labels(a0, art_f, window_chars=150, debug_path=dbg)

    (c,) = _commitments(dbg)
    assert c["chosen"] == "sixty" and c["label_source"] == "actions"
    # segment-0 reads (deliberation) precede the action -> should_ask
    assert ds.phase[0] == 0 and ds.phase[1] == 0
    # segment-1 reads follow the action -> settled with the WRITTEN class
    assert ds.phase[2] == 1 and ds.phase[3] == 1
    assert ds.cls[2] >= 0 and ds.vocab.classes[ds.cls[2]][2] == "sixty"


def test_readonly_action_does_not_commit(tmp_path):
    """A grep containing a class signature is exploration, not commitment ->
    falls back to whole-trace scoring (prose favors thirty)."""
    from wta.labeling import build_labels
    from wta.logging_schema import ActionEvent

    a0, art_f, dbg = _value_fork_fixture(tmp_path, [
        ActionEvent(token_idx=0, action_text="grep -rn 'timeout = 60' cfg.py",
                    segment_idx=1),
    ])
    build_labels(a0, art_f, window_chars=150, debug_path=dbg)

    (c,) = _commitments(dbg)
    assert c["chosen"] == "thirty" and c["label_source"] == "trace"
