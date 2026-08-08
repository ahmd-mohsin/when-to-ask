"""Contract: TurnStepper reproduces run_agent turn-for-turn and supports
injection (spec eval, decisions/022).

Docker-free and model-free, same fake pattern as test_agent_loop.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from wta.agent_loop import AgentLoopConfig, parse_action, run_agent
from wta.eval.fullinfo import augment_full_info
from wta.eval.patch import extract_patch, prediction_row
from wta.eval.stepper import (
    ASK_AFFORDANCE_PARAGRAPH, StepOutcome, TurnStepper, parse_ask,
)
from wta.logging_schema import ReadRecord

H = 16


class FakeSession:
    def __init__(self, turns):
        self.turns = turns
        self.calls = []          # snapshots of the messages list per call

    def generate_segment(self, messages, *, seed, temperature, max_new_tokens,
                         segment_idx):
        self.calls.append([dict(m) for m in messages])
        text = self.turns[segment_idx]
        rng = np.random.default_rng(seed * 10 + segment_idx)
        reads = [ReadRecord(token_idx=t, trigger="cadence", cue=None,
                            h=rng.standard_normal(H).astype(np.float16),
                            segment_idx=segment_idx) for t in (7, 15)]
        return reads, text


class FakeEnv:
    def __init__(self, outputs=None):
        self.commands = []
        self.outputs = outputs or {}

    def execute(self, cmd):
        self.commands.append(cmd)
        if cmd in self.outputs:
            return self.outputs[cmd]
        return 0, f"ok output for: {cmd[:30]}"


TURNS = [
    "THOUGHT: look around first.\n```bash\nls lib/ansible/module_utils\n```",
    "THOUGHT: edit the file.\n```bash\nsed -i 's/a/b/' lib/x.py\n```",
    "THOUGHT: all good, submitting.\n```bash\necho TASK_DONE\n```",
]


def _drive(stepper):
    outs = []
    while not stepper.done():
        outs.append(stepper.step())
    return outs


def test_equivalence_with_run_agent():
    cfg = AgentLoopConfig(max_steps=10)
    ref = run_agent(FakeSession(TURNS), FakeEnv(), "Fix the bug.", run_id="r0",
                    task_id="t0", seed=0, cfg=cfg, model_id="fake", mid_layer=4)

    env = FakeEnv()
    st = TurnStepper(FakeSession(TURNS), env, "Fix the bug.", run_id="r0",
                     task_id="t0", seed=0, cfg=cfg, model_id="fake", mid_layer=4)
    _drive(st)
    res = st.result()

    assert res.finished == ref.finished and res.stop_reason == ref.stop_reason
    assert res.n_steps == ref.n_steps and res.segments == ref.segments
    assert res.commands == ref.commands
    assert env.commands == [parse_action(TURNS[0]), parse_action(TURNS[1])]
    assert [(r.segment_idx, r.token_idx) for r in res.log.reads] == \
           [(r.segment_idx, r.token_idx) for r in ref.log.reads]
    assert len(res.log.actions) == len(ref.log.actions)
    for a, b in zip(res.log.actions, ref.log.actions):
        assert a.action_text == b.action_text
        assert a.observables == b.observables


def test_transcript_matches_run_agent_exactly():
    cfg = AgentLoopConfig(max_steps=10)
    ses_ref, ses_new = FakeSession(TURNS), FakeSession(TURNS)
    run_agent(ses_ref, FakeEnv(), "Fix the bug.", run_id="r0", task_id="t0",
              seed=0, cfg=cfg, model_id="fake", mid_layer=4)
    st = TurnStepper(ses_new, FakeEnv(), "Fix the bug.", run_id="r0",
                     task_id="t0", seed=0, cfg=cfg, model_id="fake", mid_layer=4)
    _drive(st)
    assert ses_new.calls == ses_ref.calls


def test_no_block_nag_and_max_steps():
    cfg = AgentLoopConfig(max_steps=4)
    env = FakeEnv()
    st = TurnStepper(FakeSession(["no block here"] * 4), env, "Task.",
                     run_id="r1", task_id="t0", seed=0, cfg=cfg,
                     model_id="fake", mid_layer=4)
    outs = _drive(st)
    res = st.result()
    assert not res.finished and res.stop_reason == "max_steps"
    assert env.commands == [] and res.log.actions == []
    assert all(o.command is None for o in outs)


def test_injected_message_lands_once_before_next_turn():
    cfg = AgentLoopConfig(max_steps=10)
    ses = FakeSession(TURNS)
    st = TurnStepper(ses, FakeEnv(), "Fix the bug.", run_id="r0", task_id="t0",
                     seed=0, cfg=cfg, model_id="fake", mid_layer=4)
    st.step()
    st.inject_user_message("[HUMAN ANSWER] Use interpretation 0.")
    st.step()
    joined = [m["content"] for m in ses.calls[1] if m["role"] == "user"]
    assert sum("HUMAN ANSWER" in c for c in joined) == 1
    # merged into the trailing user turn -> strict role alternation preserved
    roles = [m["role"] for m in ses.calls[1]]
    assert all(a != b for a, b in zip(roles, roles[1:]))
    # exactly once ever: not re-sent on the following turn
    st.step()
    all_users = "".join(m["content"] for m in ses.calls[2] if m["role"] == "user")
    assert all_users.count("HUMAN ANSWER") == 1


ASK_TURNS = [
    'THOUGHT: ambiguous spec.\n```bash\nask_human "Which timeout should the retry use?"\n```',
    "THOUGHT: got it, submitting.\n```bash\necho TASK_DONE\n```",
]


def test_ask_interception_and_answer_flow():
    cfg = AgentLoopConfig(max_steps=10)
    env = FakeEnv()
    st = TurnStepper(FakeSession(ASK_TURNS), env, "Task.", run_id="r0",
                     task_id="t0", seed=0, cfg=cfg, ask_affordance=True,
                     model_id="fake", mid_layer=4)
    assert ASK_AFFORDANCE_PARAGRAPH.strip() in st.messages[0]["content"]

    out = st.step()
    assert out.asked == "Which timeout should the retry use?"
    assert env.commands == []                     # never reached the env
    with pytest.raises(RuntimeError, match="unanswered ask"):
        st.step()
    obs = st.answer_ask("Use 30 seconds.")
    assert "Use 30 seconds." in obs
    st.step()
    res = st.result()
    assert res.finished
    # the ask is a logged action whose error_signature came from the answer
    ask_events = [a for a in res.log.actions if a.action_text.startswith("ask_human")]
    assert len(ask_events) == 1
    assert ask_events[0].observables["error_signature"].startswith("exit 0")


def test_affordance_off_ask_goes_to_env():
    cfg = AgentLoopConfig(max_steps=2)
    env = FakeEnv()
    st = TurnStepper(FakeSession(ASK_TURNS), env, "Task.", run_id="r0",
                     task_id="t0", seed=0, cfg=cfg, ask_affordance=False,
                     model_id="fake", mid_layer=4)
    out = st.step()
    assert out.asked is None
    assert env.commands == [parse_action(ASK_TURNS[0])]
    assert ASK_AFFORDANCE_PARAGRAPH.strip() not in st.messages[0]["content"]


def test_parse_ask_forms():
    assert parse_ask('ask_human "a question?"') == "a question?"
    assert parse_ask("ask_human 'single quotes?'") == "single quotes?"
    assert parse_ask("  ask_human \"padded?\"  ") == "padded?"
    assert parse_ask("grep ask_human file.py") is None
    assert parse_ask("ask_human unquoted") is None


def test_full_info_augmentation_matches_jinja_render():
    # expected string pinned from the vendored template rendered with the
    # default jinja2 Environment (verified 2026-08-07 @352d14c)
    blockers = [{"description": "D1\n", "resolution": "R1\n"},
                {"description": "D2", "resolution": "R2"}]
    expected = ("STMT\n\n---\n\n## Additional Context\n\n"
                "The following clarifications are provided to help you "
                "complete this task:\n\n"
                "\n### D1\n\n\nR1\n\n\n"
                "\n### D2\n\nR2\n\n")
    assert augment_full_info("STMT", blockers) == expected
    assert augment_full_info("STMT", []) == "STMT"

    class B:  # xtid Blocker-shaped
        description, resolution = "D2", "R2"

    assert "### D2\n\nR2\n\n" in augment_full_info("STMT", [B()])


def test_patch_extraction_and_prediction_row():
    env = FakeEnv(outputs={
        "git add -A": (0, ""),
        "git diff --cached HEAD": (0, "diff --git a/x.py b/x.py\n+fixed\n"),
    })
    patch = extract_patch(env)
    assert patch.startswith("diff --git")
    assert env.commands[0] == "git config core.fileMode false"
    row = prediction_row("swe_0", patch, "qwen3-32b")
    assert set(row) == {"instance_id", "model_name_or_path", "model_patch"}

    env_fail = FakeEnv(outputs={"git add -A": (128, "not a git repo")})
    assert extract_patch(env_fail) == ""
