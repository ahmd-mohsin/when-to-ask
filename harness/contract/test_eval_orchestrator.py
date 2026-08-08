"""Contract: lockstep N-run orchestrator (spec eval, decisions/022).

Fake session/env/judge throughout. Pins: lockstep round order; policy asks
judged once and injected into ALL runs before their next turn; model-initiated
asks answered as the asking run's observation; committed-trajectory rule
(incl. the no-finisher case); AskSession per-blocker cap honored; stub
statements refused; full-info augmentation applied.
"""

from __future__ import annotations

import numpy as np
import pytest

from wta.agent_loop import AgentLoopConfig
from wta.eval.orchestrator import (
    ArmSpec, TaskPassResult, committed_run, run_task_pass,
)
from wta.eval.policies import AskRequest, TriggerPolicy
from wta.logging_schema import ReadRecord
from xtid.harness.ask_f1 import AskSession
from xtid.harness.judge import IRRELEVANT_QUESTION, MockJudge
from xtid.harness.tasks import Blocker, Task

H = 8


class ScriptedSession:
    """Per-run scripted turns keyed by (seed, segment_idx); records call order."""

    def __init__(self, scripts):
        self.scripts = scripts          # seed -> list of turn texts
        self.call_order = []            # (seed, segment_idx)
        self.seen_messages = {}         # seed -> last messages snapshot

    def generate_segment(self, messages, *, seed, temperature, max_new_tokens,
                         segment_idx):
        self.call_order.append((seed, segment_idx))
        self.seen_messages[seed] = [dict(m) for m in messages]
        turns = self.scripts[seed]
        text = turns[min(segment_idx, len(turns) - 1)]
        reads = [ReadRecord(token_idx=7, trigger="cadence", cue=None,
                            h=np.zeros(H, dtype=np.float16),
                            segment_idx=segment_idx)]
        return reads, text


class FakeEnv:
    def __init__(self):
        self.commands = []

    def execute(self, cmd):
        self.commands.append(cmd)
        if cmd == "git diff --cached HEAD":
            return 0, "diff --git a/f b/f\n+x\n"
        return 0, "ok"


def _task(statement="Fix the retry timeout.", stub=False):
    return Task(
        instance_id="swe_t", domain="swe", statement=statement,
        source="hil_bench",
        blockers=[Blocker(id="retry_timeout_duration",
                          description="Retry timeout ambiguous 30 vs 60.",
                          resolution="Use 30 seconds.",
                          example_questions=["- Which timeout duration should "
                                             "the retry logic use?"],
                          type="ambiguous requirements")],
        meta={"statement_stub": stub})


DONE = "THOUGHT: done.\n```bash\necho TASK_DONE\n```"
WORK = "THOUGHT: work.\n```bash\nls\n```"


def _run(arm, task=None, scripts=None, judge=None, envs=None):
    envs = envs if envs is not None else {}

    def env_factory(run_id):
        envs[run_id] = FakeEnv()
        return envs[run_id]

    ses = ScriptedSession(scripts or {0: [WORK, DONE], 1: [WORK, WORK, DONE]})
    ask = AskSession()
    res = run_task_pass(arm, task or _task(), session=ses,
                        env_factory=env_factory, judge=judge or MockJudge(),
                        ask_session=ask, cfg=AgentLoopConfig(max_steps=6),
                        pass_idx=0, seed_base=0, model_id="fake")
    return res, ses, ask, envs


def test_lockstep_round_robin_order():
    arm = ArmSpec("no_ask", n_runs=2, temperatures=(0.7, 0.9))
    res, ses, _, _ = _run(arm, scripts={0: [WORK, DONE], 1: [WORK, WORK, DONE]})
    # round 1: seeds 0,1 ; round 2: 0,1 ; round 3: only 1 (0 finished)
    assert ses.call_order == [(0, 0), (1, 0), (0, 1), (1, 1), (1, 2)]
    assert res.compute["rounds"] == 3 and res.compute["n_runs"] == 2


def test_policy_ask_injected_into_all_runs_before_next_turn():
    class AskOncePolicy(TriggerPolicy):
        name = "test_ask"

        def __init__(self):
            super().__init__()
            self._round = 0

        def poll(self):
            self._round += 1
            if self._round == 1 and self._mark_asked("k1"):
                return AskRequest(
                    "Which timeout duration should the retry logic use?",
                    key="k1")
            return None

    arm = ArmSpec("detector", n_runs=2, temperatures=(0.7,),
                  policy_factory=AskOncePolicy)
    res, ses, ask, _ = _run(arm, scripts={0: [WORK, WORK, DONE],
                                          1: [WORK, WORK, WORK, DONE]})
    # the answer (resolution, via MockJudge match) reached BOTH runs' turn-2
    for seed in (0, 1):
        users = [m["content"] for m in ses.seen_messages[seed]
                 if m["role"] == "user"]
        joined = "\n".join(users)
        assert "HUMAN ANSWER" in joined and "Use 30 seconds." in joined
    (ev,) = res.ask_events
    assert ev["blocker_id"] == "retry_timeout_duration"
    assert ev["source"] == "test_ask" and ev["run_id"] is None
    m = ask.metrics()
    assert m.n_questions == 1 and m.n_blockers_discovered == 1


def test_model_initiated_answered_as_observation_of_asking_run_only():
    ask_turn = ('THOUGHT: unsure.\n```bash\nask_human "Which timeout duration '
                'should the retry logic use?"\n```')
    arm = ArmSpec("model_initiated", n_runs=1, ask_affordance=True,
                  temperatures=(1.0,))
    res, ses, ask, envs = _run(arm, scripts={0: [ask_turn, DONE]})
    (ev,) = res.ask_events
    assert ev["source"] == "model" and ev["run_id"] == "swe_t-p0-s0"
    assert ev["response"] == "Use 30 seconds."
    users = [m["content"] for m in ses.seen_messages[0] if m["role"] == "user"]
    assert any("Use 30 seconds." in u for u in users)
    assert not any("HUMAN ANSWER" in u for u in users)   # observation, not injection
    assert envs["swe_t-p0-s0"].commands[0] != ev["question"]  # never executed


def test_committed_trajectory_rule():
    # finished run with lowest seed wins
    results = {"a-s0": type("R", (), {"finished": False})(),
               "a-s1": type("R", (), {"finished": True})(),
               "a-s2": type("R", (), {"finished": True})()}
    seeds = {"a-s0": 0, "a-s1": 1, "a-s2": 2}
    assert committed_run(results, seeds) == "a-s1"
    # nobody finished -> lowest seed
    for r in results.values():
        r.finished = False
    assert committed_run(results, seeds) == "a-s0"


def test_prediction_from_committed_run_and_patch():
    arm = ArmSpec("no_ask", n_runs=2, temperatures=(1.0,))
    res, _, _, envs = _run(arm, scripts={0: [WORK, DONE], 1: [WORK, WORK, DONE]})
    assert res.committed_run_id == "swe_t-p0-s0"
    assert res.prediction["instance_id"] == "swe_t"
    assert res.prediction["model_patch"].startswith("diff --git")
    # patch was extracted from the COMMITTED run's env
    assert "git diff --cached HEAD" in envs["swe_t-p0-s0"].commands


def test_ask_session_cap_forces_third_hit_irrelevant():
    class SpamPolicy(TriggerPolicy):
        name = "spam"

        def __init__(self):
            super().__init__()
            self._n = 0

        def poll(self):
            if self._n < 3 and self._mark_asked(f"k{self._n}"):
                self._n += 1
                return AskRequest("Which timeout duration should the retry "
                                  "logic use?", key=f"k{self._n}")
            return None

    arm = ArmSpec("detector", n_runs=1, temperatures=(1.0,),
                  policy_factory=SpamPolicy)
    res, _, ask, _ = _run(arm, scripts={0: [WORK, WORK, WORK, WORK, DONE]})
    responses = [e["response"] for e in res.ask_events]
    assert responses.count("Use 30 seconds.") == 2          # upstream cap
    assert responses.count(IRRELEVANT_QUESTION) == 1
    m = ask.metrics()
    assert m.n_questions == 3 and m.n_blockers_discovered == 1


def test_stub_statement_refused_and_full_info_augmented():
    arm = ArmSpec("no_ask", n_runs=1, temperatures=(1.0,))
    with pytest.raises(ValueError, match="stub"):
        _run(arm, task=_task(stub=True))

    arm_fi = ArmSpec("full_info", n_runs=1, full_info=True, temperatures=(1.0,))
    _, ses, _, _ = _run(arm_fi, scripts={0: [DONE]})
    instr = ses.seen_messages[0][1]["content"]
    assert "## Additional Context" in instr and "Use 30 seconds." in instr

    arm_plain = ArmSpec("no_ask", n_runs=1, temperatures=(1.0,))
    _, ses2, _, _ = _run(arm_plain, scripts={0: [DONE]})
    assert "Additional Context" not in ses2.seen_messages[0][1]["content"]
