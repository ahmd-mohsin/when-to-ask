"""Contract: eval arm trigger policies (spec eval, decisions/022).

Per-arm ask behaviour on scripted fake outcomes; dedup once-per-fork; B4
budget matching; the runner's nudge assertion and n_runs null-refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from wta.eval.policies import (  # noqa: E402
    AskRequest, DetectorPolicy, OutputDivergencePolicy, RandomPolicy,
    TaskContext, TriggerPolicy, VerbalizedPolicy,
)
from wta.eval.stepper import StepOutcome  # noqa: E402


def _outcome(command, exit_code=0, obs="ok"):
    return StepOutcome(reads=[], text=f"THOUGHT\n```bash\n{command}\n```",
                       command=command, exit_code=exit_code, observation=obs,
                       finished=False)


def _ctx(**kw):
    defaults = dict(task_id="t0", statement="stmt", steppers={}, session=None,
                    seed=0)
    defaults.update(kw)
    return TaskContext(**defaults)


def test_base_policy_never_asks():
    p = TriggerPolicy()
    p.bind(_ctx())
    p.on_turn("r0", _outcome("ls"))
    assert p.poll() is None and p.stats["asks"] == 0


def test_output_divergence_fires_once_per_signature_set():
    p = OutputDivergencePolicy()
    p.bind(_ctx())
    p.on_turn("r0", _outcome("sed -i 's/x/30/' cfg.py"))
    assert p.poll() is None                      # one run: no divergence
    p.on_turn("r1", _outcome("sed -i 's/x/30/' cfg.py"))
    assert p.poll() is None                      # identical signatures
    p.on_turn("r1", _outcome("sed -i 's/x/60/' cfg.py"))
    req = p.poll()
    assert isinstance(req, AskRequest) and req.key.startswith("b1:")
    assert req.meta["ambiguous"] is True
    assert p.poll() is None                      # same set: deduped
    assert p.stats == {**p.stats, "asks": 1}
    # whitespace-only variation does not re-fire (normalization)
    p.on_turn("r1", _outcome("sed -i  's/x/60/'   cfg.py"))
    assert p.poll() is None
    assert p.stats["suppressed_refires"] >= 1


def test_detector_policy_full_loop(monkeypatch):
    """A fake TaskDetector wires the policy: fire -> ask once -> injection
    resolves the bucket -> re-fire suppressed."""
    from wta.online import AskDecision

    class FakeTaskDetector:
        def __init__(self):
            self.resolved, self.actions, self.envs = [], [], []
            self.fire_next = False

        def observe_read(self, run_id, h):
            if self.fire_next:
                self.fire_next = False
                return AskDecision(bucket_id=3, runs=["r0", "r1"],
                                   options=[{"run_id": "r0", "r": np.zeros(2),
                                             "weight": 1.0,
                                             "action_text": "sed 30"},
                                            {"run_id": "r1", "r": np.ones(2),
                                             "weight": 0.9,
                                             "action_text": "sed 60"}],
                                   looping_runs=[], spread=1.2, cusum=6.5)
            return None

        def register_action(self, run_id, text):
            self.actions.append((run_id, text))

        def notify_env_state(self, run_id, state):
            self.envs.append((run_id, state))

        def inject_resolution(self, bucket_id):
            self.resolved.append(bucket_id)

    class FakeRuntime:
        def __init__(self, td):
            self.td = td

        def new_task(self):
            return self.td

    td = FakeTaskDetector()
    p = DetectorPolicy(FakeRuntime(td))
    p.bind(_ctx())

    class Read:
        h = np.zeros(4, dtype=np.float32)

    p.on_reads("r0", [Read()])
    assert p.poll() is None                      # no fire yet
    td.fire_next = True
    p.on_reads("r0", [Read()])
    req = p.poll()
    assert req.key == "bucket:3" and req.bucket_id == 3
    assert req.meta["spread"] == pytest.approx(1.2)
    assert "?" in req.question                   # template path (session=None)
    p.on_answer(req, None)
    assert td.resolved == [3]
    # bucket already asked: a re-fire is suppressed, not re-asked (022 §2b)
    td.fire_next = True
    p.on_reads("r1", [Read()])
    assert p.poll() is None
    assert p.stats["asks"] == 1 and p.stats["suppressed_refires"] == 1
    # side channels flowed through on_turn
    p.on_turn("r0", _outcome("git diff", exit_code=1, obs="boom"))
    assert td.actions == [("r0", "git diff")]
    assert td.envs and td.envs[0][1].startswith("1:boom")


def test_verbalized_elicits_and_fires_once():
    class ElicitSession:
        def __init__(self, reply):
            self.reply = reply
            self.elicit_calls, self.phrasing_calls = 0, 0

        def generate_segment(self, messages, *, seed, temperature,
                             max_new_tokens, segment_idx):
            if "confident" in messages[-1]["content"]:
                self.elicit_calls += 1
                return [], self.reply
            self.phrasing_calls += 1        # question phrasing on fire
            return [], "Which behaviour should the change implement?"

    class DoneableStepper:
        messages = [{"role": "system", "content": "s"}]

        def done(self):
            return False

    ses = ElicitSession("20")     # low confidence -> score 0.8
    p = VerbalizedPolicy(threshold=0.5, every=2)
    p.bind(_ctx(session=ses, steppers={"r0": DoneableStepper(),
                                       "r1": DoneableStepper()}))
    p.on_turn("r0", _outcome("ls"))
    assert p.poll() is None                      # round 1: not yet (every=2)
    req = p.poll()                               # round 2: elicits, fires
    assert req is not None and req.key == "b2"
    assert ses.elicit_calls == 2                 # one elicitation per live run
    assert ses.phrasing_calls == 1               # LLM-phrased question on fire
    assert req.meta["score"] == pytest.approx(0.8)
    for _ in range(4):
        assert p.poll() is None                  # once per task
    # confident replies never fire
    p2 = VerbalizedPolicy(threshold=0.5, every=1)
    p2.bind(_ctx(session=ElicitSession("95"),
                 steppers={"r0": DoneableStepper()}))
    assert p2.poll() is None


def test_random_policy_budget_matched():
    rounds = 40
    budget = 2.0
    p = RandomPolicy(budget=budget, expected_rounds=rounds, seed=7)
    p.bind(_ctx())
    p.on_turn("r0", _outcome("ls"))
    asks = sum(1 for _ in range(rounds) if p.poll() is not None)
    assert 1 <= asks <= int(np.ceil(budget))     # capped at ceil(budget)
    # deterministic under the same seed
    p2 = RandomPolicy(budget=budget, expected_rounds=rounds, seed=7)
    p2.bind(_ctx())
    p2.on_turn("r0", _outcome("ls"))
    asks2 = sum(1 for _ in range(rounds) if p2.poll() is not None)
    assert asks == asks2


def test_runner_nudge_assertion_and_n_runs_refusal():
    import run_eval

    with pytest.raises(ValueError, match="nudge"):
        run_eval.check_instruction(
            "Task text. Do not ask for clarification; commit to your reading.")
    assert run_eval.check_instruction("Plain task text.") == "Plain task text."

    with pytest.raises(SystemExit, match="pre-registered"):
        run_eval.resolve_n_runs(None, smoke=False)
    assert run_eval.resolve_n_runs(None, smoke=True) == 4
    assert run_eval.resolve_n_runs(8, smoke=False) == 8


def test_task_selector_parsing():
    import run_eval

    assert run_eval.parse_task_selector("swe_60..swe_63") == [
        "swe_60", "swe_61", "swe_62", "swe_63"]
    assert run_eval.parse_task_selector("swe_0,swe_4") == ["swe_0", "swe_4"]
    assert run_eval.parse_task_selector("swe_0, swe_60..swe_61") == [
        "swe_0", "swe_60", "swe_61"]
