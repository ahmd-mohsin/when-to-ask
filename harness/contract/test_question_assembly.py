"""Contract: assembled questions pass the frozen judge's form rubric and the
phrasing prompt is registry-free (spec eval, decisions/022 §2f).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from wta.eval.question import (
    AssembledQuestion, _valid_question, assemble_question,
)
from wta.online import AskDecision
from xtid.harness.judge import IRRELEVANT_QUESTION, MockJudge
from xtid.harness.tasks import Blocker


def _decision(actions):
    return AskDecision(
        bucket_id=0,
        runs=[f"r{i}" for i in range(len(actions))],
        options=[{"run_id": f"r{i}", "r": np.zeros(4), "weight": 1.0 - 0.1 * i,
                  "action_text": a} for i, a in enumerate(actions)],
        looping_runs=[], spread=1.0, cusum=7.0)


class PhrasingSession:
    """Returns a canned phrased question; records the prompt it saw."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate_segment(self, messages, *, seed, temperature, max_new_tokens,
                         segment_idx):
        self.prompts.append(messages[0]["content"])
        return [], self.reply


TIMEOUT_BLOCKER = Blocker(
    id="retry_timeout_duration",
    description="The retry timeout duration is ambiguous between 30 and 60 seconds.",
    resolution="Use 30 seconds.",
    example_questions=["- Which timeout duration should the retry logic use?"],
    type="ambiguous requirements")

OTHER_BLOCKER = Blocker(
    id="log_format", description="The log line format is unspecified.",
    resolution="Use JSON lines.",
    example_questions=["- What format should the log lines use?"],
    type="missing parameters")


def test_llm_phrased_question_matches_planted_blocker_via_mockjudge():
    ses = PhrasingSession("Which timeout duration should the retry logic use?")
    dec = _decision(["sed -i 's/timeout = x/timeout = 30/' cfg.py",
                     "sed -i 's/timeout = x/timeout = 60/' cfg.py"])
    q = assemble_question(dec, "Implement retry with a sensible timeout.",
                          session=ses, seed=3)
    assert q.method == "llm"
    res = MockJudge().ask(q.text, [TIMEOUT_BLOCKER, OTHER_BLOCKER])
    assert res.blocker_id == "retry_timeout_duration"
    assert res.response == "Use 30 seconds."


def test_template_fallback_is_form_valid_for_the_judge():
    dec = _decision(["sed -i 's/x/retry_backoff(30)/' handler.py",
                     "sed -i 's/x/retry_backoff(60)/' handler.py"])
    q = assemble_question(dec, "task", session=None)
    assert q.method == "template"
    assert _valid_question(q.text)
    # MockJudge accepts the FORM (may or may not match a blocker)
    res = MockJudge().ask(q.text, [OTHER_BLOCKER])
    assert res.response in (OTHER_BLOCKER.resolution, IRRELEVANT_QUESTION)


def test_bad_generation_falls_back_to_template():
    for reply in ("", "I think option A is right.",   # no question
                  "What about X? And what about Y? Or Z?"):  # multi-?
        ses = PhrasingSession(reply)
        q = assemble_question(_decision(["a", "b"]), "task", session=ses)
        assert q.method == "template"
        assert _valid_question(q.text)


def test_generation_exception_falls_back():
    class Boom:
        def generate_segment(self, *a, **k):
            raise RuntimeError("cuda oom")

    q = assemble_question(_decision(["a"]), "task", session=Boom())
    assert q.method == "template" and _valid_question(q.text)


def test_prompt_contains_only_statement_and_options():
    ses = PhrasingSession("Which timeout duration should the retry use?")
    dec = _decision(["cmd A uses thirty", "cmd B uses sixty"])
    q = assemble_question(dec, "STATEMENT-TEXT retry timeout", session=ses)
    prompt = ses.prompts[0]
    assert "STATEMENT-TEXT" in prompt
    assert "cmd A uses thirty" in prompt and "cmd B uses sixty" in prompt
    assert q.prompt == prompt
    # structural leak rule: no registry data can even be passed in
    params = inspect.signature(assemble_question).parameters
    assert "blockers" not in params and "registry" not in params
    for leak in (TIMEOUT_BLOCKER.resolution, TIMEOUT_BLOCKER.description,
                 TIMEOUT_BLOCKER.example_questions[0]):
        assert leak not in prompt


def test_options_deduped_capped_and_ordered_by_weight():
    dec = _decision(["same", "same", "b", "c", "d", "e"])
    q = assemble_question(dec, "task", session=None)
    assert isinstance(q, AssembledQuestion)
    assert len(q.options_used) <= 4
    assert q.options_used[0] == "same" and q.options_used.count("same") == 1
