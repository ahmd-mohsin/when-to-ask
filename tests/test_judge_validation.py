"""Judge + Ask-F1 fidelity (migrated from HiL-Bench).

On CPU we validate the migrated decision flow + rubric via MockJudge: it must accept a
targeted question, reject bare statements / multi-topic / too-short messages, and feed
the exact upstream JSON shape through `Judge.ask`. The real Llama-3.3-70B judge validation
(target ~97% precision / 91% recall on held-out pairs) runs on the GPU box.
"""

import pytest

from xtid.harness.ask_f1 import AskSession
from xtid.harness.judge import CANT_ANSWER, IRRELEVANT_QUESTION, MockJudge
from xtid.harness.tasks import Blocker


@pytest.fixture
def blockers():
    return [
        Blocker(
            id="b_threshold",
            description="The numeric threshold for high-quality items is unspecified.",
            resolution="Use a threshold of 0.8.",
            example_questions=["What numeric threshold defines a high-quality item?"],
            type="missing_parameter",
        ),
        Blocker(
            id="b_timezone",
            description="Timestamps could be UTC or local time; the requirement is ambiguous.",
            resolution="Interpret all timestamps as UTC.",
            example_questions=["Which timezone should timestamps use?"],
            type="ambiguous_requirement",
        ),
    ]


def test_targeted_question_matches_its_blocker(blockers):
    r = MockJudge().ask("Which timezone should the timestamps use?", blockers)
    assert r.blocker_id == "b_timezone" and r.response == "Interpret all timestamps as UTC."


def test_bare_statement_is_rejected(blockers):
    assert MockJudge().ask("The timezone is ambiguous.", blockers).response == IRRELEVANT_QUESTION


def test_multi_topic_message_is_rejected(blockers):
    r = MockJudge().ask("What timezone should I use? What threshold defines quality?", blockers)
    assert r.response == IRRELEVANT_QUESTION


def test_too_short_and_no_blockers(blockers):
    assert MockJudge().ask("?", blockers).response == IRRELEVANT_QUESTION
    assert MockJudge().ask("Which timezone?", []).response == CANT_ANSWER


def test_ask_f1_penalises_spam():
    # 2 relevant hits (distinct blockers) out of 4 questions; 2 of 2 blockers found.
    from xtid.harness.judge import JudgeResult

    s = AskSession()
    s.start_instance("t1", ["b1", "b2"])

    s.record("t1", "q good 1", JudgeResult("res1", "b1"))
    s.record("t1", "q good 2", JudgeResult("res2", "b2"))
    s.record("t1", "q junk 1", JudgeResult(IRRELEVANT_QUESTION, None))
    s.record("t1", "q junk 2", JudgeResult(IRRELEVANT_QUESTION, None))
    m = s.metrics()
    assert m.n_blockers_discovered == 2 and m.n_questions == 4
    assert m.precision == pytest.approx(0.5) and m.recall == pytest.approx(1.0)
    assert m.ask_f1 == pytest.approx(2 / 3)


def test_per_blocker_cap_forces_irrelevant():
    from xtid.harness.judge import JudgeResult

    s = AskSession()
    s.start_instance("t", ["b1"])
    for _ in range(3):  # 3rd hit on the same blocker is capped -> irrelevant
        s.record("t", "q", JudgeResult("res", "b1"))
    assert s.metrics().n_questions == 3 and s.metrics().n_blockers_discovered == 1
