"""Contract: judge-agreement comparison logic + frozen pairs file (spec eval,
decisions/022; brief §5a). The live two-path comparison runs on the GPU box;
here two fake judges exercise the comparison core.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from validate_judge import PAIRS_FILE, compare, load_pairs  # noqa: E402


def test_pairs_file_frozen_and_well_formed():
    assert PAIRS_FILE.exists(), "data/judge_validation_pairs.json missing"
    pairs = load_pairs()
    assert len(pairs) >= 30
    kinds = {p["kind"] for p in pairs}
    assert {"positive", "form_reject"} <= kinds
    for p in pairs:
        assert p["question"].strip()
        assert isinstance(p["blockers"], list) and p["blockers"]
        for b in p["blockers"]:
            assert b["id"] and "resolution" in b
        if p["kind"] == "positive":
            assert p["expected"] in {b["id"] for b in p["blockers"]}
        else:
            assert p["expected"] is None
    # positives dominate but negatives are represented
    n_pos = sum(1 for p in pairs if p["kind"] == "positive")
    assert n_pos >= 15 and len(pairs) - n_pos >= 8


def test_compare_full_agreement():
    pairs = [{"case_id": "c1", "kind": "positive", "question": "Which x?",
              "blockers": [{"id": "b1", "resolution": "r"}], "expected": "b1"},
             {"case_id": "c2", "kind": "negative", "question": "Statement.",
              "blockers": [{"id": "b1", "resolution": "r"}], "expected": None}]

    def oracle(question, blockers):
        return "b1" if question.endswith("?") else None

    rep = compare(pairs, oracle, oracle)
    assert rep["agreement"] == 1.0
    assert rep["ours_accuracy"] == 1.0 and rep["theirs_accuracy"] == 1.0
    assert rep["disagreements"] == []


def test_compare_surfaces_disagreements_verbatim():
    pairs = [{"case_id": "c1", "kind": "positive", "question": "Which x?",
              "blockers": [{"id": "b1", "resolution": "r"}], "expected": "b1"}]
    rep = compare(pairs, lambda q, b: "b1", lambda q, b: None)
    assert rep["agreement"] == 0.0
    (d,) = rep["disagreements"]
    assert d["ours"] == "b1" and d["theirs"] is None
    assert d["question"] == "Which x?"
    assert rep["ours_accuracy"] == 1.0 and rep["theirs_accuracy"] == 0.0


def test_mockjudge_passes_the_hand_written_pairs():
    """Sanity: the migrated rubric approximation agrees with the frozen
    expectations on the hand-written form cases (the LLM-judge cases are
    GPU-time)."""
    from xtid.harness.judge import MockJudge
    from xtid.harness.tasks import Blocker

    judge = MockJudge()

    def ask(question, blocker_dicts):
        blockers = [Blocker(id=b["id"], description=b.get("description", ""),
                            resolution=b.get("resolution", ""),
                            example_questions=list(b.get("example_questions", [])),
                            type=b.get("type")) for b in blocker_dicts]
        return judge.ask(question, blockers).blocker_id

    # MockJudge approximates the rubric's mechanical form rules (statement,
    # multi-topic, assumption-confirmation) but NOT the semantic
    # definition-request rule -- that one is real-judge-only (GPU-time).
    hand = [p for p in load_pairs() if p["case_id"].startswith("hand_")
            and p["kind"] in ("positive", "form_reject")
            and "definition" not in p["case_id"]]
    assert hand
    rep = compare(hand, ask, ask)
    for row in rep["rows"]:
        if row["kind"] == "form_reject":
            assert row["ours"] is None, row["case_id"]
