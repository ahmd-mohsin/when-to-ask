"""Contract: headline scoring (spec eval, decisions/022 §3).

Pins: pass-record collection from run_eval.py's on-disk layout; Ask-F1
recompute with TRUE registry counts incl. the zero-question regression
(upstream's compute_zero_hil_metrics bug zeroes the recall denominator);
bridge ask-log recompute grouped by embedded mode; regime slicing with the
frozen structural/value annotations; table assembly with a fake resolver.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wta.eval.scoring import (
    PassRecord, ask_metrics_by_arm, bridge_ask_metrics, budget_by_arm,
    collect_our_results, our_rows, predictions_by_arm, regime_recall_by_arm,
    score, split_expanded_id,
)
from xtid.harness.passk import TRAJECTORY_HICCUP_OBS
from xtid.harness.tasks import Blocker

B1 = Blocker(id="b_timeout", description="timeout ambiguous", resolution="30",
             type="ambiguous requirements")
B2 = Blocker(id="b_format", description="format unspecified", resolution="json",
             type="missing parameters")


def _write_pass(root: Path, task, arm, p, *, questions=(), n_blockers=2,
                finished=True, stats=None, patch="diff --git a/x b/x\n"):
    d = root / task / arm / f"pass_{p}"
    d.mkdir(parents=True)
    iid = f"inst_{task}"
    (d / "committed.json").write_text(json.dumps({
        "committed_run_id": f"{task}-p{p}-s0",
        "seeds": {f"{task}-p{p}-s0": 0},
        "finished": {f"{task}-p{p}-s0": finished},
        "prediction": {"instance_id": iid, "model_name_or_path": "ours",
                       "model_patch": patch}}))
    key = f"{iid}__{arm}__pass_{p}"
    (d / "ask_log.json").write_text(json.dumps({
        "events": [], "stats": stats or {"asks": len(questions),
                                         "suppressed_refires": 0,
                                         "fired_keys": []},
        "instance_key": key,
        "session": {key: {"n_blockers": n_blockers,
                          "questions": list(questions)}}}))
    (d / "compute.json").write_text(json.dumps(
        {"n_runs": 4, "rounds": 3, "turns": 9, "generated_chars": 900,
         "asks": len(questions), "phrasing_calls": 1}))


def _q(blocker, response="30"):
    return {"question": "Which timeout duration should the retry use?",
            "response": response, "blocker_name": blocker}


@pytest.fixture()
def eval_tree(tmp_path):
    root = tmp_path / "eval"
    for p in range(3):
        _write_pass(root, "swe_60", "detector", p,
                    questions=[_q("b_timeout")])
        _write_pass(root, "swe_60", "no_ask", p)
    _write_pass(root, "swe_61", "detector", 0, questions=[])   # zero-question
    return root


def test_collect_and_predictions(eval_tree):
    records = collect_our_results(eval_tree)
    assert len(records) == 7
    r = next(r for r in records if r.task == "swe_60" and r.arm == "detector"
             and r.pass_idx == 1)
    assert r.expanded_id == "inst_swe_60__ours__detector__pass_1"
    assert r.n_blockers == 2 and len(r.questions) == 1
    preds = predictions_by_arm(records)
    assert set(preds) == {"detector", "no_ask"}
    assert "inst_swe_60__ours__detector__pass_2" in preds["detector"]


def test_zero_question_pass_keeps_true_denominator(eval_tree):
    """The upstream regression: a zero-question ask pass must still count its
    blockers in the recall denominator."""
    records = [r for r in collect_our_results(eval_tree) if r.arm == "detector"]
    m = ask_metrics_by_arm(records, true_counts={"swe_60": 2, "swe_61": 2})["detector"]
    # 4 detector passes x 2 blockers each = 8 present, incl. the zero-question
    assert m.n_blockers_present == 8
    assert m.n_questions == 3 and m.n_blockers_discovered == 3
    assert m.precision == 1.0 and m.recall == pytest.approx(3 / 8)


def test_bridge_ask_metrics_grouped_by_mode_with_true_counts():
    logs = {
        "inst_swe_60__qwen__ask_human__pass_0": {
            "n_blockers": 2, "questions": [_q("b_timeout")]},
        # upstream bug shape: zero-question pass recorded with n_blockers 0
        "inst_swe_60__qwen__ask_human__pass_1": {
            "n_blockers": 0, "questions": []},
        "inst_swe_60__qwen__baseline__pass_0": {
            "n_blockers": 2, "questions": []},
    }
    out = bridge_ask_metrics(logs, {"inst_swe_60": 2})
    ask = out["ask_human"]
    assert ask.n_blockers_present == 4          # true counts, bug bypassed
    assert ask.recall == pytest.approx(1 / 4)
    assert out["baseline"].n_questions == 0


def test_split_expanded_id():
    assert split_expanded_id("a__m__ask_human__pass_2") == ("a", "m",
                                                           "ask_human", 2)
    assert split_expanded_id("swe_60") == ("swe_60", "", "", 0)
    # original ids containing __ split from the right
    assert split_expanded_id("a__b__m__full_info__pass_0")[0] == "a__b"


def test_regime_recall_with_frozen_annotations(eval_tree):
    records = collect_our_results(eval_tree)
    blockers = {"swe_60": [B1, B2], "swe_61": [B1]}
    ann = {"b_timeout": "structural", "b_format": "value"}
    out = regime_recall_by_arm(records, blockers, ann)
    det = out["detector"]
    # b_timeout discovered on swe_60 (not swe_61), b_format never
    assert det["fork_structural"] == {"discovered": 1, "present": 2,
                                      "recall": 0.5}
    assert det["fork_value"]["recall"] == 0.0
    assert out["no_ask"]["fork_structural"]["discovered"] == 0
    # without annotations everything is fork_unannotated
    out2 = regime_recall_by_arm(records, blockers, None)
    assert set(out2["detector"]) == {"fork_unannotated"}


def test_our_rows_trajectory_carries_ask_responses(eval_tree):
    records = collect_our_results(eval_tree)
    hiccup_rec = next(r for r in records if r.arm == "detector"
                      and r.pass_idx == 0 and r.task == "swe_60")
    hiccup_rec.questions[0]["response"] = TRAJECTORY_HICCUP_OBS
    rows = our_rows(records, resolved={})
    bad = next(r for r in rows if r["mode"] == "detector"
               and r["pass_num"] == 0 and r["task_name"] == "swe_60")
    assert bad["trajectory"][0]["obs"] == TRAJECTORY_HICCUP_OBS


def test_score_end_to_end_with_fake_resolver(eval_tree):
    records = collect_our_results(eval_tree)
    # fake resolver output: detector solves swe_60 on pass 1 only
    resolved = {"inst_swe_60__ours__detector__pass_1": True}
    blockers = {"swe_60": [B1, B2], "swe_61": [B1]}
    out = score(records, resolved, blockers,
                {"b_timeout": "structural", "b_format": "value"},
                expected_passes=3, include_partial=True,
                bridge_ask_logs={
                    "inst_swe_60__qwen__ask_human__pass_0": {
                        "n_blockers": 0, "questions": [_q("b_timeout")]}})
    det = out["pass_summary"]["detector"]["ours"]
    assert det["pass_at_1"] == 0.0 and det["pass_at_2"] == 1.0
    assert out["ask_by_arm"]["detector"]["ask_f1"] > 0
    assert out["bridge_ask"]["ask_human"]["recall"] == pytest.approx(0.5)
    md = out["markdown"]
    assert "| ours | detector |" in md.replace("|  ", "| ") or "detector" in md
    rows = out["csv_rows"]
    assert {r["source"] for r in rows} == {"ours", "bridge"}
    assert any(r["arm"] == "detector" and r["source"] == "ours" for r in rows)
    # budgets & compute flowed through
    assert out["budget_by_arm"]["detector"]["asks_per_pass"] == pytest.approx(3 / 4)
    assert out["compute_by_arm"]["detector"]["n_runs"] == 4
