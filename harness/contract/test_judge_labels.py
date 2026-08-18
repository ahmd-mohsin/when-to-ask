"""Contract: the judge-labelling arm (spec judge_labels.md, decisions/025
Amendment A) is deterministic, bias-controlled, and evidence-gated.

CPU-only: fabricated mini A0 fixtures, no network, no SDK, no subagents.
The one test that runs build_labels end-to-end needs a locally-cached
tokenizer and skips when none is available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from wta.judge_labels import (  # noqa: E402
    MAX_TRACE_CHARS, SYSTEM_PROMPT, JudgeItem, build_judge_items,
    build_judge_prompt, class_permutation, freeze_results, load_judge_labels,
    locate_evidence, make_excerpt, parse_judge_response, session_responses,
    unlabeled_pairs, write_item_files,
)
from wta.logging_schema import ReadRecord, RunLog, save_run_log  # noqa: E402

RESOLUTION_SECRET = "RESOLUTION_SECRET_MUST_NOT_REACH_THE_JUDGE"


# ---------------------------------------------------------------- fixtures

def _artifact(tmp: Path) -> Path:
    art = {"swe_90": {
        "blk_a": {"anchors": ["frobnicate", "widget mode"],
                  "classes": [
                      {"name": "cls_on", "canonical": True,
                       "signatures": ["enable frobnication"]},
                      {"name": "cls_off",
                       "signatures": ["disable frobnication"]}]},
        "blk_b": {"anchors": ["gadget"],
                  "classes": [
                      {"name": "cls_alpha", "canonical": True,
                       "signatures": ["alpha path"]},
                      {"name": "cls_beta",
                       "signatures": ["beta path"]}]}}}
    p = tmp / "classes.json"
    p.write_text(json.dumps(art), encoding="utf-8")
    return p


TRACE = ("Investigating the widget mode handling. The gadget subsystem is "
         "unclear.\nAfter review we will disable frobnication now to keep "
         "behaviour stable.\nDone.")


def _a0(tmp: Path) -> Path:
    a0 = tmp / "a0"
    task_dir = a0 / "swe_90"
    log = RunLog(run_id="swe_90-s0", task_id="swe_90", seed=0,
                 temperature=0.7, model_id="fake", mid_layer=0,
                 reads=[ReadRecord(token_idx=0, trigger="cadence", cue=None,
                                   h=np.zeros(8, dtype=np.float16))])
    save_run_log(log, task_dir)
    # write_bytes, not write_text: Windows newline translation would put
    # \r\n on disk and shift raw commit_char offsets (labels.md v3.1).
    (task_dir / "swe_90-s0.txt").write_bytes(TRACE.encode("utf-8"))
    return a0


def _registries(tmp: Path) -> Path:
    reg_dir = tmp / "registries"
    d = reg_dir / "swe_90" / "shared" / "ask-human-data"
    d.mkdir(parents=True)
    (d / "blocker_registry.json").write_text(json.dumps({"blockers": [
        {"id": "blk_a", "description": "Frobnication default is ambiguous.",
         "resolution": RESOLUTION_SECRET, "example_questions": []},
        {"id": "blk_b", "description": "Gadget path is unspecified.",
         "resolution": RESOLUTION_SECRET, "example_questions": []},
    ]}), encoding="utf-8")
    return reg_dir


def _items(tmp: Path) -> list[JudgeItem]:
    return build_judge_items(_a0(tmp), _artifact(tmp),
                             [("swe_90-s0", "blk_a"), ("swe_90-s0", "blk_b")],
                             registries_dir=_registries(tmp))


# ------------------------------------------------------------------- tests

def test_class_permutation_deterministic_and_varied():
    p1 = class_permutation("swe_1-s3", "some_blocker", 4)
    p2 = class_permutation("swe_1-s3", "some_blocker", 4)
    assert p1 == p2 and sorted(p1) == [0, 1, 2, 3]
    perms = {tuple(class_permutation(f"swe_{i}-s0", "b", 3))
             for i in range(50)}
    assert len(perms) > 1  # not always identity -> position bias is broken


def test_make_excerpt_policies():
    text, policy = make_excerpt("short trace", ["anchor"], [])
    assert (text, policy) == ("short trace", "full")
    long = ("x" * 40_000 + "\nthe ANCHOR term here\n" + "y" * 40_000
            + "\nfinal answer tail")
    excerpt, policy = make_excerpt(long, ["anchor term"], ["sed -i s/a/b/ f"])
    assert policy == "excerpt"
    assert len(excerpt) <= MAX_TRACE_CHARS + 200
    assert excerpt.startswith("x")               # head kept
    assert excerpt.endswith("final answer tail")  # tail kept
    assert "ANCHOR term" in excerpt               # anchor window kept
    assert "[... trace elided ...]" in excerpt


def test_prompts_exclude_bias_fields(tmp_path):
    items = _items(tmp_path)
    assert [i.custom_id for i in items] == ["i00000", "i00001"]
    for it in items:
        system, user = build_judge_prompt(it)
        assert system == SYSTEM_PROMPT
        assert RESOLUTION_SECRET not in user       # registry resolution out
        assert "canonical" not in user.lower()     # canonical flag out
        assert it.description in user              # description in
        for name in it.class_names:
            assert f"name: {name}" in user
    # presentation order follows the deterministic permutation
    it = items[0]
    first = it.class_names[it.perm[0]]
    _, user0 = build_judge_prompt(it)
    assert f"1. name: {first}" in user0


def test_parse_judge_response():
    ok = parse_judge_response(
        '{"class": "cls_off", "confidence": 0.83, "evidence": "e",'
        ' "reasoning": "r"}')
    assert ok == {"class": "cls_off", "confidence": 0.83, "evidence": "e",
                  "reasoning": "r"}
    fenced = parse_judge_response(
        '```json\n{"class": null, "confidence": 2.5, "evidence": ""}\n```')
    assert fenced["class"] is None and fenced["confidence"] == 1.0
    prosey = parse_judge_response(
        'Sure - here is the label:\n{"class": "a", "confidence": "0.4"}\nok')
    assert prosey["class"] == "a" and prosey["confidence"] == 0.4
    assert parse_judge_response("no json here") is None
    assert parse_judge_response('{"confidence": 1}') is None  # no class key
    assert parse_judge_response('{"class": 3}') is None       # non-str class


def test_locate_evidence():
    trace = "line one\n  we   will disable\tfrobnication now\nend"
    exact = locate_evidence(trace, "disable\tfrobnication")
    assert trace[exact:exact + 7] == "disable"
    fuzzy = locate_evidence(trace, "We will DISABLE frobnication")
    assert fuzzy >= 0 and trace[fuzzy:fuzzy + 2].lower() == "we"
    assert locate_evidence(trace, "never said this") == -1
    assert locate_evidence(trace, "") == -1


def test_freeze_results_gates_and_load(tmp_path):
    items = _items(tmp_path)
    a0 = tmp_path / "a0"
    ev = "disable frobnication now"
    responses = session_responses([
        {"custom_id": "i00000", "class": "cls_off", "confidence": 0.9,
         "evidence": ev, "reasoning": "r"},
        {"custom_id": "i00001", "class": "not_a_class", "confidence": 0.9,
         "evidence": ev, "reasoning": "r"},
    ])
    frozen = freeze_results(items, responses, a0)
    by = {r["custom_id"]: r for r in frozen}
    assert by["i00000"]["status"] == "accepted"
    assert by["i00000"]["commit_char"] == TRACE.find(ev)
    assert by["i00001"]["status"] == "bad_class"

    # evidence not in trace -> rejected; abstain -> abstained; missing kept
    responses2 = session_responses([
        {"custom_id": "i00000", "class": "cls_off", "confidence": 0.9,
         "evidence": "fabricated quote", "reasoning": "r"}])
    frozen2 = freeze_results(items, responses2, a0)
    by2 = {r["custom_id"]: r for r in frozen2}
    assert by2["i00000"]["status"] == "evidence_not_found"
    assert by2["i00001"]["status"] == "missing"

    out = tmp_path / "labels.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in frozen:
            fh.write(json.dumps(r) + "\n")
    loaded = load_judge_labels(out, min_conf=0.7)
    assert loaded == {("swe_90-s0", "blk_a"):
                      ("cls_off", TRACE.find(ev), 0.9)}
    assert load_judge_labels(out, min_conf=0.95) == {}


def test_write_item_files_split_and_content(tmp_path):
    items = _items(tmp_path)
    files = write_item_files(items, tmp_path / "work", max_items=1)
    assert len(files) == 2
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["items"][0]["custom_id"] == "i00000"
    assert payload["items"][0]["system"] == SYSTEM_PROMPT
    assert "class_names" in payload["items"][0]


def test_unlabeled_pairs_scope(tmp_path):
    dbg = tmp_path / "labels_debug.jsonl"
    rows = [
        {"kind": "commitment", "run": "r1", "blocker": "b1", "chosen": None,
         "reason": "no signature hits", "scores": {}},
        {"kind": "commitment", "run": "r1", "blocker": "b2", "chosen": None,
         "reason": "tie between top classes", "scores": {}},
        {"kind": "commitment", "run": "r1", "blocker": "b3", "chosen": "c",
         "commit_char": 1, "label_source": "trace", "scores": {}},
        {"kind": "read", "run": "r1", "read_idx": 0},
    ]
    dbg.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                   encoding="utf-8")
    assert unlabeled_pairs(dbg) == [("r1", "b1")]
    assert unlabeled_pairs(dbg, include_ties=True) == [("r1", "b1"),
                                                       ("r1", "b2")]


def test_validation_set_builder_deterministic(tmp_path, monkeypatch):
    import validate_label_judge as vlj
    dbg = tmp_path / "labels_debug.jsonl"
    rows = []
    for i in range(5):
        rows.append({"kind": "commitment", "run": f"swe_1-s{i}",
                     "blocker": "bt", "chosen": "c", "label_source": "trace"})
    for i in range(400):
        rows.append({"kind": "commitment", "run": f"swe_2-s{i}",
                     "blocker": "ba", "chosen": "c",
                     "label_source": "actions"})
    dbg.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                   encoding="utf-8")
    set_path = tmp_path / "val.json"
    monkeypatch.setattr(vlj, "SET_PATH", set_path)
    assert vlj.build_set(str(dbg), force=False) == 0
    first = set_path.read_bytes()
    payload = json.loads(first)
    assert len(payload["items"]) == vlj.N_TOTAL
    assert payload["_meta"]["composition"] == {"trace": 5, "actions": 295}
    # frozen: refuses without --force; identical bytes with it
    assert vlj.build_set(str(dbg), force=False) == 1
    assert vlj.build_set(str(dbg), force=True) == 0
    assert set_path.read_bytes() == first


def _local_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    for name in ("Qwen/Qwen3-32B", "Qwen/Qwen2.5-Coder-7B-Instruct"):
        try:
            AutoTokenizer.from_pretrained(name, local_files_only=True)
            return name
        except Exception:  # noqa: BLE001 - any cache miss means "try next"
            continue
    return None


def test_build_labels_judge_merge(tmp_path):
    """Spec labels.md v3: judge labels fill ONLY lexicon-abstained pairs and
    never override. blk_a is trace-labelable (cls_off); blk_b is not."""
    tok = _local_tokenizer()
    if tok is None:
        pytest.skip("no locally-cached tokenizer")
    from wta.labeling import build_labels
    a0, classes = _a0(tmp_path), _artifact(tmp_path)
    judge = {
        # attempts to OVERRIDE a lexicon label -> must be ignored
        ("swe_90-s0", "blk_a"): ("cls_on", 0, 0.99),
        # fills a genuinely abstained pair -> must land as label_source=judge
        ("swe_90-s0", "blk_b"): ("cls_beta", 10, 0.88),
    }
    dbg = tmp_path / "dbg.jsonl"
    build_labels(a0, classes, tokenizer_name=tok, debug_path=dbg,
                 judge_labels=judge)
    commits = {}
    for line in dbg.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") == "commitment":
            commits[row["blocker"]] = row
    assert commits["blk_a"]["chosen"] == "cls_off"          # not overridden
    assert commits["blk_a"]["label_source"] == "trace"
    assert commits["blk_b"]["chosen"] == "cls_beta"
    assert commits["blk_b"]["label_source"] == "judge"
    assert commits["blk_b"]["judge_conf"] == 0.88
    assert commits["blk_b"]["commit_char"] == 10

    # without judge labels, blk_b stays unlabeled (baseline unchanged)
    dbg2 = tmp_path / "dbg2.jsonl"
    build_labels(a0, classes, tokenizer_name=tok, debug_path=dbg2)
    commits2 = {}
    for line in dbg2.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") == "commitment":
            commits2[row["blocker"]] = row
    assert commits2["blk_b"]["chosen"] is None
    assert commits2["blk_b"]["reason"] == "no signature hits"
