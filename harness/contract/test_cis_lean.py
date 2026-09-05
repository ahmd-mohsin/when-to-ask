"""Contract: the 029.3 lean readout (decisions/029 Amendment 029.3).

- The statement fixture is a pure function of the registry + the reviewed
  rival fixture under the frozen template (idempotent regeneration), every
  complete blocker has one statement per class in class-artifact order, and
  class 0's statement wraps the verbatim canonical resolution.
- Readout math: PMI, softmax, normalised entropy, argmax; shared-template
  tokens cancel; a constant added to every option leaves p unchanged.
- The null context renders with the same generation header the collector
  used (tokenizer test, skipped if unavailable).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from wta.cis_lean import (NULL_USER, STMT_PREFIX, entropy_norm, null_messages,
                          readout, softmax, statement)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
FIX = REPO / "data" / "cis_lean_statements_pilot.json"
TASKS = REPO / "third_party" / "hil-bench" / "harbor_swe"


def test_readout_math():
    r = readout("b", ["canon", "r1", "r2"], lp_ctx=[-10.0, -12.0, -11.0],
                lp_null=[-9.0, -9.0, -9.0], n_tok=[5, 5, 5])
    assert r.pmi == [-1.0, -3.0, -2.0]
    assert abs(sum(r.p) - 1) < 1e-12 and r.argmax == "canon" and r.p_canonical == max(r.p)
    # a constant added to every option (shared template tokens) cancels
    r2 = readout("b", ["canon", "r1", "r2"], lp_ctx=[-17.0, -19.0, -18.0],
                 lp_null=[-16.0, -16.0, -16.0], n_tok=[5, 5, 5])
    assert all(abs(a - b) < 1e-12 for a, b in zip(r.p, r2.p))
    assert entropy_norm([1.0, 0.0, 0.0]) == 0.0
    assert abs(entropy_norm([1 / 3] * 3) - 1.0) < 1e-12
    assert entropy_norm([1.0]) == 0.0
    assert softmax([0.0, 0.0]) == [0.5, 0.5]


def test_statement_fixture_is_idempotent_and_well_formed():
    if not FIX.exists() or not (TASKS / "swe_0").exists():
        pytest.skip("fixture or tasks absent")
    from cis_build_statements import PILOT_TASKS, build
    fresh = build(TASKS, REPO / "data" / "interpretation_classes.json",
                  REPO / "data" / "cis_rival_resolutions_pilot.json", PILOT_TASKS)
    on_disk = json.loads(FIX.read_text(encoding="utf-8"))
    assert fresh["blockers"] == on_disk["blockers"], "regenerate with scripts/cis_build_statements.py"
    assert on_disk["_provenance"]["template"] == {"STMT_PREFIX": STMT_PREFIX, "NULL_USER": NULL_USER}
    art = json.loads((REPO / "data" / "interpretation_classes.json").read_text(encoding="utf-8"))
    for b in on_disk["blockers"]:
        classes = [c["name"] for c in art[b["task"]][b["blocker_id"]]["classes"]]
        assert b["classes"] == classes
        assert len(b["statements"]) == len(classes)
        if b["complete"]:
            assert all(s is not None and s.startswith(STMT_PREFIX) for s in b["statements"])
            assert len(set(b["statements"])) == len(b["statements"]), "duplicate options"
    # every pilot blocker is complete (23 approved rivals cover all 23 rival classes)
    assert on_disk["n_complete"] == on_disk["n_blockers"] == 12
    assert on_disk["n_statements"] == 12 + 23


def test_canonical_statement_wraps_verbatim_resolution():
    if not FIX.exists():
        pytest.skip("no fixture")
    from wta.cis_registry import load_resolutions
    res = load_resolutions(TASKS, REPO / "data" / "interpretation_classes.json",
                           task_ids=["swe_0", "swe_10", "swe_11"])
    on_disk = json.loads(FIX.read_text(encoding="utf-8"))
    for b in on_disk["blockers"]:
        assert b["statements"][0] == statement(res[(b["task"], b["blocker_id"])].resolution)


def test_null_context_renders_with_generation_header():
    try:
        from wta.hf_reader import _load_tokenizer
        tok = _load_tokenizer("Qwen/Qwen3-32B")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer unavailable: {type(e).__name__}")
    from wta.cis_context import GEN_HEADER
    p = tok.apply_chat_template(null_messages(), tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    assert p.endswith(GEN_HEADER)
    assert "swe_" not in p and "blocker" not in p.lower()
    s = tok(statement("x must be y."), add_special_tokens=False)["input_ids"]
    assert len(s) > 5
