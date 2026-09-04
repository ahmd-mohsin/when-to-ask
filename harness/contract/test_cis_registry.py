"""Contract: the CIS resolution loader (decisions/029).

- The four load-time assertions hold on the real 60-task registry set.
- The per-resolution content is PINNED by sha256 in data/cis_registry_pins.json
  (with the id order and the defective-id list), so any drift in the vendored
  benchmark, or any accidental unescaping, fails loudly.
- Foreign controls: different task, matched type / code / length, no
  identifier leak into the target instruction, deterministic under seed.
- The rival fixture loader only admits owner-approved entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wta.cis_registry import (Resolution, defective_ids, foreign_controls,
                              identifiers, load_resolutions, load_rival_fixture,
                              pins, render_full_info)

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "third_party" / "hil-bench" / "harbor_swe"
CLASSES = REPO / "data" / "interpretation_classes.json"
PINS = REPO / "data" / "cis_registry_pins.json"
RIVALS = REPO / "data" / "cis_rival_resolutions_pilot.json"


@pytest.fixture(scope="module")
def res():
    if not (TASKS / "swe_0").exists():
        pytest.skip("hil-bench tasks not vendored on this machine")
    return load_resolutions(TASKS, CLASSES)


def test_universe_and_assertions(res):
    assert len(res) == 214
    assert len({k[0] for k in res}) == 60
    for r in res.values():
        assert r.resolution == r.resolution.strip() and r.resolution
        assert r.n_classes >= 2


def test_pins_match_vendored_registry(res):
    assert PINS.exists(), "run scripts/cis_pin_registry.py to regenerate"
    want = json.loads(PINS.read_text(encoding="utf-8"))
    got = pins(res)
    assert got["n_tasks"] == want["n_tasks"] == 60
    assert got["n_blockers"] == want["n_blockers"] == 214
    assert got["defective_ids"] == want["defective_ids"]
    for task, rec in want["tasks"].items():
        assert got["tasks"][task]["id_order"] == rec["id_order"], task
        assert got["tasks"][task]["sha256"] == rec["sha256"], task


def test_no_unescaping_ever_happens(res):
    """029 policy: verbatim. A resolution known to carry a literal backslash-n
    must still carry it after loading."""
    bad = defective_ids(res)
    assert "swe_38/ambiguous_expired_entry_refresh_failure_behavior" in bad
    with_n = [k for k, r in res.items() if "\\n" in r.resolution]
    assert with_n, "expected literal \\n in some resolutions"
    for k in with_n:
        assert "\\n" in res[k].resolution


def test_render_template_is_exact_on_one_task():
    task = TASKS / "swe_0"
    if not task.exists():
        pytest.skip("no tasks")
    reg = json.loads((task / "shared" / "ask-human-data" / "blocker_registry.json")
                     .read_text(encoding="utf-8"))["blockers"]
    base = (task / "baseline" / "instruction.md").read_text(encoding="utf-8")
    full = (task / "full_info" / "instruction.md").read_text(encoding="utf-8")
    assert render_full_info(base, reg) == full


def test_foreign_controls_invariants(res):
    instr = {t: (TASKS / t / "baseline" / "instruction.md").read_text(encoding="utf-8")
             for t in sorted({k[0] for k in res})}
    key = ("swe_0", "non_linux_distribution_source_precedence_on_conflict")
    a = foreign_controls(key, res, instr, n=2, seed=0)
    b = foreign_controls(key, res, instr, n=2, seed=0)
    assert a == b and len(a) == 2
    tgt = res[key]
    tgt_idents = identifiers(instr["swe_0"])
    for k in a:
        r = res[k]
        assert r.task != "swe_0"
        assert r.has_code == tgt.has_code
        assert not (identifiers(r.resolution) & tgt_idents)
    # a different seed may pick different controls; a different key must not
    # collide with the target's own task
    c = foreign_controls(key, res, instr, n=2, seed=1)
    assert all(res[k].task != "swe_0" for k in c)
    # coverage: every pilot blocker gets its two controls
    for t in ("swe_0", "swe_10", "swe_11"):
        for k in [k for k in res if k[0] == t]:
            assert len(foreign_controls(k, res, instr, n=2, seed=0)) == 2, k


def test_rival_fixture_admits_only_approved():
    if not RIVALS.exists():
        pytest.skip("no rival fixture")
    d = json.loads(RIVALS.read_text(encoding="utf-8"))
    approved = load_rival_fixture(RIVALS)
    n_approved = sum(1 for e in d["entries"] if e.get("status") == "approved")
    assert len(approved) == n_approved
    for e in d["entries"]:
        assert e["status"] in ("draft_pending_owner_review", "approved", "rejected")
        assert e["class"] != e["canonical_class"]
