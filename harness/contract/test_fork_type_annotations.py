"""Contract: the frozen fork-type map stays complete and well-formed (022 §2g).

data/fork_type_annotations.json is the pre-registered blocker -> {structural|
value} split that score_eval.py slices the headline table by. Once signed
(decisions/024) it must cover EXACTLY the blockers of the interpretation-class
artifact — a blocker added to the artifact without a fork-type entry, a stale
entry for a removed blocker, or a value outside the two kinds would silently
corrupt the pre-registered slice.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "data" / "interpretation_classes.json"
ANN = ROOT / "data" / "fork_type_annotations.json"

pytestmark = pytest.mark.skipif(
    not (ART.exists() and ANN.exists()), reason="artifacts not present")


def test_annotations_cover_exactly_the_artifact_blockers():
    art = json.loads(ART.read_text(encoding="utf-8"))
    ann = json.loads(ANN.read_text(encoding="utf-8"))
    blockers = {b for task, bl in art.items() if task != "_provenance"
                for b in bl}
    missing = blockers - set(ann)
    stale = set(ann) - blockers
    assert not missing, f"blockers without a fork-type entry: {sorted(missing)}"
    assert not stale, f"fork-type entries for unknown blockers: {sorted(stale)}"


def test_annotation_values_are_the_two_kinds():
    ann = json.loads(ANN.read_text(encoding="utf-8"))
    bad = {b: k for b, k in ann.items() if k not in ("structural", "value")}
    assert not bad, f"invalid fork kinds: {bad}"
