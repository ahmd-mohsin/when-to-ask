"""Contract: the R6 LLM-cell prompts are registry-blind (decisions/028 T1).

The judge must never receive the interpretation-class registry: no blocker
names, no class names, no mention of a registry/artifact in the prompt
templates, and the judge-visible payload keys must be exactly the frozen
whitelists (item_id + instruction/prefix for R6a, item_id + excerpts for
R6b) — ground-truth fields like truth/blocker/class stay out.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from r6_build_items import (R6A_PAYLOAD_KEYS, R6A_PROMPT,  # noqa: E402
                            R6B_PAYLOAD_KEYS, R6B_PROMPT)

ART = ROOT / "data" / "interpretation_classes.json"
FORBIDDEN_WORDS = ("registry", "blocker", "interpretation class",
                   "class artifact", "signature list", "anchor")


def test_templates_never_mention_the_registry():
    for tpl in (R6A_PROMPT, R6B_PROMPT):
        low = tpl.lower()
        for w in FORBIDDEN_WORDS:
            assert w not in low, f"template mentions {w!r}"


def test_templates_only_take_payload_placeholders():
    assert set(re.findall(r"{(\w+)}", R6A_PROMPT)) <= set(R6A_PAYLOAD_KEYS)
    assert set(re.findall(r"{(\w+)}", R6B_PROMPT)) <= set(R6B_PAYLOAD_KEYS)


def test_payload_whitelists_exclude_truth_fields():
    for keys in (R6A_PAYLOAD_KEYS, R6B_PAYLOAD_KEYS):
        for k in keys:
            assert "truth" not in k and "blocker" not in k and \
                "class" not in k and "label" not in k


CHUNKS = ROOT / "results" / "r6_chunks"


@pytest.mark.skipif(not CHUNKS.exists(), reason="payload chunks not built")
def test_payload_item_ids_are_opaque():
    """Build-order ids monotonically encode ground truth (pre-run review);
    judge-visible ids must be opaque salted hashes, never r6a_NNN."""
    seq = re.compile(r"^r6[ab]_\d+$")
    hexid = re.compile(r"^[0-9a-f]{12}$")
    files = sorted(CHUNKS.glob("payload_*_chunk_*.json"))
    assert files, "no payload chunk files"
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        for it in doc["items"]:
            assert not seq.match(it["item_id"]), \
                f"{f.name}: sequential id {it['item_id']} leaks build order"
            assert hexid.match(it["item_id"]), \
                f"{f.name}: unexpected id form {it['item_id']}"


@pytest.mark.skipif(not ART.exists(), reason="artifact not present")
def test_templates_contain_no_artifact_terms():
    art = json.loads(ART.read_text(encoding="utf-8"))
    terms = {b for task, bl in art.items() if task != "_provenance"
             for b in bl}
    terms |= {c["name"] for task, bl in art.items() if task != "_provenance"
              for spec in bl.values() for c in spec["classes"]}
    for tpl in (R6A_PROMPT, R6B_PROMPT):
        low = tpl.lower()
        for t in terms:
            assert t.lower() not in low, f"template leaks artifact term {t!r}"
