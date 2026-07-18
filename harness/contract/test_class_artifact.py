"""Contract: the interpretation-class artifact stays leak-free (decisions/005).

Runs scripts/audit_class_artifact.py's checks over the REAL artifact on every
test run, so a leaky anchor (anchor == class signature), a signature shared by
two classes, or a schema violation can never silently enter the artifact
again. The 8 leaks that shipped in the original 20-task derivation were
repaired 2026-07-18 after a robustness check proved them a no-op for all
measured results (decisions/018) — the tolerance is now ZERO.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "data" / "interpretation_classes.json"

sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(not ART.exists(), reason="artifact not present")


def test_artifact_has_zero_leak_errors():
    from audit_class_artifact import audit

    art = json.loads(ART.read_text(encoding="utf-8"))
    errors, _warnings = audit(art)
    assert not errors, "artifact leaks (fix before any run):\n" + "\n".join(errors)
