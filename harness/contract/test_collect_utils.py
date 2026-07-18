"""Contract: collection utilities, esp. the leak-sensitive pieces (ADR 012)."""

import numpy as np
import pytest

from wta.collect_utils import (
    DELIBERATION_NUDGE, answer_signature, build_prompt, patch_touched_files,
)

PATCH = """\
--- /dev/null
+++ b/changelogs/fragments/17587-fix.yml
@@ -0,0 +1,7 @@
--- a/lib/ansible/module_utils/common/sys_info.py
+++ b/lib/ansible/module_utils/common/sys_info.py
@@ -42,11 +40,12 @@ def get_distribution():
--- a/test/units/module_utils/common/test_sys_info.py
+++ b/test/units/module_utils/common/test_sys_info.py
@@ -1,3 +1,9 @@
--- a/lib/ansible/module_utils/common/sys_info.py
+++ b/lib/ansible/module_utils/common/sys_info.py
@@ -55,28 +54,27 @@ def get_distribution_version():
"""


def test_patch_touched_files_excludes_leaky_paths():
    """Tests and changelogs never become context: test NAMES encode expected
    behaviour (the swe_0 test ids literally contain SunOS->Solaris)."""
    files = patch_touched_files(PATCH)
    assert files == ["lib/ansible/module_utils/common/sys_info.py"]  # deduped too


def test_build_prompt_modes():
    p_bare = build_prompt("Fix the bug.", None, nudge=False)
    assert "Fix the bug." in p_bare and DELIBERATION_NUDGE not in p_bare
    assert "Relevant source files" not in p_bare

    ctx = [("lib/a.py", "def f():\n    pass")]
    p_ctx = build_prompt("Fix the bug.", ctx, nudge=True)
    assert "--- lib/a.py ---" in p_ctx and "def f():" in p_ctx
    assert DELIBERATION_NUDGE in p_ctx
    assert p_ctx.index("Fix the bug.") < p_ctx.index("lib/a.py")


def test_answer_signature_uses_last_code_block():
    a = "reasoning...\n```python\nx = 1\n```\nmore\n```python\ny = 2\n```\n"
    b = "totally different reasoning\n```python\ny  =  2\n```\n"
    c = "```python\ny = 3\n```\n"
    assert answer_signature(a) == answer_signature(b)  # last block, ws-normalized
    assert answer_signature(a) != answer_signature(c)
    # block-less fallback is deterministic and text-sensitive
    assert answer_signature("no code here") == answer_signature("no code here")
    assert answer_signature("no code here") != answer_signature("other text")


def test_artifact_task_ids_filters_train_pool(tmp_path):
    """collect_v2 --classes: only artifact tasks are collectable (sorted-dir
    order interleaves swe_60 before swe_7 — the filter is what keeps the
    sealed test pool untouched)."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from collect_v2 import artifact_task_ids

    art = {"_provenance": {"note": "x"}, "swe_0": {}, "swe_59": {}}
    p = tmp_path / "classes.json"
    p.write_text(json.dumps(art), encoding="utf-8")
    assert artifact_task_ids(p) == {"swe_0", "swe_59"}
