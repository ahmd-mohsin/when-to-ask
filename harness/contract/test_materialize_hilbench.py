"""Contract: flat-task materialization for the bridge (spec eval-bridge,
decisions/022).

Uses swe_0 (train pool) with local files only; --extract-scripts (docker) is
GPU-time and not exercised here. Pins: the flat layout resolve_swe_input_path
needs (problem_statement.txt / metadata.json / blocker_registry.json at task
root), the prepare_swe_task metadata conventions, and pool hygiene (no
train+sealed mixing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from materialize_hilbench_tasks import (  # noqa: E402
    SWEAP_LOG_PARSER, check_pool_hygiene, materialize_task, task_number,
)

SWE0 = REPO / "third_party" / "hil-bench" / "harbor_swe" / "swe_0"

pytestmark = pytest.mark.skipif(
    not SWE0.exists(), reason="third_party/hil-bench clone not present")


def test_materialize_swe0_flat_layout(tmp_path):
    st = materialize_task(SWE0, tmp_path, extract_scripts=False)
    dest = tmp_path / "swe_0"
    # the three files hil swe / ask_human_server need at the ROOT
    for name in ("problem_statement.txt", "metadata.json",
                 "blocker_registry.json"):
        assert (dest / name).exists(), name
    stmt = (dest / "problem_statement.txt").read_text(encoding="utf-8",
                                                      errors="replace")
    assert stmt.strip() and not stmt.startswith("HiL-Bench task")
    reg = json.loads((dest / "blocker_registry.json").read_text(encoding="utf-8"))
    assert reg.get("blockers"), "registry lost in materialization"
    meta = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    # prepare_swe_task conventions
    assert meta["instance_id"] == "swe_0"
    assert meta["repo_name"] == "app"
    assert meta["log_parser"] == SWEAP_LOG_PARSER
    assert "run_script.sh" in meta["test_cmd"]
    assert meta.get("image_name"), "image name must survive"
    assert st["scripts"] == "skipped"
    # source tree untouched: files still under shared/
    assert (SWE0 / "shared" / "problem_statement.txt").exists()
    assert not (SWE0 / "problem_statement.txt").exists()


def test_pool_hygiene():
    assert check_pool_hygiene(["swe_60", "swe_99"], 60) == "sealed"
    assert check_pool_hygiene(["swe_0", "swe_59"], 60) == "train"
    with pytest.raises(ValueError, match="mixes train"):
        check_pool_hygiene(["swe_0", "swe_60"], 60)
    assert task_number("swe_42") == 42 and task_number("nope") is None
