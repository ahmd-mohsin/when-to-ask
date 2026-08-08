"""Contract: hil_bench task statements load from harbor's shared/ layout
(spec eval, decisions/022).

The original candidate list missed harbor's shared/problem_statement.txt and
shared/metadata.json, silently yielding stub statements for every real task.
Uses swe_0 (train pool; the sealed pool swe_60+ stays untouched on the laptop).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xtid.harness.tasks import HIL_ROOT, load_hil_bench_tasks

SWE0 = HIL_ROOT / "harbor_swe" / "swe_0"

pytestmark = pytest.mark.skipif(
    not SWE0.exists(), reason="third_party/hil-bench clone not present"
)


def _swe0_task():
    tasks = load_hil_bench_tasks(domain="swe", limit=1)
    assert tasks, "no tasks loaded from harbor_swe"
    t = tasks[0]
    assert Path(t.meta["task_dir"]).name == "swe_0"
    return t


def test_statement_is_real_not_stub():
    t = _swe0_task()
    assert t.meta["statement_stub"] is False
    stmt_file = SWE0 / "shared" / "problem_statement.txt"
    expected_head = stmt_file.read_text(encoding="utf-8", errors="replace")[:200]
    assert t.statement.startswith(expected_head)
    assert not t.statement.startswith("HiL-Bench task ")


def test_instance_id_from_shared_metadata():
    t = _swe0_task()
    # shared/metadata.json carries the canonical instance_id; the fallback
    # (dir name) is also acceptable only when metadata is absent -- here it
    # exists, so the id must be non-empty and stable.
    assert t.instance_id
    assert t.instance_id != "HiL-Bench task swe_0"


def test_registry_still_resolves():
    t = _swe0_task()
    assert t.blockers, "blocker registry lost by the statement fix"
    assert Path(t.meta["registry"]).exists()
    b = t.blockers[0]
    assert b.id and b.resolution


def test_stub_flag_true_when_statement_missing(tmp_path):
    # a bare dir with only a registry -> stub statement, flagged
    task = tmp_path / "harbor_swe" / "swe_x"
    task.mkdir(parents=True)
    (task / "blocker_registry.json").write_text(
        '{"blockers": [{"id": "b1", "description": "d", "resolution": "r"}]}'
    )
    tasks = load_hil_bench_tasks(domain="swe", root=tmp_path)
    assert len(tasks) == 1
    assert tasks[0].meta["statement_stub"] is True
    assert tasks[0].statement.startswith("HiL-Bench task ")
