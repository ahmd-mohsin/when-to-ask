"""Contract: pass@k port pins upstream semantics (spec eval, decisions/022 §3;
MIGRATED from run_hil_bench.py @352d14c).

Hand-built rows pin: ordered-first-k "any resolved"; infra_error filter;
trajectory_needs_rerun exclusions (hiccup strict=1, timeout lenient=3,
env-died last-step, unknown-error last response); the num_valid >=
expected_passes inclusion gate and its --include-partial variant.
"""

from __future__ import annotations

from xtid.harness.passk import (
    TRAJECTORY_HICCUP_OBS, summarize_rows, trajectory_needs_rerun,
)


def _row(task="t0", model="m", mode="baseline", pass_num=0, resolved=False,
         status="success", trajectory=None, **extra):
    row = {"task_name": task, "model": model, "mode": mode,
           "pass_num": pass_num, "resolved": resolved, "status": status,
           "trajectory": trajectory or []}
    row.update(extra)
    return row


def test_ordered_first_k_any_resolved():
    rows = [_row(pass_num=0, resolved=False),
            _row(pass_num=1, resolved=True),
            _row(pass_num=2, resolved=False)]
    m = summarize_rows(rows, include_partial=False, expected_passes=3)
    metrics = m["baseline"]["m"]
    assert metrics["pass_at_1"] == 0.0        # first pass unresolved
    assert metrics["pass_at_2"] == 1.0        # any of first 2
    assert metrics["pass_at_3"] == 1.0
    assert metrics["pass_at_3_n"] == 1


def test_pass_order_is_by_pass_num_not_input_order():
    rows = [_row(pass_num=2, resolved=False),
            _row(pass_num=0, resolved=True),
            _row(pass_num=1, resolved=False)]
    m = summarize_rows(rows, include_partial=False, expected_passes=3)
    assert m["baseline"]["m"]["pass_at_1"] == 1.0   # pass 0 resolved


def test_infra_error_pass_dropped_and_attempt_excluded():
    rows = [_row(pass_num=0, resolved=True, status="infra_error"),
            _row(pass_num=1, resolved=True),
            _row(pass_num=2, resolved=True)]
    # only 2 valid < expected 3 -> attempt excluded entirely
    m = summarize_rows(rows, include_partial=False, expected_passes=3)
    assert m == {}
    # include_partial admits it; ordered over the VALID passes
    m = summarize_rows(rows, include_partial=True, expected_passes=3)
    metrics = m["baseline"]["m"]
    assert metrics["pass_at_1"] == 1.0
    assert metrics["pass_at_3_n"] == 0        # nobody has 3 valid passes
    assert metrics["pass_at_2_n"] == 1


def test_trajectory_filters():
    hiccup = [{"obs": TRAJECTORY_HICCUP_OBS}]
    assert trajectory_needs_rerun(hiccup)                       # strict: 1 hit
    timeout_obs = {"obs": "Command '[foo]' timed out after 120 seconds"}
    assert not trajectory_needs_rerun([timeout_obs] * 2)        # lenient: <3
    assert trajectory_needs_rerun([timeout_obs] * 3)
    assert trajectory_needs_rerun(
        [{"obs": "fine"}, {"obs": "x Environment died unexpectedly"}])
    assert not trajectory_needs_rerun(
        [{"obs": "Environment died unexpectedly"}, {"obs": "fine"}])  # last-step only
    assert trajectory_needs_rerun(
        [{"obs": "ok", "response": "Exit due to unknown error"}])
    assert not trajectory_needs_rerun([{"obs": "all good"}])
    assert not trajectory_needs_rerun([])


def test_rerun_pass_excluded_from_attempt():
    rows = [_row(pass_num=0, resolved=True,
                 trajectory=[{"obs": TRAJECTORY_HICCUP_OBS}]),
            _row(pass_num=1, resolved=False),
            _row(pass_num=2, resolved=False)]
    m = summarize_rows(rows, include_partial=True, expected_passes=3)
    metrics = m["baseline"]["m"]
    # the resolved pass was invalid -> the remaining ordered passes miss
    assert metrics["pass_at_1"] == 0.0 and metrics["pass_at_2"] == 0.0


def test_multiple_tasks_pool_into_the_rate():
    rows = []
    for t, solved in (("t0", True), ("t1", False)):
        for p in range(3):
            rows.append(_row(task=t, pass_num=p,
                             resolved=(solved and p == 0)))
    m = summarize_rows(rows, include_partial=False, expected_passes=3)
    metrics = m["baseline"]["m"]
    assert metrics["pass_at_3"] == 0.5 and metrics["pass_at_3_n"] == 2
    assert metrics["num_included_attempts"] == 2


def test_modes_and_models_grouped_separately():
    rows = ([_row(mode="baseline", pass_num=p, resolved=True) for p in range(3)]
            + [_row(mode="detector", pass_num=p, resolved=False)
               for p in range(3)])
    m = summarize_rows(rows, include_partial=False, expected_passes=3)
    assert m["baseline"]["m"]["pass_at_3"] == 1.0
    assert m["detector"]["m"]["pass_at_3"] == 0.0
