"""pass@k semantics -- MIGRATED from HiL-Bench.

Source: third_party/hil-bench/run_hil_bench.py (commit 352d14c):
``summarize_rows`` (the ONLY real pass@k implementation -- ordered-first-k
"any resolved" over filtered passes), ``trajectory_needs_rerun`` + its
helper predicates and constants, ``extract_public_trajectory_steps``.
Ported line-faithfully (decisions/022 §3) so the identical aggregation runs
over our-loop arms and bridge rows; reimplementing it loosely would break
comparability with the paper's Table 1 protocol.

Deviations, all mechanical:
  * trajectory loading is injectable (``load_trajectory(row) -> steps``);
    the default reads ``row["trajectory"]`` (a steps list) or, failing
    that, ``row["trajectory_dir"]`` via ``load_trajectory_steps_from_dir``
    exactly as upstream.
  * the cost/token averages that need litellm cost tracking are kept but
    default to 0.0 when the row lacks the field (same ``or 0.0`` upstream).
  * the ask_* metrics gated on ``mode == "ask_human"`` upstream are NOT
    computed here -- Ask-F1 comes from our ``ask_f1.compute_hil_metrics``
    recompute for every arm (decisions/022 §3: the upstream summary is
    wrong for zero-question passes and hides non-"ask_human" mode names).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Verbatim constants (run_hil_bench.py @352d14c)
# ---------------------------------------------------------------------------

TRAJECTORY_TIMEOUT_OBS_RE = re.compile(r"Command '\[.*\]' timed out after \d+ seconds")
TRAJECTORY_HICCUP_OBS = "can't answer (perhaps transient hiccup)"
TRAJECTORY_ENV_DIED_OBS = "Environment died unexpectedly"
TRAJECTORY_UNKNOWN_ERROR = "Exit due to unknown error"
KB_QUERY_ERROR = "Error querying knowledge base"
SQL_QUOTING_BUG_MARKERS = (
    ("get_database_info", "Error: database $"),
    ("get_table_info", "Error: table $"),
    ("get_column_info", "Error: column $"),
    ("get_business_info", "No business information found matching '$"),
)
TRAJECTORY_RERUN_OCCURRENCE_THRESHOLD_STRICT = 1
TRAJECTORY_RERUN_OCCURRENCE_THRESHOLD_LENIENT = 3


# ---------------------------------------------------------------------------
# Verbatim trajectory predicates
# ---------------------------------------------------------------------------


def trajectory_has_timeout_obs(trajectory: list[dict[str, str]]) -> bool:
    count = 0
    for step in trajectory:
        obs = step.get("obs", "")
        if isinstance(obs, str) and TRAJECTORY_TIMEOUT_OBS_RE.search(obs):
            count += 1
    return count >= TRAJECTORY_RERUN_OCCURRENCE_THRESHOLD_LENIENT


def trajectory_has_hiccup_obs(trajectory: list[dict[str, str]]) -> bool:
    count = 0
    for step in trajectory:
        obs = step.get("obs", "")
        if isinstance(obs, str) and obs.strip() == TRAJECTORY_HICCUP_OBS:
            count += 1
    return count >= TRAJECTORY_RERUN_OCCURRENCE_THRESHOLD_STRICT


def trajectory_has_env_died_obs(trajectory: list[dict[str, str]]) -> bool:
    if not trajectory:
        return False
    obs = trajectory[-1].get("obs", "")
    return isinstance(obs, str) and TRAJECTORY_ENV_DIED_OBS in obs


def trajectory_has_unknown_error(trajectory: Any) -> bool:
    if not isinstance(trajectory, list) or not trajectory:
        return False
    last_step = trajectory[-1]
    if not isinstance(last_step, dict):
        return False
    response = last_step.get("response", "")
    if not isinstance(response, str):
        return False
    return TRAJECTORY_UNKNOWN_ERROR in response


def trajectory_has_kb_query_error(trajectory: Any) -> bool:
    if not isinstance(trajectory, list) or not trajectory:
        return False
    count = 0
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        obs = step.get("obs", "")
        if isinstance(obs, str) and KB_QUERY_ERROR in obs:
            count += 1
            if count >= TRAJECTORY_RERUN_OCCURRENCE_THRESHOLD_STRICT:
                return True
    return False


def trajectory_has_sql_quoting_bug_obs(trajectory: Any) -> bool:
    if not isinstance(trajectory, list):
        return False
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        act = step.get("act", "")
        obs = step.get("obs", "")
        if not isinstance(act, str) or not isinstance(obs, str):
            continue
        tool = act.split(None, 1)[0] if act.strip() else ""
        for marker_tool, marker_obs in SQL_QUOTING_BUG_MARKERS:
            if tool == marker_tool and obs.startswith(marker_obs):
                return True
    return False


def trajectory_needs_rerun(trajectory: list[dict[str, str]]) -> bool:
    return (
        trajectory_has_timeout_obs(trajectory)
        or trajectory_has_hiccup_obs(trajectory)
        or trajectory_has_env_died_obs(trajectory)
        or trajectory_has_unknown_error(trajectory)
        or trajectory_has_kb_query_error(trajectory)
        or trajectory_has_sql_quoting_bug_obs(trajectory)
    )


# ---------------------------------------------------------------------------
# Verbatim .traj loading (for bridge rows)
# ---------------------------------------------------------------------------


def stringify_trajectory_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return str(value)


def extract_public_trajectory_steps(traj_payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_steps = traj_payload.get("trajectory", [])
    if not isinstance(raw_steps, list):
        return []
    steps: list[dict[str, str]] = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        step_payload: dict[str, str] = {
            "act": stringify_trajectory_value(step.get("action", "")),
            "obs": stringify_trajectory_value(step.get("observation", "")),
        }
        for key in ["response", "thought", "execution_time", "state",
                    "extra_info", "tool_calls", "tool_call_ids",
                    "thinking_blocks"]:
            if key in step:
                value = step.get(key)
                if value is not None and value != "":
                    step_payload[key] = stringify_trajectory_value(value)
        steps.append(step_payload)
    return steps


def load_trajectory_steps_from_dir(trajectory_dir: str | None) -> list[dict[str, str]]:
    if not trajectory_dir:
        return []
    traj_dir = Path(trajectory_dir)
    if not traj_dir.exists() or not traj_dir.is_dir():
        return []
    traj_files = sorted(traj_dir.glob("*.traj"))
    if not traj_files:
        return []
    try:
        payload = json.loads(traj_files[0].read_text())
    except Exception:
        return []
    return extract_public_trajectory_steps(payload)


def _default_load_trajectory(row: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(row.get("trajectory"), list):
        return row["trajectory"]
    return load_trajectory_steps_from_dir(str(row.get("trajectory_dir") or ""))


# ---------------------------------------------------------------------------
# summarize_rows -- the pass@k aggregation, ported line-faithfully
# ---------------------------------------------------------------------------


def summarize_rows(
    rows: list[dict[str, Any]],
    include_partial: bool,
    expected_passes: int,
    load_trajectory=None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """rows: one dict per (task, model, mode, pass) with at least
    ``task_name, model, mode, pass_num, status, resolved`` and a trajectory
    source (``trajectory`` steps list or ``trajectory_dir``)."""
    load_trajectory = load_trajectory or _default_load_trajectory

    grouped_attempt_rows = defaultdict(list)
    for row in rows:
        grouped_attempt_rows[(row["task_name"], row["model"], row["mode"])].append(row)

    grouped_mode_model_attempts = defaultdict(list)
    for (task_name, model, mode), attempt_rows in grouped_attempt_rows.items():
        valid_passes = []
        for row in sorted(attempt_rows, key=lambda r: int(r.get("pass_num", 0))):
            if row.get("status") == "infra_error":
                continue
            if trajectory_needs_rerun(load_trajectory(row)):
                continue
            valid_passes.append(row)
        num_valid = len(valid_passes)
        should_include = num_valid >= 1 if include_partial else num_valid >= expected_passes
        if not should_include:
            continue
        grouped_mode_model_attempts[(mode, model)].append(valid_passes)

    finalized: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (mode, model), attempt_passes in grouped_mode_model_attempts.items():
        num_solved_by_pass_k = {k: 0 for k in range(1, expected_passes + 1)}
        num_attempts_with_k_passes = {k: 0 for k in range(1, expected_passes + 1)}
        total_attempts_and_passes = 0
        total_steps = 0.0
        total_questions = 0.0
        for valid_passes in attempt_passes:
            num_valid = len(valid_passes)
            for k in range(1, expected_passes + 1):
                if num_valid >= k:
                    num_attempts_with_k_passes[k] += 1
            for k in range(1, num_valid + 1):
                if any(bool(valid_passes[i].get("resolved")) for i in range(k)):
                    num_solved_by_pass_k[k] += 1
            for row in valid_passes:
                total_attempts_and_passes += 1
                total_steps += float(row.get("num_steps") or 0.0)
                total_questions += float(row.get("num_questions") or 0.0)

        metrics: dict[str, Any] = {
            "num_included_attempts": len(attempt_passes),
            "num_passes": expected_passes,
            "total_attempts_and_passes": total_attempts_and_passes,
            "avg_steps_per_pass": (total_steps / total_attempts_and_passes)
            if total_attempts_and_passes > 0 else 0.0,
            "avg_num_questions_per_pass": (total_questions / total_attempts_and_passes)
            if total_attempts_and_passes > 0 else 0.0,
        }
        for k in range(1, expected_passes + 1):
            denominator = num_attempts_with_k_passes[k]
            metrics[f"pass_at_{k}"] = (num_solved_by_pass_k[k] / denominator
                                       if denominator > 0 else 0.0)
            metrics[f"pass_at_{k}_n"] = denominator
        finalized[mode][model] = metrics
    return dict(finalized)
