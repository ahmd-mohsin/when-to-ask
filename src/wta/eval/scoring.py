"""Headline-table scoring (spec eval, decisions/022 §3).

One code path scores BOTH sides: our-loop pass dirs (run_eval.py output) and
bridge rows (hil-bench's own harness output). Ask-F1 always goes through the
ported ``compute_hil_metrics`` with TRUE registry blocker counts -- never
upstream's summary (its zero-question passes get n_blockers_present=0 via a
swallowed ImportError, and non-"ask_human" mode names get no ask metrics).
pass@k always goes through the ported ``passk.summarize_rows``. Test
execution (`resolved`) is always hil-bench's own evaluator, invoked --
``HilBenchResolver`` (GPU/docker); CPU tests inject a fake resolver at the
same seam.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from xtid.harness.ask_f1 import GlobalMetrics, InstanceLog, compute_hil_metrics
from xtid.harness.passk import summarize_rows

REPO_ROOT = Path(__file__).resolve().parents[3]
HIL_ROOT = REPO_ROOT / "third_party" / "hil-bench"


# ---------------------------------------------------------------------------
# Resolver seam
# ---------------------------------------------------------------------------


class HilBenchResolver:
    """GPU/docker-only: hil-bench's own patch-apply-and-test evaluator.

    predictions: {expanded_instance_id: {instance_id, model_name_or_path,
    model_patch}} -- expanded ids ({orig}__{model}__{mode}__pass_{n}) keep
    same-task passes distinct; custom_eval strips them back to the original
    for metadata lookup."""

    def __init__(self, hil_root: Path = HIL_ROOT, run_id: str = "wta-eval",
                 max_workers: int = 1, timeout: int = 1800):
        self.hil_root, self.run_id = Path(hil_root), run_id
        self.max_workers, self.timeout = max_workers, timeout

    def resolve(self, predictions: dict, tasks_dir: Path) -> dict[str, bool]:
        sys.path.insert(0, str(self.hil_root))
        try:
            from hil_bench.utils.custom_eval import evaluate_custom_instances
        finally:
            sys.path.pop(0)
        out = evaluate_custom_instances(
            predictions, Path(tasks_dir), run_id=self.run_id,
            timeout=self.timeout, max_workers=self.max_workers)
        resolved = set(out.get("resolved_ids", []))
        return {iid: iid in resolved for iid in predictions}


# ---------------------------------------------------------------------------
# Our-loop pass records
# ---------------------------------------------------------------------------


@dataclass
class PassRecord:
    task: str                      # task dir name (swe_60)
    instance_id: str               # canonical instance id (prediction target)
    arm: str
    pass_idx: int
    prediction: dict | None
    questions: list[dict]          # [{question, response, blocker_name}]
    n_blockers: int
    ask_events: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    compute: dict = field(default_factory=dict)

    @property
    def expanded_id(self) -> str:
        return f"{self.instance_id}__ours__{self.arm}__pass_{self.pass_idx}"


def collect_our_results(eval_root: Path) -> list[PassRecord]:
    """Walk run_eval.py's output tree: <root>/<task>/<arm>/pass_<k>/."""
    records: list[PassRecord] = []
    eval_root = Path(eval_root)
    for committed_f in sorted(eval_root.glob("*/*/pass_*/committed.json")):
        pass_dir = committed_f.parent
        arm = pass_dir.parent.name
        task = pass_dir.parent.parent.name
        pass_idx = int(pass_dir.name.split("_")[-1])
        committed = json.loads(committed_f.read_text(encoding="utf-8"))
        ask = json.loads((pass_dir / "ask_log.json").read_text(encoding="utf-8")) \
            if (pass_dir / "ask_log.json").exists() else {}
        compute = json.loads((pass_dir / "compute.json").read_text(encoding="utf-8")) \
            if (pass_dir / "compute.json").exists() else {}
        session = ask.get("session", {})
        inst_key = ask.get("instance_key")
        slot = session.get(inst_key, {}) if inst_key else \
            (next(iter(session.values())) if session else {})
        prediction = committed.get("prediction")
        records.append(PassRecord(
            task=task,
            instance_id=(prediction or {}).get("instance_id", task),
            arm=arm, pass_idx=pass_idx, prediction=prediction,
            questions=slot.get("questions", []),
            n_blockers=int(slot.get("n_blockers", 0)),
            ask_events=ask.get("events", []), stats=ask.get("stats", {}),
            compute=compute))
    return records


def predictions_by_arm(records: list[PassRecord]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        if r.prediction is None:
            continue
        row = dict(r.prediction)
        row["instance_id"] = r.expanded_id
        out.setdefault(r.arm, {})[r.expanded_id] = row
    return out


def our_rows(records: list[PassRecord], resolved: dict[str, bool],
             model: str = "ours") -> list[dict]:
    """Rows for passk.summarize_rows. The trajectory given to the rerun
    filters carries each ask RESPONSE as an observation, so a judge hiccup
    (CANT_ANSWER) invalidates the pass exactly as upstream's tool output
    does. A pass whose committed.json is missing produces no row (run_eval
    logs pass_error instead of an infra_error row -- disclosed deviation)."""
    rows = []
    for r in records:
        rows.append({
            "task_name": r.task, "model": model, "mode": r.arm,
            "pass_num": r.pass_idx, "status": "success",
            "resolved": bool(resolved.get(r.expanded_id, False)),
            "num_steps": r.compute.get("turns", 0),
            "num_questions": len(r.questions),
            "trajectory": [{"obs": q.get("response", "")} for q in r.questions],
        })
    return rows


# ---------------------------------------------------------------------------
# Ask-F1 (both sides, true blocker counts)
# ---------------------------------------------------------------------------


def ask_metrics_by_arm(records: list[PassRecord],
                       true_counts: dict[str, int] | None = None
                       ) -> dict[str, GlobalMetrics]:
    logs_by_arm: dict[str, dict[str, InstanceLog]] = {}
    for r in records:
        n = true_counts.get(r.task, r.n_blockers) if true_counts else r.n_blockers
        logs_by_arm.setdefault(r.arm, {})[r.expanded_id] = InstanceLog(
            n_blockers=n, questions=list(r.questions))
    return {arm: compute_hil_metrics(logs) for arm, logs in logs_by_arm.items()}


def split_expanded_id(instance_id: str) -> tuple[str, str, str, int]:
    """{orig}__{model}__{mode}__pass_{n} -> (orig, model, mode, n); a plain
    id comes back as (id, '', '', 0)."""
    parts = instance_id.rsplit("__", 3)
    if len(parts) == 4 and parts[3].startswith("pass_"):
        return parts[0], parts[1], parts[2], int(parts[3].split("_")[-1])
    return instance_id, "", "", 0


def bridge_ask_metrics(ask_logs: dict, true_counts_by_orig: dict[str, int]
                       ) -> dict[str, GlobalMetrics]:
    """Recompute bridge Ask-F1 from the harness's ask_human_logs.json
    ({instance_id: {questions, n_blockers, ...}}) with TRUE registry counts
    (decisions/022 §3 -- the upstream zero-question bug). Grouped by the
    mode embedded in the expanded instance id."""
    logs_by_mode: dict[str, dict[str, InstanceLog]] = {}
    for iid, log in ask_logs.items():
        orig, _model, mode, _n = split_expanded_id(iid)
        n = true_counts_by_orig.get(orig)
        if n is None:
            n = int(log.get("n_blockers", 0))
        logs_by_mode.setdefault(mode or "ask_human", {})[iid] = InstanceLog(
            n_blockers=n, questions=list(log.get("questions", [])))
    return {mode: compute_hil_metrics(logs)
            for mode, logs in logs_by_mode.items()}


# ---------------------------------------------------------------------------
# Regime-sliced recall (structural vs value pre-registered separately)
# ---------------------------------------------------------------------------


def regime_recall_by_arm(records: list[PassRecord],
                         blockers_by_task: dict[str, list],
                         fork_annotations: dict[str, str] | None
                         ) -> dict[str, dict[str, dict]]:
    """Per arm, per slice: {discovered, present, recall}. Slices:
    fork_structural / fork_value / fork_unannotated (every registered
    hil-bench blocker is should-ask at the coarse level; the behavioural
    confident-convergent refinement is a separate analysis)."""
    fork_annotations = fork_annotations or {}

    def slice_of(blocker) -> str:
        kind = fork_annotations.get(blocker.id)
        if kind in ("structural", "value"):
            return f"fork_{kind}"
        return "fork_unannotated"

    out: dict[str, dict[str, dict]] = {}
    arms = sorted({r.arm for r in records})
    for arm in arms:
        arm_recs = [r for r in records if r.arm == arm]
        tally: dict[str, dict] = {}
        for task, blockers in blockers_by_task.items():
            task_recs = [r for r in arm_recs if r.task == task]
            if not task_recs:
                continue
            discovered_ids = {q.get("blocker_name")
                              for r in task_recs for q in r.questions
                              if q.get("blocker_name")}
            for b in blockers:
                s = tally.setdefault(slice_of(b), {"discovered": 0, "present": 0})
                s["present"] += 1
                if b.id in discovered_ids:
                    s["discovered"] += 1
        for s in tally.values():
            s["recall"] = s["discovered"] / s["present"] if s["present"] else 0.0
        out[arm] = tally
    return out


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------


def budget_by_arm(records: list[PassRecord]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for arm in sorted({r.arm for r in records}):
        recs = [r for r in records if r.arm == arm]
        n_pass = len(recs)
        asks = sum(len(r.questions) for r in recs)
        suppressed = sum(int(r.stats.get("suppressed_refires", 0)) for r in recs)
        fired = sum(int(r.stats.get("asks", 0)) for r in recs)
        out[arm] = {
            "asks_per_pass": asks / n_pass if n_pass else 0.0,
            "suppressed_refires": suppressed,
            "asks_per_fired": asks / fired if fired else 0.0,
        }
    return out


def compute_by_arm(records: list[PassRecord]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for arm in sorted({r.arm for r in records}):
        recs = [r for r in records if r.arm == arm]
        n_pass = len(recs) or 1
        out[arm] = {
            "n_runs": max((r.compute.get("n_runs", 1) for r in recs), default=1),
            "turns_per_pass": sum(r.compute.get("turns", 0) for r in recs) / n_pass,
            "generated_chars_per_pass": sum(r.compute.get("generated_chars", 0)
                                            for r in recs) / n_pass,
            "phrasing_calls": sum(r.compute.get("phrasing_calls", 0) for r in recs),
        }
    return out


def build_headline(pass_summary: dict, ask_by_arm: dict, regime_by_arm: dict,
                   budgets: dict, computes: dict, expected_passes: int,
                   bridge_ask: dict | None = None,
                   bridge_pass: dict | None = None) -> tuple[str, list[dict]]:
    """-> (markdown, csv rows). pass_summary is passk.summarize_rows output
    for our arms; bridge_* the same two structures for bridge rows."""
    csv_rows: list[dict] = []

    def rows_for(source, summary, ask):
        for mode in sorted(set(list(summary or {}) + list(ask or {}))):
            models = (summary or {}).get(mode, {})
            metrics = next(iter(models.values()), {}) if models else {}
            am = (ask or {}).get(mode)
            reg = regime_by_arm.get(mode, {}) if source == "ours" else {}
            bud = budgets.get(mode, {}) if source == "ours" else {}
            comp = computes.get(mode, {}) if source == "ours" else {}
            csv_rows.append({
                "source": source, "arm": mode,
                "ask_f1": round(am.ask_f1, 4) if am else "",
                "ask_precision": round(am.precision, 4) if am else "",
                "ask_recall": round(am.recall, 4) if am else "",
                f"pass_at_{expected_passes}":
                    round(metrics.get(f"pass_at_{expected_passes}", 0.0), 4)
                    if metrics else "",
                "pass_n": metrics.get(f"pass_at_{expected_passes}_n", "")
                    if metrics else "",
                "fork_structural_recall":
                    round(reg.get("fork_structural", {}).get("recall", 0.0), 4)
                    if reg.get("fork_structural") else "",
                "fork_value_recall":
                    round(reg.get("fork_value", {}).get("recall", 0.0), 4)
                    if reg.get("fork_value") else "",
                "asks_per_pass": round(bud.get("asks_per_pass", 0.0), 3)
                    if bud else "",
                "suppressed_refires": bud.get("suppressed_refires", "")
                    if bud else "",
                "n_runs": comp.get("n_runs", "") if comp else "",
                "turns_per_pass": round(comp.get("turns_per_pass", 0.0), 1)
                    if comp else "",
            })

    rows_for("ours", pass_summary, ask_by_arm)
    if bridge_ask or bridge_pass:
        rows_for("bridge", bridge_pass, bridge_ask)

    cols = list(csv_rows[0].keys()) if csv_rows else ["source", "arm"]
    lines = ["# Phase-4 headline table", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in csv_rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    lines += ["", f"pass@k = pass@{expected_passes}, ported semantics "
              "(src/xtid/harness/passk.py @352d14c). Ask-F1 recomputed with "
              "true registry counts on both sides (decisions/022 §3)."]
    return "\n".join(lines), csv_rows


def score(records: list[PassRecord], resolved: dict[str, bool],
          blockers_by_task: dict[str, list],
          fork_annotations: dict[str, str] | None,
          expected_passes: int, include_partial: bool = False,
          bridge_ask_logs: dict | None = None,
          bridge_rows: list[dict] | None = None) -> dict:
    """The one-call scoring entry (score_eval.py drives this)."""
    true_counts = {t: len(bs) for t, bs in blockers_by_task.items()}
    pass_summary = summarize_rows(our_rows(records, resolved),
                                  include_partial, expected_passes)
    ask = ask_metrics_by_arm(records, true_counts)
    regime = regime_recall_by_arm(records, blockers_by_task, fork_annotations)
    budgets = budget_by_arm(records)
    computes = compute_by_arm(records)
    bridge_ask = bridge_pass = None
    if bridge_ask_logs is not None:
        # bridge instance ids are canonical instance ids, not task dir names
        # -- index true counts under both
        iid_by_task = {r.task: r.instance_id for r in records}
        counts_by_orig = {}
        for task, bs in blockers_by_task.items():
            counts_by_orig[task] = len(bs)
            if task in iid_by_task:
                counts_by_orig[iid_by_task[task]] = len(bs)
        bridge_ask = bridge_ask_metrics(bridge_ask_logs, counts_by_orig)
    if bridge_rows:
        bridge_pass = summarize_rows(bridge_rows, include_partial,
                                     expected_passes)
    markdown, csv_rows = build_headline(pass_summary, ask, regime, budgets,
                                        computes, expected_passes,
                                        bridge_ask=bridge_ask,
                                        bridge_pass=bridge_pass)
    return {"markdown": markdown, "csv_rows": csv_rows,
            "pass_summary": pass_summary,
            "ask_by_arm": {a: vars(m) for a, m in ask.items()},
            "bridge_ask": {a: vars(m) for a, m in (bridge_ask or {}).items()},
            "regime_by_arm": regime, "budget_by_arm": budgets,
            "compute_by_arm": computes}
