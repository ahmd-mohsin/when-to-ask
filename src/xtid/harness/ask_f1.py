"""Ask-F1 -- MIGRATED from HiL-Bench.

Source: third_party/hil-bench/hil_bench/utils/compute_hil_metrics.py and the per-instance
logging + per-blocker cap in ask_human_server.py (commit 352d14c).

  precision = blockers_discovered / questions_asked
  recall    = blockers_discovered / blockers_present
  Ask-F1    = harmonic mean of the two (global over all instances)

The harmonic mean is what punishes question-spam (brief S5f hazard 1): high recall via
many questions tanks precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .judge import IRRELEVANT_QUESTION, JudgeResult

# Upstream cap: only two answered hits per blocker; the 3rd+ matched question for the
# same blocker is forced to "irrelevant question".
MAX_ANSWERED_QUESTIONS_PER_BLOCKER = 2


@dataclass
class InstanceLog:
    n_blockers: int
    questions: list[dict] = field(default_factory=list)  # {question, response, blocker_name}
    blockers: dict[str, bool] = field(default_factory=dict)


class AskSession:
    """Accumulates ask_human interactions across instances; produces Ask-F1.

    Mirrors the upstream GLOBAL_LOGS bookkeeping, including the per-blocker answered cap.
    """

    def __init__(self) -> None:
        self.logs: dict[str, InstanceLog] = {}

    def start_instance(self, instance_id: str, blocker_ids: list[str]) -> None:
        self.logs[instance_id] = InstanceLog(
            n_blockers=len(blocker_ids), blockers={b: False for b in blocker_ids}
        )

    def record(self, instance_id: str, question: str, result: JudgeResult) -> str:
        log = self.logs[instance_id]
        blocker_name = result.blocker_id
        response = result.response
        if blocker_name is not None:
            prior_hits = sum(1 for q in log.questions if q.get("blocker_name") == blocker_name)
            if prior_hits >= MAX_ANSWERED_QUESTIONS_PER_BLOCKER:
                blocker_name, response = None, IRRELEVANT_QUESTION
            else:
                log.blockers[blocker_name] = True
        log.questions.append(
            {"question": question, "response": response, "blocker_name": blocker_name}
        )
        return response

    def metrics(self) -> "GlobalMetrics":
        return compute_hil_metrics(self.logs)


@dataclass
class GlobalMetrics:
    precision: float
    recall: float
    ask_f1: float
    n_questions: int
    n_blockers_present: int
    n_blockers_discovered: int


def compute_hil_metrics(logs: dict[str, InstanceLog]) -> GlobalMetrics:
    """Verbatim port of HiL-Bench compute_hil_metrics (global precision/recall/F1)."""
    total_questions = 0
    total_blockers_present = 0
    total_blockers_discovered = 0

    for log in logs.values():
        questions = log.questions
        n_discovered = len({name for e in questions if (name := e["blocker_name"]) is not None})
        n_questions = len(questions)
        total_questions += n_questions
        total_blockers_present += log.n_blockers
        total_blockers_discovered += n_discovered

    precision = total_blockers_discovered / total_questions if total_questions > 0 else 0.0
    recall = (
        total_blockers_discovered / total_blockers_present if total_blockers_present > 0 else 0.0
    )
    ask_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return GlobalMetrics(
        precision=precision,
        recall=recall,
        ask_f1=ask_f1,
        n_questions=total_questions,
        n_blockers_present=total_blockers_present,
        n_blockers_discovered=total_blockers_discovered,
    )
