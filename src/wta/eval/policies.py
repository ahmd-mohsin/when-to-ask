"""Eval arm trigger policies (spec eval, decisions/022).

One policy instance per (arm, task, pass). The orchestrator drives the
protocol: ``on_reads``/``on_turn`` per run turn, ``poll`` at each round
boundary (returns at most one AskRequest per call; drained until None),
``on_answer`` after the judge replies. Single-trajectory arms (no_ask,
full_info, model_initiated) need no policy -- the arm spec flags cover them.

Every asking policy dedups its own trigger key (once per detected fork,
decisions/022 §2b); re-fires are counted in ``stats`` for the interruption
budget, never re-asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from wta.eval.question import assemble_question
from wta.online import AskDecision
from xtid.signals.output_divergence import output_divergence
from xtid.signals.probe import dp_ask_score

_WS = re.compile(r"\s+")
_NUM = re.compile(r"(\d{1,3}(?:\.\d+)?)")

# seed-mixing segment base for B2 confidence elicitations (distinct from
# trajectory turns and from question phrasing's 10_000 base)
_ELICIT_SEGMENT_BASE = 20_000


@dataclass
class TaskContext:
    """What a policy may see: the runs' public state and the shared session
    for phrasing/elicitation. NEVER the blocker registry (leak rule)."""
    task_id: str
    statement: str
    steppers: dict            # run_id -> TurnStepper
    session: object | None    # shared model session; None => template questions
    seed: int = 0


@dataclass
class AskRequest:
    question: str
    key: str                  # dedup key ("bucket:3", "b1:...", ...)
    bucket_id: int | None = None
    meta: dict = field(default_factory=dict)


class TriggerPolicy:
    """Base: never asks; tracks last commands (question-material only)."""

    name = "no_ask"

    def __init__(self):
        self.ctx: TaskContext | None = None
        self._last_cmd: dict = {}
        self._asked_keys: set = set()
        self.stats = {"asks": 0, "suppressed_refires": 0, "fired_keys": []}

    def bind(self, ctx: TaskContext) -> None:
        self.ctx = ctx

    def on_reads(self, run_id, reads) -> None:
        pass

    def on_turn(self, run_id, outcome) -> None:
        if outcome.command:
            self._last_cmd[run_id] = outcome.command

    def poll(self) -> AskRequest | None:
        return None

    def on_answer(self, req: AskRequest, judge_result) -> None:
        pass

    # -- helpers for subclasses --------------------------------------------

    def _mark_asked(self, key: str) -> bool:
        """True if the key was fresh (caller should ask); False = suppressed."""
        if key in self._asked_keys:
            self.stats["suppressed_refires"] += 1
            return False
        self._asked_keys.add(key)
        self.stats["asks"] += 1
        self.stats["fired_keys"].append(key)
        return True

    def _pseudo_decision(self, commands: list[str]) -> AskDecision:
        return AskDecision(
            bucket_id=0, runs=list(self._last_cmd),
            options=[{"run_id": f"o{i}", "r": np.zeros(1), "weight": 1.0,
                      "action_text": c} for i, c in enumerate(commands)],
            looping_runs=[], spread=0.0, cusum=0.0)

    def _question_from_commands(self, commands: list[str]) -> str:
        q = assemble_question(self._pseudo_decision(commands),
                              self.ctx.statement if self.ctx else "",
                              session=self.ctx.session if self.ctx else None,
                              seed=self.ctx.seed if self.ctx else 0)
        return q.text


class DetectorPolicy(TriggerPolicy):
    """Ours: the Part B trigger over live activation reads."""

    name = "detector"

    def __init__(self, runtime):
        super().__init__()
        self._task_detector = runtime.new_task()   # fresh per (task, pass)
        self._pending: dict[int, AskDecision] = {}

    def on_reads(self, run_id, reads) -> None:
        for read in reads:
            decision = self._task_detector.observe_read(run_id, read.h)
            if decision is None:
                continue
            if f"bucket:{decision.bucket_id}" in self._asked_keys:
                self.stats["suppressed_refires"] += 1
            else:
                self._pending[decision.bucket_id] = decision

    def on_turn(self, run_id, outcome) -> None:
        super().on_turn(run_id, outcome)
        if outcome.command:
            self._task_detector.register_action(run_id, outcome.command)
        if outcome.observation is not None:
            state = f"{outcome.exit_code}:{outcome.observation[:200]}"
            self._task_detector.notify_env_state(run_id, state)

    def poll(self) -> AskRequest | None:
        while self._pending:
            bucket_id, decision = self._pending.popitem()
            if not self._mark_asked(f"bucket:{bucket_id}"):
                continue
            q = assemble_question(decision, self.ctx.statement,
                                  session=self.ctx.session, seed=self.ctx.seed)
            return AskRequest(q.text, key=f"bucket:{bucket_id}",
                              bucket_id=bucket_id,
                              meta={"spread": decision.spread,
                                    "cusum": decision.cusum,
                                    "method": q.method,
                                    "options": q.options_used})
        return None

    def on_answer(self, req: AskRequest, judge_result) -> None:
        if req.bucket_id is not None:
            self._task_detector.inject_resolution(req.bucket_id)


class OutputDivergencePolicy(TriggerPolicy):
    """B1 (ClarifyGPT, load-bearing baseline): cross-run signature divergence.

    Signature source at eval time: the normalized latest committed command
    per run (behavioral proxy; ClarifyGPT's original signature is the test
    OUTPUT vector, which needs the executor -- a GPU-time option, pluggable
    via ``signature_fn``)."""

    name = "output_divergence"

    def __init__(self, signature_fn=None, min_runs: int = 2):
        super().__init__()
        self._sig_fn = signature_fn or (lambda cmd: _WS.sub(" ", cmd).strip())
        self._min_runs = min_runs

    def poll(self) -> AskRequest | None:
        sigs = [self._sig_fn(c) for c in self._last_cmd.values()]
        if len(sigs) < self._min_runs:
            return None
        div = output_divergence(sigs)
        if not div["ambiguous"]:
            return None
        key = "b1:" + "|".join(sorted(set(sigs)))[:400]
        if not self._mark_asked(key):
            return None
        # one representative command per distinct signature
        rep: dict[str, str] = {}
        for cmd in self._last_cmd.values():
            rep.setdefault(self._sig_fn(cmd), cmd)
        return AskRequest(self._question_from_commands(list(rep.values())),
                          key=key, meta=div)


class ProbePolicy(TriggerPolicy):
    """B3 (OPENIA): frozen linear probe on mid-layer reads, aggregated over N."""

    name = "probe"

    def __init__(self, probe, threshold: float, layer_pos: int | None = None,
                 agg: str = "mean"):
        super().__init__()
        self._probe, self._threshold, self._agg = probe, threshold, agg
        self._layer_pos = layer_pos
        self._last_h: dict = {}

    def on_reads(self, run_id, reads) -> None:
        if reads:
            h = np.asarray(reads[-1].h, dtype=np.float32)
            if h.ndim == 2:
                if self._layer_pos is None:
                    raise ValueError("multi-layer h but no layer_pos configured")
                h = h[self._layer_pos]
            self._last_h[run_id] = h

    def poll(self) -> AskRequest | None:
        if not self._last_h:
            return None
        H = np.stack(list(self._last_h.values()))
        score = dp_ask_score(self._probe.proba_incorrect(H), agg=self._agg)
        if score <= self._threshold or not self._mark_asked("b3"):
            return None
        return AskRequest(self._question_from_commands(list(self._last_cmd.values())),
                          key="b3", meta={"score": float(score)})


class VerbalizedPolicy(TriggerPolicy):
    """B2: elicit a 0-100 confidence from each live run every ``every``
    rounds; ask once when 1 - mean(conf) crosses the threshold."""

    name = "verbalized"

    def __init__(self, threshold: float = 0.5, every: int = 3):
        super().__init__()
        self._threshold, self._every = threshold, every
        self._round = 0

    def _elicit(self, stepper, run_seed: int) -> float:
        msgs = list(stepper.messages) + [{
            "role": "user",
            "content": ("On a scale of 0 to 100, how confident are you that "
                        "you know the single correct next action without "
                        "needing to ask a clarifying question? Answer with "
                        "just the number.")}]
        try:
            _, text = self.ctx.session.generate_segment(
                msgs, seed=run_seed, temperature=0.0, max_new_tokens=8,
                segment_idx=_ELICIT_SEGMENT_BASE + self._round)
        except Exception:
            return 0.5
        m = _NUM.search(text or "")
        return float(np.clip(float(m.group(1)) / 100.0, 0, 1)) if m else 0.5

    def poll(self) -> AskRequest | None:
        self._round += 1
        if self.ctx is None or self.ctx.session is None:
            return None
        if self._round % self._every != 0 or "b2" in self._asked_keys:
            return None
        live = [(rid, st) for rid, st in self.ctx.steppers.items()
                if not st.done()]
        if not live:
            return None
        confs = [self._elicit(st, self.ctx.seed + i)
                 for i, (_, st) in enumerate(live)]
        score = 1.0 - float(np.mean(confs))
        if score < self._threshold or not self._mark_asked("b2"):
            return None
        return AskRequest(self._question_from_commands(list(self._last_cmd.values())),
                          key="b2", meta={"score": score, "confs": confs})


class RandomPolicy(TriggerPolicy):
    """B4: uniformly random asks, budget-matched to the detector arm's
    measured asks/task (decisions/022 §2i -- B4 is scored last, its budget
    read from the detector arm's ask counts only)."""

    name = "random"

    def __init__(self, budget: float, expected_rounds: int, seed: int = 0):
        super().__init__()
        self._budget = float(budget)
        self._p = min(1.0, self._budget / max(1, expected_rounds))
        self._rng = np.random.default_rng(seed)
        self._round = 0

    def poll(self) -> AskRequest | None:
        self._round += 1
        if self.stats["asks"] >= int(np.ceil(self._budget)):
            return None
        if self._rng.random() >= self._p:
            return None
        key = f"b4:round{self._round}"
        if not self._mark_asked(key):
            return None
        return AskRequest(self._question_from_commands(list(self._last_cmd.values())),
                          key=key)
