"""Turn-granular agent stepper with ask/answer injection (spec eval,
decisions/022).

``wta.agent_loop.run_agent`` is a closed loop; the eval orchestrator needs
turn-granular control to run N trajectories in lockstep and inject judge
answers between turns. ``TurnStepper`` re-expresses the SAME protocol --
every constant and extractor is imported from ``wta.agent_loop``, nothing is
copied, and a contract test pins turn-for-turn equivalence against
``run_agent`` on identical fakes. ``run_agent`` itself stays untouched.

Injection modes:
  * ``inject_user_message(text)`` -- queued; lands before the next
    generation (detector-initiated answers, injected into all N runs).
  * ``ask_affordance=True`` -- the system prompt gains an ask_human
    paragraph and ``step()`` intercepts ``ask_human "..."`` commands BEFORE
    the env: the question surfaces on ``StepOutcome.asked``, the caller
    judges it and MUST call ``answer_ask()`` before the next ``step()``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from wta.agent_loop import (
    SYSTEM_PROMPT, AgentLoopConfig, AgentRunResult, error_signature,
    extract_file_observables, extract_region, extract_subgoal, parse_action,
    truncate_obs,
)
from wta.logging_schema import ActionEvent, RunLog

# Register mirrors the vendored ask config's system template ("A human
# reviewer is available via the ask_human tool... Each question should be
# about one specific ambiguity."), adapted to the one-bash-block protocol.
ASK_AFFORDANCE_PARAGRAPH = """
- A human reviewer is available. If the task leaves a genuinely ambiguous \
choice you cannot resolve from the repository, you may ask by running exactly: \
ask_human "your one specific question" \
as your single bash command. The reviewer's answer is returned as the \
command output. Each question should be about one specific ambiguity.
"""

_ASK_CMD = re.compile(r"^\s*ask_human\s+(?:\"(.+?)\"|'(.+?)')\s*$", re.DOTALL)

_NO_BLOCK_NAG = ("Your reply had no ```bash block. Reply with a "
                 "THOUGHT and exactly one ```bash block.")


def parse_ask(command: str) -> str | None:
    """The question text iff the command is an ask_human invocation."""
    m = _ASK_CMD.match(command or "")
    if not m:
        return None
    return (m.group(1) or m.group(2)).strip()


@dataclass
class StepOutcome:
    reads: list
    text: str
    command: str | None
    exit_code: int | None
    observation: str | None
    finished: bool
    asked: str | None = None    # intercepted ask_human question, if any


class TurnStepper:
    def __init__(self, session, env, instruction: str, *, run_id: str,
                 task_id: str, seed: int, cfg: AgentLoopConfig,
                 ask_affordance: bool = False, model_id: str = "",
                 mid_layer: int = 0, layers: list[int] | None = None):
        self.session, self.env, self.cfg = session, env, cfg
        self.seed = seed
        self.ask_affordance = ask_affordance
        system = SYSTEM_PROMPT + (ASK_AFFORDANCE_PARAGRAPH if ask_affordance else "")
        self.messages = [{"role": "system", "content": system},
                         {"role": "user", "content": instruction}]
        self.log = RunLog(run_id=run_id, task_id=task_id, seed=seed,
                          temperature=cfg.temperature, model_id=model_id,
                          mid_layer=mid_layer, layers=layers)
        self._result = AgentRunResult(log=self.log, segments=[])
        self._step_idx = 0
        self._pending_injections: list[str] = []
        self._pending_ask_event: ActionEvent | None = None
        self._done = False

    # -- injection ----------------------------------------------------------

    def inject_user_message(self, text: str) -> None:
        """Queue a user-turn message (e.g. a human answer). Lands before the
        next generation; merged into the trailing user message so the chat
        keeps strict role alternation."""
        self._pending_injections.append(text)

    def _drain_injections(self) -> None:
        for text in self._pending_injections:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages[-1]["content"] += "\n\n" + text
            else:
                self.messages.append({"role": "user", "content": text})
        self._pending_injections.clear()

    def answer_ask(self, answer: str) -> str:
        """Feed the judge's reply back as the intercepted ask_human command's
        observation (the in-container tool prints the server response
        verbatim -- same shape here). Returns the observation text."""
        if self._pending_ask_event is None:
            raise RuntimeError("answer_ask() with no pending ask")
        self._pending_ask_event.observables["error_signature"] = (
            error_signature(0, answer))
        self._pending_ask_event = None
        obs = truncate_obs(answer, self.cfg.obs_head, self.cfg.obs_tail)
        msg = f"[exit 0]\n{obs}\n\nNext step?"
        self.messages.append({"role": "user", "content": msg})
        return msg

    # -- the turn -----------------------------------------------------------

    def done(self) -> bool:
        return self._done

    def step(self) -> StepOutcome:
        if self._done:
            raise RuntimeError("step() after done")
        if self._pending_ask_event is not None:
            raise RuntimeError("step() with an unanswered ask_human pending")
        self._drain_injections()

        step = self._step_idx
        reads, text = self.session.generate_segment(
            self.messages, seed=self.seed, temperature=self.cfg.temperature,
            max_new_tokens=self.cfg.max_new_tokens_per_turn, segment_idx=step)
        self.log.reads.extend(reads)
        self._result.segments.append(text)
        self.messages.append({"role": "assistant", "content": text})
        self._step_idx += 1
        self._result.n_steps = self._step_idx

        cmd = parse_action(text)
        last_tok = reads[-1].token_idx if reads else 0
        if cmd is None:
            self.messages.append({"role": "user", "content": _NO_BLOCK_NAG})
            self._maybe_exhaust()
            return StepOutcome(reads, text, None, None, None, False)

        self._result.commands.append(cmd)
        event = ActionEvent(
            token_idx=last_tok, segment_idx=step, action_text=cmd,
            observables={"files": extract_file_observables(cmd), "step": step,
                         "region": extract_region(cmd),
                         "subgoal": extract_subgoal(text)})
        self.log.actions.append(event)

        if self.cfg.submit_marker in cmd:
            self._result.finished = True
            self._result.stop_reason = "submit_marker"
            self._done = True
            return StepOutcome(reads, text, cmd, None, None, True)

        if self.ask_affordance:
            question = parse_ask(cmd)
            if question is not None:
                # never reaches the env; caller judges then answer_ask()
                self._pending_ask_event = event
                self._maybe_exhaust()
                return StepOutcome(reads, text, cmd, None, None, False,
                                   asked=question)

        code, out = self.env.execute(cmd)
        event.observables["error_signature"] = error_signature(code, out)
        obs = truncate_obs(out, self.cfg.obs_head, self.cfg.obs_tail)
        self.messages.append({"role": "user",
                              "content": f"[exit {code}]\n{obs}\n\nNext step?"})
        self._maybe_exhaust()
        return StepOutcome(reads, text, cmd, code, obs, False)

    def _maybe_exhaust(self) -> None:
        if self._step_idx >= self.cfg.max_steps:
            self._result.stop_reason = "max_steps"
            self._done = True

    def result(self) -> AgentRunResult:
        self.log.validate()
        return self._result
