"""v2 agent loop: multi-step tool-calling trajectories with per-turn reads --
OURS, loop convention adapted from mini-swe-agent (third_party/mini-swe-agent
@ 531dbaf, MIT; one bash block per turn, executed, observation appended);
the activation reading, segment logging, and action-observable extraction
are the contribution (decisions/017).

Ground rule intact: reads happen DURING each turn's generation (cadence + cue
+ value triggers); the ActionEvent log is an offline TEACHER for labels --
nothing here uses actions to decide when to read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from wta.logging_schema import ActionEvent, RunLog

SYSTEM_PROMPT = """You are a software engineering agent working in a real repository via a shell.

Rules:
- Each reply: first a short THOUGHT section reasoning about what to do next \
(weigh alternatives where the task leaves choices open), then EXACTLY ONE \
shell command in a ```bash fenced block. Nothing after the block.
- The command runs in the repo; its output is returned to you.
- Inspect before you change. Make your code changes with shell commands \
(e.g. applying a patch via a heredoc, sed, or writing files with cat).
- When the task is fully complete, run exactly: echo TASK_DONE
"""

_BASH_BLOCK = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)
_FILE_TOKEN = re.compile(
    r"[\w./~-]+\.(?:py|go|ts|tsx|js|jsx|json|yaml|yml|md|txt|cfg|toml|sh|c|h|"
    r"cpp|rs|java|scss|css|cue)\b")


@dataclass
class AgentLoopConfig:
    max_steps: int = 15
    max_new_tokens_per_turn: int = 1024
    obs_head: int = 1500          # observation truncation (head/tail chars)
    obs_tail: int = 500
    submit_marker: str = "TASK_DONE"
    temperature: float = 0.7


def parse_action(text: str) -> str | None:
    """The turn's command: the LAST ```bash block (mini-swe-agent convention)."""
    blocks = _BASH_BLOCK.findall(text)
    return blocks[-1].strip() if blocks else None


def extract_file_observables(command: str) -> list[str]:
    """File paths the action touches -- offline label observables only."""
    return sorted(set(_FILE_TOKEN.findall(command)))


def truncate_obs(out: str, head: int, tail: int) -> str:
    if len(out) <= head + tail + 60:
        return out
    return (out[:head] + f"\n... [{len(out) - head - tail} chars truncated] ...\n"
            + out[-tail:])


@dataclass
class AgentRunResult:
    log: RunLog
    segments: list[str]           # generated text per turn (labeling needs these)
    n_steps: int = 0
    finished: bool = False        # saw the submit marker
    stop_reason: str = ""
    commands: list[str] = field(default_factory=list)


def run_agent(session, env, instruction: str, *, run_id: str, task_id: str,
              seed: int, cfg: AgentLoopConfig, model_id: str = "",
              mid_layer: int = 0, layers: list[int] | None = None) -> AgentRunResult:
    """Drive one trajectory. `session.generate_segment(messages, seed=...,
    temperature=..., max_new_tokens=..., segment_idx=...) -> (reads, text)`;
    `env.execute(cmd) -> (exit_code, output)`. Both fakeable in tests."""
    log = RunLog(run_id=run_id, task_id=task_id, seed=seed,
                 temperature=cfg.temperature, model_id=model_id,
                 mid_layer=mid_layer, layers=layers)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction}]
    result = AgentRunResult(log=log, segments=[])

    for step in range(cfg.max_steps):
        reads, text = session.generate_segment(
            messages, seed=seed, temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens_per_turn, segment_idx=step)
        log.reads.extend(reads)
        result.segments.append(text)
        messages.append({"role": "assistant", "content": text})
        result.n_steps = step + 1

        cmd = parse_action(text)
        last_tok = reads[-1].token_idx if reads else 0
        if cmd is None:
            messages.append({"role": "user", "content":
                             "Your reply had no ```bash block. Reply with a "
                             "THOUGHT and exactly one ```bash block."})
            continue
        result.commands.append(cmd)
        log.actions.append(ActionEvent(
            token_idx=last_tok, segment_idx=step, action_text=cmd,
            observables={"files": extract_file_observables(cmd), "step": step}))

        if cfg.submit_marker in cmd:
            result.finished = True
            result.stop_reason = "submit_marker"
            break
        code, out = env.execute(cmd)
        obs = truncate_obs(out, cfg.obs_head, cfg.obs_tail)
        messages.append({"role": "user", "content":
                         f"[exit {code}]\n{obs}\n\nNext step?"})
    else:
        result.stop_reason = "max_steps"

    log.validate()
    return result
