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

# --- composite decision-identity observables (spec A0; method doc's labeling
# note: "Do not label a decision by the file alone -- that conflates two
# decisions that touch the same file. Use a composite label: file + code
# region/span + stated sub-goal + error signature.") Recorded at COLLECTION
# time because none of it can be reconstructed afterwards (decisions/021 §4).

# line spans: sed ranges ('12s/', '10,20d', -n '5,15p') and patch hunk headers
_SED_SPAN = re.compile(r"['\"]?(\d+)(?:\s*,\s*(\d+))?\s*[a-z]")
_HUNK = re.compile(r"@@\s*-(\d+)(?:,(\d+))?")
_ERR_LINE = re.compile(
    r"^.*?((?:[A-Z]\w*Error|[A-Z]\w*Exception|Traceback|FAILED|fatal|"
    r"command not found|No such file|undefined|cannot find)\b.*)$",
    re.MULTILINE)


def extract_region(command: str) -> list[str]:
    """Line spans the action touches, as 'start-end' strings. Distinguishes two
    edits to the SAME file at different places -- the conflation the method doc
    warns about. Empty when the command carries no line information."""
    spans: list[str] = []
    for m in _HUNK.finditer(command):
        start = int(m.group(1))
        spans.append(f"{start}-{start + int(m.group(2) or 1) - 1}")
    if not spans and ("sed" in command or "awk" in command):
        for m in _SED_SPAN.finditer(command):
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            spans.append(f"{a}-{b}")
    return sorted(set(spans))[:8]


def extract_subgoal(text: str, limit: int = 400) -> str:
    """The turn's stated sub-goal: the THOUGHT prose before the bash block.
    This is where a model verbalizes WHICH open choice it is resolving, so it
    is the highest-value composite field for decision identity."""
    head = text.split("```bash")[0] if "```bash" in text else text
    return _WS_RUN.sub(" ", head).strip()[:limit]


def error_signature(exit_code: int, output: str) -> str:
    """Normalized failure fingerprint of the observation an action produced:
    'exit <code>' plus the most specific error line. Part of the composite key
    because the same file+region under two different failures is usually two
    different decisions.

    "Most specific" matters: a Python failure starts with a bare 'Traceback'
    line and ends with the typed exception, so first-match would fingerprint
    every crash identically. Prefer a typed 'Name: message' line."""
    sig = f"exit {exit_code}"
    hits = [_WS_RUN.sub(" ", m.group(1)).strip()
            for m in _ERR_LINE.finditer(output or "")]
    if not hits:
        return sig
    typed = [h for h in hits if ":" in h and not h.lower().startswith("traceback")]
    return sig + " | " + (typed[-1] if typed else hits[0])[:160]


_WS_RUN = re.compile(r"\s+")


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
        event = ActionEvent(
            token_idx=last_tok, segment_idx=step, action_text=cmd,
            observables={"files": extract_file_observables(cmd), "step": step,
                         "region": extract_region(cmd),
                         "subgoal": extract_subgoal(text)})
        log.actions.append(event)

        if cfg.submit_marker in cmd:
            result.finished = True
            result.stop_reason = "submit_marker"
            break
        code, out = env.execute(cmd)
        # the signature is only knowable after execution -- fill it in on the
        # event we already logged (observables is mutable by design)
        event.observables["error_signature"] = error_signature(code, out)
        obs = truncate_obs(out, cfg.obs_head, cfg.obs_tail)
        messages.append({"role": "user", "content":
                         f"[exit {code}]\n{obs}\n\nNext step?"})
    else:
        result.stop_reason = "max_steps"

    log.validate()
    return result
