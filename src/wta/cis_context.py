"""CIS context rebuild -- OURS (decisions/029).

The collector never persisted the model's context (only its generated
segments, `<run>.segments.json`; decisions/028:478-480). Counterfactual
Information Sensitivity needs `ctx_k`, the exact message list the model saw
before generating segment k. This module rebuilds it by driving
`wta.agent_loop.run_agent` ITSELF with a session that replays the stored
segments and an environment that re-executes the recorded commands in the
task container -- so `truncate_obs`, the no-bash-block nag, `parse_action`
and the TASK_DONE stop are the collector's own code, not a copy.

Two facts this module is built around, both verified on real runs before it
was written:

1. Qwen3's chat template emits the empty ``<think>\\n\\n</think>\\n\\n`` block
   ONLY on the generation prompt, never on assistant turns rendered as
   history -- and it STRIPS a leaked ``</think>`` from history (146 segments
   across 102 runs carry one). So the per-turn prompt is
   ``apply_chat_template(messages_k, add_generation_prompt=False) + GEN_HEADER``
   -- rebuilt from the template each turn -- and NOT a manual string
   concatenation of the previous turn (that fails 19/50 and 11/50 on the leak
   runs swe_0-s3 / swe_0-s7). `render_turn` asserts equality with the
   collector's own `add_generation_prompt=True` render every time.
2. The token boundary between history-through-assistant_{k-1} (G_k) and the
   last user turn falls on special tokens, so G_k, the user suffix and the
   header tokenize independently and concatenate to the collector's prompt
   ids. `render_turn` asserts that too; it is what lets a KV cache hold G_k
   and branch per injection.
3. G_k is obtained by SPLITTING H_k at its last ``<|im_start|>user\\n`` --
   never by rendering ``messages_k[:-1]``. The template renders an assistant
   message that is the LAST message after the final user query WITH the
   think block (it is treated as the answer in progress), but the same
   message BEFORE a user turn without it. So ``render(messages_k[:-1])`` is
   not a prefix of ``H_k`` (found by the contract test on the first real
   run). The split is verified exactly: the suffix must equal
   ``<|im_start|>user\\n`` + content + ``<|im_end|>\\n``.

Nothing here imports torch. The tokenizer is passed in.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path

from wta.agent_env import DockerTaskEnv, _run
from wta.agent_loop import _BASH_BLOCK, AgentLoopConfig, run_agent

# --- literals shared with scripts/collect_v2.py ---------------------------
# collect_v2 lives in scripts/ and is not importable from the library. These
# are VERBATIM copies, pinned by harness/contract/test_cis_context_rebuild.py
# against collect_v2's source so they cannot drift.
INSTRUCTION_SUFFIX = ("\n\nYou are working inside the repository this task "
                      "refers to. Explore it with shell commands as needed.")
DELIBERATION_NUDGE = (
    "Where the task leaves a choice open, state the alternatives you see in "
    "your THOUGHT, pick one, and proceed with it. Do not ask for "
    "clarification; commit to your reading and implement it."
)
# What the collector's apply_chat_template(add_generation_prompt=True,
# enable_thinking=False) appends after the rendered history (Qwen3).
GEN_HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
# 029 frozen injection wrapper: mirrors hil-bench's own full_info header
# (src/wta/eval/fullinfo.py:_HEADER) and is appended with the stepper's merge
# rule (src/wta/eval/stepper.py:_drain_injections).
INJECT_HEADER = ("The following clarification is provided by the task author "
                 "to help you complete this task:\n\n")
OBS_HEAD, OBS_TAIL = 1500, 500          # AgentLoopConfig defaults, R2 as-run
MAX_NEW_TOKENS = 2048                   # R2 as-run
CAPTURE_CAP_BYTES = 4 * 1024 * 1024     # per stream; Amendment D was 723 MB


def build_instruction(task_dir: str | Path, mode: str = "baseline",
                      nudge: bool = True) -> str:
    """The user[0] message exactly as collect_v2 built it (R2: nudge ON)."""
    text = (Path(task_dir) / mode / "instruction.md").read_text(
        encoding="utf-8", errors="replace")
    text += INSTRUCTION_SUFFIX
    if nudge:
        text += "\n\n" + DELIBERATION_NUDGE
    return text


class ReplaySession:
    """A `session` for run_agent that returns the RECORDED segment for each
    turn and snapshots the message list it was handed -- that snapshot IS
    ctx_k. Produces no reads (activations come from the fresh forward)."""

    def __init__(self, segments: list[str]):
        self.segments = list(segments)
        self.contexts: list[list[dict]] = []

    def generate_segment(self, messages, *, seed, temperature, max_new_tokens,
                         segment_idx):
        if segment_idx >= len(self.segments):
            raise IndexError(f"replay asked for segment {segment_idx} of "
                             f"{len(self.segments)}")
        self.contexts.append([dict(m) for m in messages])
        return [], self.segments[segment_idx]


_RC = re.compile(r"__WTA_RC__:(-?\d+)")


@dataclass
class ObservationEnv(DockerTaskEnv):
    """DockerTaskEnv that returns the command's OUTPUT (the two replay scripts
    discard it in-container to seal the Amendment D blow-up; CIS needs it).

    Output is redirected to files inside the container and read back with a
    byte cap, so a runaway command cannot flood the host. The exit code is
    read from a marker echoed AFTER the redirected group, so it is the
    command's own status (a pipe into `head` would substitute the pipe's).
    Concatenation order (stdout then stderr) matches `_run`, so the observation
    string is what the collector would have built.

    `last_capture` records byte sizes and whether either stream hit the cap;
    above the cap the `truncate_obs` marker's digit count is approximate
    (flag `marker_approx`) -- the one bounded infidelity 029 records.
    """

    cap_bytes: int = CAPTURE_CAP_BYTES

    def __post_init__(self):
        self.last_capture: dict = {}

    def execute(self, command: str) -> tuple[int, str]:
        if not self.started:
            raise RuntimeError("env not started")
        code, out = _run(
            ["docker", "exec", self.name, "sh", "-lc",
             f"cd {self.workdir} 2>/dev/null; {{ {command}\n}} "
             f">/tmp/.wta_o 2>/tmp/.wta_e; echo __WTA_RC__:$?"],
            timeout=self.exec_timeout)
        if code == 124:
            # verbatim what the collector's _run returned on a timeout
            self.last_capture = {"timeout": True}
            return 124, out
        m = _RC.search(out or "")
        rc = int(m.group(1)) if m else code
        _, sizes = _run(["docker", "exec", self.name, "sh", "-lc",
                         "wc -c </tmp/.wta_o; wc -c </tmp/.wta_e"], timeout=30)
        try:
            n_out, n_err = (int(x) for x in sizes.split()[:2])
        except ValueError:
            n_out = n_err = -1
        _, so = _run(["docker", "exec", self.name, "sh", "-lc",
                      f"head -c {self.cap_bytes} /tmp/.wta_o"], timeout=120)
        _, se = _run(["docker", "exec", self.name, "sh", "-lc",
                      f"head -c {self.cap_bytes} /tmp/.wta_e"], timeout=120)
        capped = (n_out > self.cap_bytes) or (n_err > self.cap_bytes)
        self.last_capture = {"stdout_bytes": n_out, "stderr_bytes": n_err,
                             "capped": capped, "marker_approx": capped}
        return rc, (so or "") + (se or "")


def rebuild_contexts(segments: list[str], instruction: str, env, *,
                     run_id: str, task_id: str, seed: int,
                     temperature: float):
    """Drive run_agent over the recorded segments. Returns
    (contexts, AgentRunResult): contexts[k] is the message list the model saw
    before generating segments[k]. Raises if the loop did not consume every
    recorded segment (the recorded run and the replay disagree on control
    flow -- a TASK_DONE or nag path mismatch)."""
    sess = ReplaySession(segments)
    cfg = AgentLoopConfig(max_steps=len(segments),
                          max_new_tokens_per_turn=MAX_NEW_TOKENS,
                          obs_head=OBS_HEAD, obs_tail=OBS_TAIL,
                          temperature=temperature)
    res = run_agent(sess, env, instruction, run_id=run_id, task_id=task_id,
                    seed=seed, cfg=cfg)
    if len(sess.contexts) != len(segments):
        raise RuntimeError(f"{run_id}: replay consumed {len(sess.contexts)} of "
                           f"{len(segments)} recorded segments")
    return sess.contexts, res


def inject_into_last_user(messages_k: list[dict], text: str) -> list[dict]:
    """The stepper's merge rule (src/wta/eval/stepper.py:_drain_injections):
    append to the trailing user turn, else add one. Returns a new list."""
    m = [dict(x) for x in messages_k]
    if m and m[-1]["role"] == "user":
        m[-1]["content"] = m[-1]["content"] + "\n\n" + INJECT_HEADER + text
    else:
        m.append({"role": "user", "content": INJECT_HEADER + text})
    return m


def bash_block_token_span(tokenizer, segment: str) -> tuple[int, int] | None:
    """[t0, t1) token span of the LAST ```bash block (parse_action's block),
    on the segment re-tokenized with add_special_tokens=False -- the same ids
    `render_turn` scores. None when the segment has no block."""
    from wta.labeling import token_char_positions
    ms = list(_BASH_BLOCK.finditer(segment))
    if not ms:
        return None
    m = ms[-1]
    starts = token_char_positions(segment, tokenizer)
    if not starts:
        return None
    t0 = max(bisect.bisect_right(starts, m.start()) - 1, 0)
    t1 = bisect.bisect_left(starts, m.end())
    return t0, max(t1, t0 + 1)


USER_OPEN = "<|im_start|>user\n"
TURN_CLOSE = "<|im_end|>\n"


def split_history(H: str, last_user_content: str) -> tuple[str, str]:
    """(G, user_suffix): H split at its last user turn, verified exactly."""
    idx = H.rfind(USER_OPEN)
    if idx < 0:
        raise RuntimeError("render drift: no user turn in history render")
    G, suffix = H[:idx], H[idx:]
    expect = USER_OPEN + last_user_content + TURN_CLOSE
    if suffix != expect:
        raise RuntimeError("render drift: last user turn did not render as "
                           "<|im_start|>user\\n{content}<|im_end|>\\n")
    return G, suffix


@dataclass
class RenderedTurn:
    prefix_ids: list[int]        # G_k: history through assistant_{k-1}
    user_suffix_ids: list[int]   # the last user turn, rendered
    header_ids: list[int]        # GEN_HEADER
    seg_ids: list[int]           # the recorded segment (no specials)
    block_span: tuple[int, int] | None   # in seg_ids
    prompt_len: int              # len(prefix + user_suffix + header)

    @property
    def prompt_ids(self) -> list[int]:
        return self.prefix_ids + self.user_suffix_ids + self.header_ids


def render_turn(tokenizer, messages_k: list[dict], segment_k: str,
                enable_thinking: bool = False) -> RenderedTurn:
    """Tokenize ctx_k + segment_k exactly as the collector did, split at the
    KV-branch point. Asserts (a) the template-rebuilt prompt equals the
    collector's add_generation_prompt render, (b) the three pieces tokenize
    independently to the collector's prompt ids."""
    kw = {"enable_thinking": enable_thinking}
    H = tokenizer.apply_chat_template(messages_k, tokenize=False,
                                      add_generation_prompt=False, **kw)
    ref = tokenizer.apply_chat_template(messages_k, tokenize=False,
                                        add_generation_prompt=True, **kw)
    prompt = H + GEN_HEADER
    if prompt != ref:
        raise RuntimeError("render drift: history + GEN_HEADER != collector's "
                           "generation-prompt render")
    if messages_k[-1]["role"] != "user":
        raise RuntimeError("ctx_k must end with a user turn")
    G, user_suffix = split_history(H, messages_k[-1]["content"])
    prompt_ids = tokenizer(prompt)["input_ids"]          # the collector's call
    g_ids = tokenizer(G)["input_ids"]
    us_ids = tokenizer(user_suffix, add_special_tokens=False)["input_ids"]
    hdr_ids = tokenizer(GEN_HEADER, add_special_tokens=False)["input_ids"]
    if g_ids + us_ids + hdr_ids != prompt_ids:
        raise RuntimeError("boundary tokenization drift: G + user + header "
                           "!= collector prompt ids")
    seg_ids = tokenizer(segment_k, add_special_tokens=False)["input_ids"]
    return RenderedTurn(prefix_ids=g_ids, user_suffix_ids=us_ids,
                        header_ids=hdr_ids, seg_ids=seg_ids,
                        block_span=bash_block_token_span(tokenizer, segment_k),
                        prompt_len=len(prompt_ids))


def history_extension_ids(tokenizer, messages_k: list[dict], segment_k: str,
                          next_user_content: str = "",
                          enable_thinking: bool = False) -> list[int]:
    """Tokens that take the cache from G_k to G_{k+1}: the last user turn plus
    assistant_k rendered AS HISTORY (no think header; a leaked </think> is
    stripped by the template). The history form only appears when a user
    turn FOLLOWS assistant_k, so messages_{k+1} is rendered with the real
    next user content when known (else a placeholder -- it cannot change how
    assistant_k renders) and split at that user turn. Asserted to be a clean
    token suffix of G_{k+1} on top of G_k."""
    kw = {"enable_thinking": enable_thinking}
    H_k = tokenizer.apply_chat_template(messages_k, tokenize=False,
                                        add_generation_prompt=False, **kw)
    G_k, _ = split_history(H_k, messages_k[-1]["content"])
    nxt = messages_k + [{"role": "assistant", "content": segment_k},
                        {"role": "user", "content": next_user_content}]
    H_next = tokenizer.apply_chat_template(nxt, tokenize=False,
                                           add_generation_prompt=False, **kw)
    G_next, _ = split_history(H_next, next_user_content)
    if not G_next.startswith(G_k):
        raise RuntimeError("render drift: G_k is not a prefix of G_{k+1}")
    ext = tokenizer(G_next[len(G_k):], add_special_tokens=False)["input_ids"]
    if tokenizer(G_k)["input_ids"] + ext != tokenizer(G_next)["input_ids"]:
        raise RuntimeError("boundary tokenization drift on history extension")
    return ext
