"""Contract: the CIS context rebuild reproduces what the collector's model saw
(decisions/029 G0.1).

Three layers, cheapest first:
1. Docker-free, model-free: ReplaySession + run_agent yields one context per
   recorded segment, with the collector's own nag / TASK_DONE control flow,
   and the shared literals equal collect_v2's source.
2. Tokenizer-only (cached Qwen3 tokenizer; skipped if unavailable): on three
   REAL runs -- swe_0-s0 (clean), swe_0-s3 and swe_0-s7 (leaked </think>) --
   the template-rebuilt prompt equals the collector's add_generation_prompt
   render token-for-token at EVERY turn, the KV-branch pieces concatenate to
   it, and the history extension is a clean suffix.
3. Block-span mapping on a synthetic segment.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from wta.agent_loop import parse_action, run_agent, AgentLoopConfig
from wta.cis_context import (DELIBERATION_NUDGE, GEN_HEADER, INJECT_HEADER,
                             INSTRUCTION_SUFFIX, ReplaySession,
                             bash_block_token_span, build_instruction,
                             history_extension_ids, inject_into_last_user,
                             rebuild_contexts, render_turn)

REPO = Path(__file__).resolve().parents[2]
A0 = REPO / "data" / "a0_v3_32b"
TASKS = REPO / "third_party" / "hil-bench" / "harbor_swe"


class FakeEnv:
    def __init__(self):
        self.commands = []

    def execute(self, cmd):
        self.commands.append(cmd)
        return 0, f"ok output for: {cmd[:30]}"


TURNS = [
    "THOUGHT: look.\n```bash\nls lib\n```",
    "no block at all this turn",
    "THOUGHT: edit.\n```bash\nsed -i 's/a/b/' lib/x.py\n```",
    "```bash\necho TASK_DONE\n```",
]


def test_literals_pinned_to_collect_v2_source():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import collect_v2
    assert DELIBERATION_NUDGE == collect_v2.DELIBERATION_NUDGE
    src = inspect.getsource(collect_v2.main)
    # the suffix is assembled as two string literals in collect_v2; pin the
    # joined text against the source without refactoring the collector
    assert "You are working inside the repository this task" in src
    assert "refers to. Explore it with shell commands as needed." in src
    assert INSTRUCTION_SUFFIX.startswith("\n\nYou are working inside the repository")
    assert 'instruction += "\\n\\n" + DELIBERATION_NUDGE' in src


def test_replay_reproduces_control_flow_one_context_per_segment():
    env = FakeEnv()
    ctxs, res = rebuild_contexts(TURNS, "Fix it.", env, run_id="r0",
                                 task_id="t0", seed=0, temperature=0.7)
    assert len(ctxs) == len(TURNS)
    assert res.finished and res.stop_reason == "submit_marker"
    # the no-block turn produced the collector's nag as the next user turn
    assert ctxs[2][-1]["role"] == "user"
    assert ctxs[2][-1]["content"].startswith("Your reply had no ```bash block")
    # observations are user turns in the collector's exact format
    assert ctxs[1][-1]["content"].startswith("[exit 0]\nok output for: ls lib")
    assert ctxs[1][-1]["content"].endswith("\n\nNext step?")
    # commands executed = parse_action of block-bearing, non-submit turns
    assert env.commands == [parse_action(TURNS[0]), parse_action(TURNS[2])]
    # context k never contains segment k
    for k, c in enumerate(ctxs):
        assert all(m["content"] != TURNS[k] for m in c if m["role"] == "assistant")


def test_replay_mismatch_is_loud():
    env = FakeEnv()
    # TASK_DONE at turn 0 -> the loop stops after 1 segment, but 2 were recorded
    with pytest.raises(RuntimeError, match="consumed 1 of 2"):
        rebuild_contexts(["```bash\necho TASK_DONE\n```", "extra"], "x", env,
                         run_id="r", task_id="t", seed=0, temperature=0.7)


def test_inject_uses_stepper_merge_rule():
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    out = inject_into_last_user(m, "R")
    assert out[-1]["content"] == "u\n\n" + INJECT_HEADER + "R"
    assert m[-1]["content"] == "u"                       # input untouched
    out2 = inject_into_last_user(m + [{"role": "assistant", "content": "a"}], "R")
    assert out2[-1] == {"role": "user", "content": INJECT_HEADER + "R"}


@pytest.fixture(scope="module")
def qwen_tok():
    try:
        from wta.hf_reader import _load_tokenizer
        return _load_tokenizer("Qwen/Qwen3-32B")
    except Exception as e:  # pragma: no cover - offline / no cache
        pytest.skip(f"Qwen3 tokenizer unavailable: {type(e).__name__}")


def _real_run(run_id: str):
    task = run_id.split("-s")[0]
    segf = A0 / task / f"{run_id}.segments.json"
    logf = A0 / task / f"{run_id}.json"
    if not (segf.exists() and logf.exists() and (TASKS / task).exists()):
        pytest.skip(f"{run_id} not on this machine")
    segs = json.loads(segf.read_text(encoding="utf-8"))
    log = json.loads(logf.read_text(encoding="utf-8"))
    return task, segs, log


class _RecordedExitEnv:
    """Replays the RECORDED exit code with placeholder text, so the control
    flow and the '[exit N]' the model saw are reproduced without docker."""

    def __init__(self, log):
        self.es = {a["segment_idx"]: a["observables"].get("error_signature", "exit 0")
                   for a in log["actions"]}
        self.i = 0
        self.segs = sorted(self.es)

    def execute(self, cmd):
        seg = self.segs[self.i] if self.i < len(self.segs) else None
        self.i += 1
        code = 0
        if seg is not None:
            import re
            m = re.search(r"exit\s+(-?\d+)", self.es[seg])
            code = int(m.group(1)) if m else 0
        return code, "<observation supplied by replay>"


@pytest.mark.parametrize("run_id", ["swe_0-s0", "swe_0-s3", "swe_0-s7"])
def test_rebuilt_prompt_equals_collector_render_every_turn(qwen_tok, run_id):
    task, segs, log = _real_run(run_id)
    instr = build_instruction(TASKS / task, "baseline", nudge=True)
    ctxs, _ = rebuild_contexts(segs, instr, _RecordedExitEnv(log), run_id=run_id,
                               task_id=task, seed=log["seed"],
                               temperature=log["temperature"])
    assert len(ctxs) == len(segs)
    leak = sum("</think>" in s for s in segs)
    if run_id != "swe_0-s0":
        assert leak >= 1, "fixture chosen for its leaked </think>"
    prev_G = None
    for k, (m, seg) in enumerate(zip(ctxs, segs)):
        r = render_turn(qwen_tok, m, seg)          # raises on any drift
        ref = qwen_tok.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        assert r.prompt_ids == qwen_tok(ref)["input_ids"], f"turn {k}"
        assert r.prompt_ids[-len(r.header_ids):] == r.header_ids
        # the cache is only ever extended: G_{k+1} == G_k + extension, exactly
        if prev_G is not None:
            assert r.prefix_ids == prev_G, f"turn {k}: cache prefix != G_k"
        nxt = ctxs[k + 1][-1]["content"] if k + 1 < len(ctxs) else ""
        ext = history_extension_ids(qwen_tok, m, seg, nxt)
        prev_G = r.prefix_ids + ext
        # the segment ids ARE the block-scored ids; span inside them
        if r.block_span is not None:
            t0, t1 = r.block_span
            assert 0 <= t0 < t1 <= len(r.seg_ids)
            assert "```bash" in qwen_tok.decode(r.seg_ids[t0:t1])
    assert GEN_HEADER.endswith("</think>\n\n")


def test_bash_block_span_on_synthetic_segment(qwen_tok):
    seg = "THOUGHT: two blocks.\n```bash\nls\n```\nand then\n```bash\ncat << EOF > f.py\nx = 1\nEOF\n```\n"
    span = bash_block_token_span(qwen_tok, seg)
    assert span is not None
    ids = qwen_tok(seg, add_special_tokens=False)["input_ids"]
    text = qwen_tok.decode(ids[span[0]:span[1]])
    assert "cat << EOF" in text and "ls" not in text.split("```bash")[-1].split("cat")[0]
    assert bash_block_token_span(qwen_tok, "no block here") is None
