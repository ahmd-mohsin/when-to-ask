"""Question assembly: AskDecision -> one judge-rubric-shaped question (spec
eval, decisions/022 §2f).

Backbone-LLM-phrased (ClarifyGPT precedent; HiL-Bench's own ask arm is
model-phrased) with a deterministic template fallback so the trigger can
never crash on a bad generation. The frozen judge rubric is the acceptance
target: ONE question, single topic, a question word, no embedded
assumptions/observations/background.

LEAK RULE (structural): this module never sees the blocker registry -- the
signature takes only the fired decision and the task statement, so registry
text (descriptions, resolutions, example_questions) cannot enter the
phrasing prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_QUESTION_WORDS = ("what", "why", "how", "where", "when", "which")
_REQUEST_VERBS = ("define", "explain", "describe", "differentiate", "specify",
                  "clarify")
# seed-mixing offset for phrasing calls so they never collide with a
# trajectory turn's (seed, segment_idx) stream
_PHRASING_SEGMENT_BASE = 10_000

_STOPWORDS = frozenset(
    "the and for with from into that this should would could use using file "
    "files echo cat sed awk grep git diff apply patch python bash run make "
    "then else done".split())


@dataclass
class AssembledQuestion:
    text: str
    method: str                  # "llm" | "template"
    options_used: list[str] = field(default_factory=list)
    prompt: str = ""             # phrasing prompt, kept for the audit trail


def _valid_question(text: str) -> bool:
    t = (text or "").strip()
    if not (10 <= len(t) <= 300) or t.count("?") != 1 or not t.endswith("?"):
        return False
    low = t.lower()
    return any(w in low for w in _QUESTION_WORDS + _REQUEST_VERBS)


def _option_texts(decision) -> list[str]:
    seen, out = set(), []
    for opt in decision.options:
        text = (opt.get("action_text") or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) >= 4:
            break
    return out


def _topic_hint(options: list[str]) -> str:
    counts: dict[str, int] = {}
    for text in options:
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text):
            tok = tok.lower()
            if tok not in _STOPWORDS:
                counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts, key=lambda t: (-counts[t], t))[:3]
    return " ".join(ranked)


def _template_question(options: list[str]) -> str:
    hint = _topic_hint(options)
    if hint:
        return f"Which behaviour should the change to {hint} implement?"
    return "Which interpretation of the task requirements should I follow?"


def _build_prompt(statement: str, options: list[str]) -> list[dict]:
    lines = [
        "A software task left an ambiguous decision, and different attempts "
        "committed to different readings of it.",
        "",
        "Task (excerpt):",
        (statement or "").strip()[:1200],
        "",
        "The divergent committed actions:",
    ]
    for i, text in enumerate(options):
        lines.append(f"- reading {chr(65 + i)}: {text[:200]}")
    lines += [
        "",
        "Write ONE short clarification question for the task author that "
        "resolves which reading is intended. Requirements: a single question "
        "about a single specific topic; it must contain a question word "
        "(what/why/how/where/when/which); do not embed assumptions, "
        "observations, option lists, or background; do not mention file "
        "names or shell commands; end with a question mark. Reply with the "
        "question only.",
    ]
    return [{"role": "user", "content": "\n".join(lines)}]


def _extract_candidate(text: str) -> str:
    lines = [ln.strip().strip('"').strip() for ln in (text or "").splitlines()
             if ln.strip()]
    for ln in reversed(lines):
        if ln.endswith("?"):
            return ln
    return lines[-1] if lines else ""


def assemble_question(decision, task_statement: str, *, session=None,
                      seed: int = 0, max_new_tokens: int = 96,
                      temperature: float = 0.3) -> AssembledQuestion:
    """`decision` is an ``wta.online.AskDecision``; `session` (optional) is
    duck-typed on ``generate_segment`` -- the shared HFStreamReader at eval
    time, a fake in tests, None for the pure-template path."""
    options = _option_texts(decision)
    if session is not None:
        messages = _build_prompt(task_statement, options)
        try:
            _, text = session.generate_segment(
                messages, seed=seed, temperature=temperature,
                max_new_tokens=max_new_tokens,
                segment_idx=_PHRASING_SEGMENT_BASE + int(decision.bucket_id))
            cand = _extract_candidate(text)
            if _valid_question(cand):
                return AssembledQuestion(cand, "llm", options,
                                         messages[0]["content"])
        except Exception:
            pass  # fall through to the deterministic template
    return AssembledQuestion(_template_question(options), "template", options)
