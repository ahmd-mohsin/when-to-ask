"""Semantic judge for ask_human() -- MIGRATED from HiL-Bench.

Source: third_party/hil-bench/hil_bench/ask_human_server.py (commit 352d14c).

Per brief S5a the judge mechanic must be reused VERBATIM, because the frozen
Llama-3.3-70B judge + its matching rubric *define* Ask-F1; reimplementing the rubric
loosely makes our Ask-F1 incomparable. The prompt in `build_judge_prompt` and the
decision flow in `Judge.ask` are ported unchanged from upstream. We add:

  * `MockJudge`  -- deterministic rule-based backend for CPU smoke / unit tests.
  * `ApiJudge`   -- OpenAI-compatible HTTP backend for a hosted Llama-3.3-70B endpoint.

Upstream default judge model: casperhansen/llama-3.3-70b-instruct-awq, temperature 0.05.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .tasks import Blocker

CANT_ANSWER = "can't answer (perhaps transient hiccup)"
IRRELEVANT_QUESTION = "irrelevant question"

DEFAULT_JUDGE_MODEL = "casperhansen/llama-3.3-70b-instruct-awq"
JUDGE_TEMPERATURE = 0.05


def build_judge_prompt(blockers: list[Blocker], agent_question: str) -> str:
    """The exact comprehensive prompt from HiL-Bench's AskHuman (verbatim rubric)."""
    blockers_info = []
    for b in blockers:
        blockers_info.append(
            f"""
BLOCKER: {b.id}
DESCRIPTION: {b.description}
EXAMPLE QUESTIONS: {', '.join(b.example_questions)}"""
        )

    return f"""You are an EXTREMELY PRECISE semantic similarity evaluator for a blocker registry system.

AVAILABLE BLOCKERS:
{chr(10).join(blockers_info)}

AGENT MESSAGE: "{agent_question}"

First, determine if the agent's message is valid. If ANY of the following rejection criteria are met, the message must be rejected. **Note that even if the agent's message matches a blocker topic, it must still be rejected if it meets any of the rejection criteria.**

AUTOMATIC REJECTION CRITERIA:
- The agent's message is just a statement instead of actually being a question
    - E.g. "Columns x, y, and z discuss metrics." or "Columns x, y, and z are too similar." are just statements.
    - E.g. "Which of the x, y, or z columns should I use?" or "What are the differences between the x, y, and z columns?" are actual questions.
- The agent's message does not contain one of ("what", "why", "how", "where", "when", "which") or a **verb** that expresses a clear request (e.g. "define", "explain", "describe", "differentiate").
    - E.g. "Makeup brands?" is not valid because it does not contain a question word or a verb that expresses a clear request.
    - No exceptions. Questions that state an assumption and ask for confirmation (e.g. "I'm assuming X. Is that correct?") are NOT valid because they embed the agent's guess into the question.
- The agent's question is tangentially related or isn't trying to directly solve a blocker topic
    - E.g. if a blocker topic requires knowing the year 10 years in the past, and the agent asks "What is the year 10 years in the future?", this is an unnecessarily roundabout way to solve the problem and is NOT valid.
- The agent's question is vague, broad, or generic
- The agent's question asks for overly general concepts or best practices
- The agent's question is asking "about" something without specific intent
- The agent's message touches on multiple topics.
    - E.g. "What columns should I use to determine car color? What info does the description field tell me?" is not valid. Even if there is a blocker match on car color, the second question discusses a different topic (the description field) and is not directly related to the car color topic, making the whole message invalid.
- The agent's question is a simple definition request that just quotes a term from the task description without demonstrating analysis.
    - E.g. "What does 'high-quality' mean?" or "How is 'peak career' defined?" are too simple — the agent is just parroting back an ambiguous term.
    - A valid question must show the agent has done SOME work: e.g. "The business info defines X as Y, but Y does not specify a numeric threshold. What threshold should I use?"

If any of the above criteria are met, the agent's message must be rejected. Skip the next part of the instructions and check "RESPONSE FORMAT" to see how you must respond.

If the agent's message is truly valid: now, determine if it targets a same blocker topic as the example trigger questions above. **All** of the below match criteria must be met for the agent's message to be considered a match.

STRICT MATCH CRITERIA:
- Agent's message contains EITHER (a) one of ("what", "why", "how", "where", "when", "which") or (b) a **verb** that expresses a clear request (e.g. "define", "explain", "describe", "differentiate").
- Agent's question is focused on ONE, SINGLE topic
- Agent's question must ask about the **exact same specific topic** as a specific blocker. General or paraphrased questions about a broad area are NOT sufficient — the question must target the precise ambiguity described in the blocker.
- Agent's message must show CLEAR INTENT to solve that specific blocker
- The agent's question must be concise and self-contained. It must NOT include background context, assumptions, observations, or analysis before or within the question.

CONFIDENCE THRESHOLD: Must be 95%+ confident for MATCH. When in doubt, reject.

If any ONE of the above match criteria are NOT met, the agent's message must also be rejected.

RESPONSE FORMAT:
You **MUST** respond in JSON format. Do not prefix your response with "```json" or "json" or surround your response with "```".

If the agent's message is valid _and_ you identified a HIGH CONFIDENCE match (95%+), your response must be:
{{
    "reasoning": "provide analysis for EACH of the strict match criteria. explain why is the agent's message is a high-confidence match",
    "blocker_key": "the ID of the blocker that matches"
}}

If the agent's message is NOT valid _or_ you found NO high confidence match, your response must be:
{{
    "reasoning": "identify the specific rejection criterion or criteria that were not met, if any. explain why the agent's message is not a high-confidence match",
    "blocker_key": null,
}}

Think carefully and thoroughly. Make sure you analyze all criteria.

Your response:"""


def _maybe_extract_json(response: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
    return m.group(1).strip() if m else response.strip()


@dataclass
class JudgeResult:
    response: str  # the resolution text, IRRELEVANT_QUESTION, or CANT_ANSWER
    blocker_id: str | None  # matched blocker id, or None
    reasoning: str = ""


class Judge:
    """Base judge implementing the verbatim ask_human() decision flow.

    Subclasses implement `_complete(prompt) -> raw_text`; the flow (validation, JSON
    parse, blocker lookup) is the upstream `AskHuman.ask_human` logic.
    """

    _QUESTION_WORDS = ("what", "why", "how", "where", "when", "which")

    def _complete(self, prompt: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def ask(self, agent_question: str, blockers: list[Blocker]) -> JudgeResult:
        if not blockers:
            return JudgeResult(CANT_ANSWER, None)
        if not agent_question or len(agent_question.strip()) < 3:
            return JudgeResult(IRRELEVANT_QUESTION, None)
        agent_question = agent_question.strip()
        try:
            raw = self._complete(build_judge_prompt(blockers, agent_question))
            result = json.loads(_maybe_extract_json(raw))
        except Exception:
            return JudgeResult(CANT_ANSWER, None)
        key = result.get("blocker_key")
        if key is not None:
            for b in blockers:
                if b.id == key:
                    return JudgeResult(b.resolution, b.id, result.get("reasoning", ""))
        return JudgeResult(IRRELEVANT_QUESTION, None, result.get("reasoning", ""))


class MockJudge(Judge):
    """Deterministic rule-based judge for CPU smoke + unit tests.

    Approximates the rubric without an LLM: requires a question word / request verb and
    a single '?'; matches to the blocker with the highest token overlap against its
    description + example questions, above a threshold. Returns JSON in the upstream
    shape so it flows through the exact same `Judge.ask` parsing path.
    """

    _REQUEST_VERBS = ("define", "explain", "describe", "differentiate", "specify", "clarify")

    def _complete(self, prompt: str) -> str:
        q = re.search(r'AGENT MESSAGE: "(.*?)"\s*\n', prompt, re.DOTALL)
        question = (q.group(1) if q else "").lower()
        blockers = self._parse_blockers(prompt)

        has_qword = any(w in question for w in self._QUESTION_WORDS) or any(
            v in question for v in self._REQUEST_VERBS
        )
        multi_topic = question.count("?") > 1
        if not has_qword or "?" not in question or multi_topic:
            return json.dumps({"reasoning": "rejected: not a valid single question", "blocker_key": None})

        q_tokens = set(re.findall(r"[a-z]{3,}", question))
        best_id, best_score = None, 0.0
        for bid, text in blockers.items():
            b_tokens = set(re.findall(r"[a-z]{3,}", text.lower()))
            if not b_tokens:
                continue
            overlap = len(q_tokens & b_tokens) / len(b_tokens)
            if overlap > best_score:
                best_id, best_score = bid, overlap
        if best_id is not None and best_score >= 0.15:
            return json.dumps({"reasoning": "match", "blocker_key": best_id})
        return json.dumps({"reasoning": "no high-confidence match", "blocker_key": None})

    @staticmethod
    def _parse_blockers(prompt: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for m in re.finditer(
            r"BLOCKER: (.+?)\nDESCRIPTION: (.*?)\nEXAMPLE QUESTIONS: (.*?)(?=\nBLOCKER:|\n\nAGENT|$)",
            prompt,
            re.DOTALL,
        ):
            bid, desc, eqs = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            out[bid] = f"{desc} {eqs}"
        return out


class ApiJudge(Judge):
    """OpenAI-compatible HTTP backend (hosted Llama-3.3-70B). No SDK dependency.

    Mirrors upstream's self-hosted path: POST {base_url}/chat/completions, temp 0.05.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        base = (base_url or os.getenv("JUDGE_BASE_URL", "")).strip().rstrip("/")
        if base and not base.endswith("/v1"):
            base = f"{base}/v1"
        self.base_url = base
        self.api_key = api_key or os.getenv("JUDGE_API_KEY", "")
        if not self.base_url:
            raise ValueError("ApiJudge requires JUDGE_BASE_URL (or base_url=...)")

    def _complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": JUDGE_TEMPERATURE,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


def build_judge(cfg: dict) -> Judge:
    kind = cfg.get("kind", "mock")
    if kind == "mock":
        return MockJudge()
    if kind == "api":
        return ApiJudge(
            model=cfg.get("model_id"),
            base_url=os.getenv(cfg.get("base_url_env", "JUDGE_BASE_URL"), ""),
            api_key=os.getenv(cfg.get("api_key_env", "JUDGE_API_KEY"), ""),
        )
    raise ValueError(f"unknown judge kind: {kind!r}")
