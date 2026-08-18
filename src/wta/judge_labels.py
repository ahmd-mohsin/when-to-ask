"""Judge-sourced commitment labels (spec judge_labels.md, decisions/025).

A SECOND labelling teacher for the (run, decision) pairs where the frozen
signature lexicon abstained ("no signature hits"). The judge is an offline
LLM (claude-fable-5 via the Anthropic Message Batches API) whose output is
frozen to a JSONL artifact; build_labels() consults that artifact — never a
live model — so the builder stays deterministic given its inputs.

A judge response becomes a label only if its cited evidence span is
verbatim-locatable in the run's raw trace (auditability gate, spec §4).
Network code (submit/poll/fetch) lazy-imports the `anthropic` SDK; everything
else is CPU-only stdlib so contract tests never need the SDK.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .labeling import _is_mutating, _norm, load_class_artifact

JUDGE_MODEL = "claude-fable-5"           # decisions/025 §5 (owner's choice)
JUDGE_MAX_TOKENS = 3000                  # caps thinking + text on this model
JUDGE_EFFORT = "low"                     # classification-shaped task
DEFAULT_MIN_CONF = 0.7                   # primary labelled arm (spec §7)

MAX_TRACE_CHARS = 30_000                 # spec §5: full trace at/below this
HEAD_CHARS = 2_000
TAIL_CHARS = 8_000
ANCHOR_WINDOW = 1_000
MAX_ANCHOR_WINDOWS = 8
ELISION = "\n\n[... trace elided ...]\n\n"
MAX_EVIDENCE_CHARS = 300

DEFAULT_REGISTRIES = "third_party/hil-bench/harbor_swe"
_REGISTRY_REL = Path("shared") / "ask-human-data" / "blocker_registry.json"

SYSTEM_PROMPT = """\
You are labelling which interpretation an autonomous coding agent's completed
run COMMITTED to, for one pre-specified ambiguous decision (a "blocker").
You are given the decision's description, topic anchors, the candidate
interpretation classes, and the agent's run trace.

Respond with JSON only (no prose outside it):
{"class": "<one class name exactly as listed, or null>",
 "confidence": <number 0.0-1.0>,
 "evidence": "<verbatim quote copied character-for-character from the trace>",
 "reasoning": "<one or two sentences>"}

Rules:
- Commitment = what the run's edits, commands, or final work product actually
  implement or assert. Deliberation that merely MENTIONS an option is not
  commitment; prefer evidence from mutating actions (file edits, sed/patch/
  redirection commands) or the final answer over discussion.
- If the run never engages this decision, or the evidence is ambiguous or
  contradictory, return "class": null. An unlabeled item is better than a
  mislabeled one.
- "evidence" must be copied VERBATIM from the trace (at most 300 characters).
  A label whose evidence cannot be located in the trace is discarded.
- The class list order is randomized; position carries no information.
- Indicator phrases are examples, not an exhaustive lexicon: a run can commit
  to a class without using any indicator phrase, and using a phrase in
  passing deliberation does not by itself mean commitment.
"""


@dataclass
class JudgeItem:
    custom_id: str
    run_id: str
    task: str
    blocker: str
    description: str
    anchors: list[str]
    class_names: list[str]          # artifact order
    signatures: list[list[str]]     # artifact order, parallel to class_names
    perm: list[int]                 # presentation order (artifact indices)
    excerpt: str
    policy: str                     # "full" | "excerpt"
    trace_chars: int
    meta: dict = field(default_factory=dict)


def class_permutation(run_id: str, blocker: str, n: int) -> list[int]:
    """Deterministic per-item shuffle of class presentation order (spec §3):
    kills position bias toward the canonical class (always artifact index 0)
    while staying reproducible."""
    seed = int.from_bytes(
        hashlib.sha256(f"{run_id}|{blocker}".encode()).digest()[:8], "big")
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    return perm


def _blocker_maps(art: dict) -> tuple[dict[str, str], dict[str, dict]]:
    """blocker id -> task, blocker id -> spec. Blocker ids are globally
    unique (pinned by harness/contract/test_fork_type_annotations.py)."""
    task_of, spec_of = {}, {}
    for task in sorted(k for k in art if not k.startswith("_")):
        for bid, spec in art[task].items():
            if bid in task_of:
                raise ValueError(f"blocker id not globally unique: {bid}")
            task_of[bid], spec_of[bid] = task, spec
    return task_of, spec_of


def _registry_descriptions(registries_dir: Path, task: str) -> dict[str, str]:
    reg = registries_dir / task / _REGISTRY_REL
    if not reg.exists():
        return {}
    data = json.loads(reg.read_text(encoding="utf-8"))
    # ONLY the description travels to the judge: `resolution` names the
    # intended answer and would bias toward the canonical class (spec §3).
    return {b["id"]: (b.get("description") or "").strip()
            for b in data.get("blockers", [])}


def _merge_intervals(spans: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    spans = sorted((max(0, a), min(n, b)) for a, b in spans if b > 0 and a < n)
    out: list[list[int]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def make_excerpt(text: str, anchors: list[str],
                 mut_action_texts: list[str]) -> tuple[str, str]:
    """Spec §5 trace-window policy. Deterministic; evidence verification
    always runs against the FULL raw text regardless of the policy used."""
    n = len(text)
    if n <= MAX_TRACE_CHARS:
        return text, "full"
    low = text.lower()
    spans: list[tuple[int, int]] = [(0, HEAD_CHARS), (n - TAIL_CHARS, n)]
    for a in mut_action_texts:
        key = a[:120]
        p = text.find(key)
        if p >= 0:
            spans.append((p - 200, p + max(len(a), 200) + 200))
    hits = 0
    for term in anchors:
        t = term.lower()
        start = 0
        while hits < MAX_ANCHOR_WINDOWS:
            p = low.find(t, start)
            if p < 0:
                break
            spans.append((p - ANCHOR_WINDOW, p + len(t) + ANCHOR_WINDOW))
            start = p + len(t)
            hits += 1
    merged = _merge_intervals(spans, n)
    # keep within budget: drop middle windows (never head/tail) until it fits
    while (sum(b - a for a, b in merged)
           + len(ELISION) * max(0, len(merged) - 1)) > MAX_TRACE_CHARS \
            and len(merged) > 2:
        mid = max(range(1, len(merged) - 1),
                  key=lambda i: merged[i][1] - merged[i][0])
        merged = merged[:mid] + merged[mid + 1:]
    return ELISION.join(text[a:b] for a, b in merged), "excerpt"


def _mut_action_texts(a0_dir: Path, task: str, run_id: str) -> list[str]:
    jf = a0_dir / task / f"{run_id}.json"
    if not jf.exists():
        return []
    log = json.loads(jf.read_text(encoding="utf-8"))
    return [a.get("action_text", "") for a in log.get("actions", [])
            if _is_mutating(a.get("action_text", ""))]


def unlabeled_pairs(debug_path: str | Path,
                    include_ties: bool = False) -> list[tuple[str, str]]:
    """(run_id, blocker) pairs the lexicon teacher left unlabeled. Iterates
    the file line-by-line: snippets contain U+2028/U+2029, splitlines() would
    shear rows (generate_labels.py caveat)."""
    reasons = {"no signature hits"}
    if include_ties:
        reasons.add("tie between top classes")
    pairs = []
    for line in Path(debug_path).open(encoding="utf-8"):
        row = json.loads(line)
        if (row.get("kind") == "commitment" and not row.get("chosen")
                and row.get("reason") in reasons):
            pairs.append((row["run"], row["blocker"]))
    return sorted(set(pairs))


def build_judge_items(a0_dir: str | Path, classes_path: str | Path,
                      pairs: list[tuple[str, str]],
                      registries_dir: str | Path = DEFAULT_REGISTRIES,
                      ) -> list[JudgeItem]:
    """One JudgeItem per (run_id, blocker) pair. Same builder serves the
    production run and the validation gate (spec §6: identical prompts)."""
    a0_dir, registries_dir = Path(a0_dir), Path(registries_dir)
    art = load_class_artifact(classes_path)
    task_of, spec_of = _blocker_maps(art)
    desc_cache: dict[str, dict[str, str]] = {}
    items: list[JudgeItem] = []
    for k, (run_id, blocker) in enumerate(sorted(pairs)):
        task = task_of[blocker]
        spec = spec_of[blocker]
        txt_path = a0_dir / task / f"{run_id}.txt"
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        if task not in desc_cache:
            desc_cache[task] = _registry_descriptions(registries_dir, task)
        excerpt, policy = make_excerpt(
            text, spec["anchors"], _mut_action_texts(a0_dir, task, run_id))
        names = [c["name"] for c in spec["classes"]]
        items.append(JudgeItem(
            custom_id=f"i{k:05d}",
            run_id=run_id, task=task, blocker=blocker,
            description=desc_cache[task].get(blocker, ""),
            anchors=list(spec["anchors"]),
            class_names=names,
            signatures=[list(c["signatures"]) for c in spec["classes"]],
            perm=class_permutation(run_id, blocker, len(names)),
            excerpt=excerpt, policy=policy, trace_chars=len(text)))
    return items


def build_judge_prompt(item: JudgeItem) -> tuple[str, str]:
    """(system, user). Class order follows item.perm; the canonical flag and
    the registry resolution are structurally absent (spec §3)."""
    lines = [
        f"TASK: {item.task}",
        f"DECISION (blocker id): {item.blocker}",
        "DESCRIPTION: " + (item.description
                           or "(no registry description available)"),
        "TOPIC ANCHORS (terms identifying this decision's topic): "
        + "; ".join(item.anchors),
        "",
        "CANDIDATE INTERPRETATION CLASSES (order randomized):",
    ]
    for rank, art_idx in enumerate(item.perm, start=1):
        sigs = "; ".join(f'"{s}"' for s in item.signatures[art_idx])
        lines.append(f"{rank}. name: {item.class_names[art_idx]}")
        lines.append(f"   indicators: {sigs}")
    lines += [
        "",
        f"AGENT RUN TRACE ({item.run_id}"
        + (", excerpted" if item.policy == "excerpt" else "") + "):",
        "<<<TRACE",
        item.excerpt,
        "TRACE>>>",
    ]
    return SYSTEM_PROMPT, "\n".join(lines)


def make_batch_requests(items: list[JudgeItem], model: str = JUDGE_MODEL,
                        max_tokens: int = JUDGE_MAX_TOKENS,
                        effort: str = JUDGE_EFFORT) -> list[dict]:
    reqs = []
    for it in items:
        system, user = build_judge_prompt(it)
        reqs.append({"custom_id": it.custom_id,
                     "params": {"model": model, "max_tokens": max_tokens,
                                "system": system,
                                "output_config": {"effort": effort},
                                "messages": [{"role": "user",
                                              "content": user}]}})
    return reqs


def parse_judge_response(text: str) -> dict | None:
    """Tolerates ```json fences and surrounding prose; requires a 'class'
    key. Returns {'class': str|None, 'confidence': float, 'evidence': str,
    'reasoning': str} or None on parse failure."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        if s.startswith("json"):
            s = s[4:]
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(s[a:b + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "class" not in obj:
        return None
    cls = obj.get("class")
    if cls is not None and not isinstance(cls, str):
        return None
    try:
        conf = float(obj.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return {"class": cls, "confidence": max(0.0, min(1.0, conf)),
            "evidence": str(obj.get("evidence") or ""),
            "reasoning": str(obj.get("reasoning") or "")}


def _norm_with_map(s: str) -> tuple[str, list[int]]:
    out, idx, prev_space = [], [], True
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def locate_evidence(trace: str, evidence: str) -> int:
    """Raw char offset of the evidence span in the FULL trace, or -1.
    Exact match first; whitespace-normalized fallback (mirrors labeling._norm
    semantics) mapped back to raw offsets. No repair beyond that: an
    unlocatable span rejects the label (spec §4)."""
    ev = evidence.strip()
    if not ev:
        return -1
    p = trace.find(ev)
    if p >= 0:
        return p
    tn, tmap = _norm_with_map(trace)
    en = _norm(ev).strip()
    if not en:
        return -1
    q = tn.find(en)
    return tmap[q] if q >= 0 else -1


# ---------------------------------------------------------------- freezing

def freeze_results(items: list[JudgeItem], responses: dict[str, dict],
                   a0_dir: str | Path) -> list[dict]:
    """One frozen record per item (spec §4). `responses` maps custom_id ->
    {'type': 'succeeded'|..., 'stop_reason': ..., 'text': ...} as written by
    fetch_batch_results()."""
    a0_dir = Path(a0_dir)
    out = []
    for it in items:
        rec = {"run": it.run_id, "task": it.task, "blocker": it.blocker,
               "status": None, "class": None, "confidence": None,
               "evidence": None, "commit_char": -1, "reasoning": None,
               "policy": it.policy, "custom_id": it.custom_id}
        resp = responses.get(it.custom_id)
        if resp is None:
            rec["status"] = "missing"
        elif resp.get("type") != "succeeded":
            rec["status"] = f"api_{resp.get('type', 'error')}"
        elif resp.get("stop_reason") == "refusal":
            rec["status"] = "refusal"
        else:
            parsed = parse_judge_response(resp.get("text", ""))
            if parsed is None:
                rec["status"] = "parse_error"
            else:
                rec["confidence"] = parsed["confidence"]
                rec["reasoning"] = parsed["reasoning"][:500]
                rec["evidence"] = parsed["evidence"][:MAX_EVIDENCE_CHARS + 100]
                if parsed["class"] is None:
                    rec["status"] = "abstained"
                elif parsed["class"] not in it.class_names:
                    rec["status"] = "bad_class"
                    rec["class"] = parsed["class"]
                else:
                    text = (a0_dir / it.task / f"{it.run_id}.txt").read_text(
                        encoding="utf-8", errors="replace")
                    pos = locate_evidence(text, parsed["evidence"])
                    if pos < 0:
                        rec["status"] = "evidence_not_found"
                        rec["class"] = parsed["class"]
                    else:
                        rec["status"] = "accepted"
                        rec["class"] = parsed["class"]
                        rec["commit_char"] = pos
        out.append(rec)
    return out


def load_judge_labels(path: str | Path, min_conf: float = DEFAULT_MIN_CONF,
                      ) -> dict[tuple[str, str], tuple[str, int, float]]:
    """Accepted labels at/above the confidence threshold, keyed
    (run_id, blocker) -> (class_name, commit_char, confidence) — the shape
    build_labels(judge_labels=...) consumes."""
    out = {}
    for line in Path(path).open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") == "accepted" and (r.get("confidence") or 0) >= min_conf:
            out[(r["run"], r["blocker"])] = (
                r["class"], int(r["commit_char"]), float(r["confidence"]))
    return out


# ------------------------------------------- session transport (025 Am. A)
# The operative path: Claude Fable 5 subagents inside the owner's Claude
# Code session label the items. Items are grouped into per-subagent work
# files by prompt-size budget; each subagent returns schema-forced JSON,
# collected into a results JSONL whose records session_responses() converts
# into the same shape fetch_batch_results() produces, so freeze_results()
# is transport-independent.

SESSION_CHAR_BUDGET = 150_000
SESSION_MAX_ITEMS = 12


def write_item_files(items: list[JudgeItem], out_dir: str | Path,
                     char_budget: int = SESSION_CHAR_BUDGET,
                     max_items: int = SESSION_MAX_ITEMS) -> list[Path]:
    """Greedy in custom_id order (deterministic): a new file when adding the
    next item would exceed char_budget or max_items. Each file carries the
    full frozen prompts -- the subagent adds nothing of its own."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    batch: list[dict] = []
    used = 0

    def flush():
        nonlocal batch, used
        if not batch:
            return
        p = out_dir / f"judge_work_{len(files):03d}.json"
        p.write_text(json.dumps({"items": batch}, ensure_ascii=False,
                                indent=0), encoding="utf-8")
        files.append(p)
        batch, used = [], 0

    for it in items:
        system, user = build_judge_prompt(it)
        size = len(system) + len(user)
        if batch and (used + size > char_budget or len(batch) >= max_items):
            flush()
        batch.append({"custom_id": it.custom_id, "system": system,
                      "user": user, "class_names": it.class_names})
        used += size
    flush()
    return files


def session_responses(records: list[dict]) -> dict[str, dict]:
    """Session results ({custom_id, class, confidence, evidence, reasoning})
    -> the {custom_id: response-record} shape freeze_results() consumes."""
    out = {}
    for r in records:
        out[r["custom_id"]] = {
            "custom_id": r["custom_id"], "type": "succeeded",
            "stop_reason": "end_turn",
            "text": json.dumps({"class": r.get("class"),
                                "confidence": r.get("confidence"),
                                "evidence": r.get("evidence"),
                                "reasoning": r.get("reasoning")},
                               ensure_ascii=False)}
    return out


# ------------------------------------------------------- batch plumbing
# Alternative transport (unused while the owner has no API key); lazy SDK
# import so CPU tests never need `anthropic`.

def submit_batches(requests: list[dict], state_path: str | Path,
                   chunk_size: int = 1000) -> list[str]:
    import anthropic
    client = anthropic.Anthropic()
    ids = []
    for i in range(0, len(requests), chunk_size):
        batch = client.messages.batches.create(
            requests=requests[i:i + chunk_size])
        ids.append(batch.id)
        print(f"submitted {batch.id}: requests "
              f"{i}..{min(i + chunk_size, len(requests)) - 1}")
    Path(state_path).write_text(json.dumps({"batch_ids": ids}, indent=1),
                                encoding="utf-8")
    return ids


def poll_batches(state_path: str | Path) -> bool:
    import anthropic
    client = anthropic.Anthropic()
    ids = json.loads(Path(state_path).read_text(encoding="utf-8"))["batch_ids"]
    done = True
    for bid in ids:
        b = client.messages.batches.retrieve(bid)
        print(f"{bid}: {b.processing_status}  {b.request_counts}")
        done &= b.processing_status == "ended"
    return done


def fetch_batch_results(state_path: str | Path,
                        raw_out: str | Path) -> dict[str, dict]:
    """Streams all batch results, writes the raw records to raw_out (JSONL,
    reproducibility) and returns {custom_id: record}."""
    import anthropic
    client = anthropic.Anthropic()
    ids = json.loads(Path(state_path).read_text(encoding="utf-8"))["batch_ids"]
    responses: dict[str, dict] = {}
    with Path(raw_out).open("w", encoding="utf-8") as fh:
        for bid in ids:
            for result in client.messages.batches.results(bid):
                rec = {"custom_id": result.custom_id,
                       "type": result.result.type, "batch_id": bid}
                if result.result.type == "succeeded":
                    msg = result.result.message
                    rec["stop_reason"] = msg.stop_reason
                    rec["text"] = "".join(b.text for b in msg.content
                                          if b.type == "text")
                    rec["usage"] = {
                        "input_tokens": msg.usage.input_tokens,
                        "output_tokens": msg.usage.output_tokens}
                else:
                    rec["error"] = str(getattr(result.result, "error", ""))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                responses[result.custom_id] = rec
    return responses


def estimate_cost(requests: list[dict], chars_per_token: float = 3.5,
                  out_tokens_per_item: int = 600) -> dict:
    """Rough pre-submit estimate. claude-fable-5 list price $10/$50 per MTok;
    the Batches API bills at 50%. Printed by --build, recorded in the
    manifest; the fetch step reports the real usage."""
    in_chars = sum(len(r["params"]["system"])
                   + len(r["params"]["messages"][0]["content"])
                   for r in requests)
    in_tok = in_chars / chars_per_token
    out_tok = len(requests) * out_tokens_per_item
    return {"n_requests": len(requests),
            "input_tokens_est": int(in_tok),
            "output_tokens_est": int(out_tok),
            "usd_est": round((in_tok * 10 + out_tok * 50) / 1e6 * 0.5, 2)}
