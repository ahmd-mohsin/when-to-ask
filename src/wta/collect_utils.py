"""A0 collection utilities -- prompt assembly, context handling, signatures.

Pulled out of scripts/ so the leak-sensitive logic is unit-testable (spec A0
v1.5, ADR 012). The grounding rule: prompts may contain the task instruction
and PRE-PATCH source files (which cannot contain the gold patch's added lines
by construction) -- never full_info instructions, never test files or names
(they encode expected behaviour), never patch content beyond file paths.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# paths in the gold patch that must never contribute context
_EXCLUDED_PATH_PARTS = ("test/", "tests/", "changelogs/", "docs/", ".github/")

DELIBERATION_NUDGE = (
    "The task may leave some choices under-specified. Where alternatives "
    "exist, briefly weigh them against each other before committing to one, "
    "then state your final implementation."
)

PROMPT_TEMPLATE = """You are a coding agent. Work on this task:

{instruction}
{context_block}
Think through how you would implement this step by step, reasoning about any
choices the task leaves open, then state the exact change you would make.
{nudge}"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def patch_touched_files(patch_text: str) -> list[str]:
    """Source files the gold patch touches (the ONLY thing we read from the
    patch -- file paths, never content). Tests/changelogs/docs excluded: test
    names encode expected behaviour (e.g. `...not_linux[SunOS-Solaris]` leaks
    a normalization resolution)."""
    files = []
    for m in re.finditer(r"^\+\+\+ b/(.+)$", patch_text, flags=re.MULTILINE):
        path = m.group(1).strip()
        if any(part in path for part in _EXCLUDED_PATH_PARTS):
            continue
        if path not in files:
            files.append(path)
    return files


def load_context_files(task_ctx_dir: str | Path, max_files: int = 3,
                       max_chars_per_file: int = 12000) -> list[tuple[str, str]]:
    """Read pre-patch source files saved by scripts/extract_task_context.py.
    Returns [(original_repo_path, possibly-truncated text)]. Order comes from
    the extractor's MANIFEST file (patch order); truncation is explicit."""
    task_ctx_dir = Path(task_ctx_dir)
    manifest = task_ctx_dir / "CONTEXT_MANIFEST.txt"
    if not manifest.exists():
        return []
    out = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repo_path, _, local_name = line.partition("\t")
        f = task_ctx_dir / local_name
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + "\n... [truncated for prompt]"
        out.append((repo_path, text))
        if len(out) >= max_files:
            break
    return out


def build_prompt(instruction: str, context_files: list[tuple[str, str]] | None = None,
                 nudge: bool = True) -> str:
    """Assemble the collection prompt. Records nothing itself -- the caller
    logs the prompt hash and grounding mode (manifest)."""
    if context_files:
        parts = ["\nRelevant source files from the repository (current state, "
                 "before your change):\n"]
        for repo_path, text in context_files:
            parts.append(f"\n--- {repo_path} ---\n```\n{text}\n```\n")
        context_block = "".join(parts)
    else:
        context_block = ""
    return PROMPT_TEMPLATE.format(
        instruction=instruction.strip(), context_block=context_block,
        nudge=DELIBERATION_NUDGE if nudge else "",
    ).strip() + "\n"


_CODE_BLOCK = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)
_WS = re.compile(r"\s+")


def answer_signature(text: str) -> str:
    """Diversity signature: hash of the LAST fenced code block (whitespace-
    normalized) -- the closest thing a v1 trace has to a committed action.
    Falls back to the normalized last 300 chars for block-less traces.
    (Replaces the crude last-200-chars signature; distinct signature is still
    only an UPPER bound on distinct interpretations -- the labeler measures
    the real thing.)"""
    blocks = _CODE_BLOCK.findall(text)
    basis = blocks[-1] if blocks else text[-300:]
    return sha256_text(_WS.sub(" ", basis.strip()))
