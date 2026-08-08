"""Full-info arm: problem-statement augmentation (spec eval, decisions/022).

MIRRORED byte-for-byte from hil-bench's
``hil_bench/templates/problem_full_info.jinja2`` as rendered by
``scripts/swe.py::augment_problem_full_info`` (default jinja2 Environment:
no trim_blocks/lstrip_blocks, keep_trailing_newline=False; verified against
the vendored template @352d14c). Kept as a plain function so the our-loop
full_info arm and the bridge's full_info arm feed the model the IDENTICAL
augmented statement.
"""

from __future__ import annotations

_HEADER = ("\n\n---\n\n## Additional Context\n\n"
           "The following clarifications are provided to help you complete "
           "this task:\n\n")


def augment_full_info(statement: str, blockers) -> str:
    """`blockers`: xtid Blocker objects or registry dicts. Empty -> unchanged
    (upstream returns the plain statement when the registry is absent/empty)."""
    blockers = list(blockers or [])
    if not blockers:
        return statement
    parts = [statement, _HEADER]
    for b in blockers:
        desc = b["description"] if isinstance(b, dict) else b.description
        res = b["resolution"] if isinstance(b, dict) else b.resolution
        parts.append(f"\n### {desc}\n\n{res}\n\n")
    return "".join(parts)
