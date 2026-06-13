"""Decision-point detection -- OURS.

A decision point is a step where >=1 trajectory is about to commit to an
under-determined choice (brief S5c). We operationalise it as a tool-call boundary, and
attach a semantic *anchor* (sub-goal / file-under-edit) used later to align the N
asynchronous trajectories.

  * synthetic: every step is a decision point; the anchor is the step's gold dp index.
  * real:      `RealDecisionPointDetector.anchor(...)` flags tool-call boundaries and
               returns a coarse semantic anchor (e.g. the file being edited), or None.

The anchor is the key `signals.alignment` groups on. Step index alone aligns the early
synchronous steps; the semantic anchor takes over once trajectories drift apart.
"""

from __future__ import annotations

import re

from ..harness.tasks import DecisionPoint, Task


def synthetic_anchor(dp: DecisionPoint) -> str:
    """For synthetic tasks the gold decision points are known; anchor by index."""
    return f"dp{dp.index}"


class RealDecisionPointDetector:
    """Flags tool-call boundaries in a real agent transcript and returns an anchor.

    Used on the GPU run with a real env. The anchor combines a step-index bucket (for
    early synchronous alignment) with the file under edit (semantic anchor once
    trajectories diverge), per brief S5c.
    """

    _EDIT_FILE = re.compile(r"(?:edit|open|write|create)\s+([\w./\-]+\.\w+)", re.IGNORECASE)
    _TOOL_CALL = re.compile(r"^\s*(?:```|bash|python|edit|str_replace|create|run)\b", re.IGNORECASE)

    def is_boundary(self, action_text: str) -> bool:
        return bool(action_text) and bool(self._TOOL_CALL.search(action_text))

    def anchor(self, step_idx: int, action_text: str, early_steps: int = 2) -> str | None:
        if not self.is_boundary(action_text):
            return None
        if step_idx < early_steps:
            return f"step{step_idx}"  # synchronous phase: align by step index
        m = self._EDIT_FILE.search(action_text)
        return f"file:{m.group(1)}" if m else f"step{step_idx}"
