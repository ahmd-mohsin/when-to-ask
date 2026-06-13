"""Verbalized-across-N (B2).

Each of the N trajectories self-reports a confidence; we aggregate to an ask-score. This
controls for "is internal needed, or does verbalized-across-N suffice?" (brief B2). It is
expected to be weak on confident-convergent blockers (overconfident-wrong: high reported
confidence despite being wrong -- brief S4).

  * real model: `elicit_confidence` asks the model for a 0-100 confidence and parses it.
  * fake model: the confidence is already in `Generation.extra['confidence']`; otherwise
    we fall back to exp(mean_logprob).
"""

from __future__ import annotations

import re

import numpy as np

from ..backbone.model import Generation, WhiteBoxModel

_NUM_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)")


def elicit_confidence(model: WhiteBoxModel, action_prompt: str, *, seed: int | None = None) -> float:
    """Ask a real model how confident it is in its next action; return in [0, 1]."""
    prompt = (
        f"{action_prompt}\n\nOn a scale of 0 to 100, how confident are you that you know "
        "the single correct next action without needing to ask a clarifying question? "
        "Answer with just the number."
    )
    gen = model.generate(prompt, temperature=0.0, seed=seed, max_new_tokens=8)
    m = _NUM_RE.search(gen.text or "")
    if not m:
        return 0.5
    return float(np.clip(float(m.group(1)) / 100.0, 0.0, 1.0))


def confidence_from_generation(gen: Generation) -> float:
    if "confidence" in gen.extra:
        return float(gen.extra["confidence"])
    if gen.mean_logprob is not None:
        return float(np.clip(np.exp(gen.mean_logprob), 0.0, 1.0))
    return 0.5


def verbalized_ask_score(generations: list[Generation]) -> float:
    """Aggregate N confidences into an ask-score (higher = more likely to ask).

    Uses 1 - mean(confidence): the trajectories are collectively unsure.
    """
    if not generations:
        return 0.0
    confs = [confidence_from_generation(g) for g in generations]
    return float(1.0 - np.mean(confs))
