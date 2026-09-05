"""CIS lean readout -- OURS (decisions/029 Amendment 029.3).

The Stage-0 pilot showed that scoring the agent's EXACT recorded action under
an injected resolution measures relevance (own vs foreign resolution, AUROC
.855) but not correctness: the right answer lowers the right action's
likelihood 94% of the time, same as the wrong one (029.2). Exact-token
likelihood of a long free-form block is a surface measure.

This module scores a SHORT FIXED MENU instead. At the pre-action position
(after the generation header, before the model emits anything), teacher-force
one templated commitment statement per registry class of each blocker:

    STMT_PREFIX + resolution_c         (c = canonical, or a reviewed rival)

Every option shares the template and the context; only the class content
differs, so the perturbation that swamped CIS_bash cancels across options.
The remaining surface-form competition (some resolutions are intrinsically
likelier strings) is removed by domain-conditional PMI (Holtzman et al.,
2021): each option's log-prob under ctx_k minus its log-prob under a fixed,
task-agnostic NULL context. The softmax of PMI over a blocker's classes is the
model's implicit interpretation distribution p_k(c | b), from which:

    P_canonical(k, b) = p_k(canonical | b)         lean toward the right answer
    H(k, b)           = entropy(p_k) / log(n_c)    uncertainty, in [0, 1]

Relevance is retained as a second target: mean PMI of the task's own
canonical statements minus that of matched foreign tasks' canonical
statements (the analogue of 029 G0.6).

Everything here is torch-free. The scorer supplies log-prob sums.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wta.agent_loop import SYSTEM_PROMPT

# --- frozen template (029.3 par.2) ------------------------------------------
STMT_PREFIX = "THOUGHT: I will follow this rule: "
# Task-agnostic null context for PMI normalisation. Same system prompt, a
# generic user turn that names no task, no repository, no blocker.
NULL_USER = "Complete the assigned task in this repository."


def statement(resolution: str) -> str:
    return STMT_PREFIX + resolution.strip()


def null_messages() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": NULL_USER}]


def pmi(lp_ctx: float, lp_null: float) -> float:
    return float(lp_ctx - lp_null)


def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    ex = [math.exp(s - m) for s in scores]
    z = sum(ex)
    return [e / z for e in ex]


def entropy_norm(p: list[float]) -> float:
    """Shannon entropy in nats divided by log(n) -> [0, 1]; 0 when n == 1."""
    n = len(p)
    if n <= 1:
        return 0.0
    h = -sum(x * math.log(x) for x in p if x > 0)
    return float(h / math.log(n))


@dataclass
class BlockerReadout:
    blocker_id: str
    classes: list[str]            # registry order; classes[0] is canonical
    pmi: list[float]              # per class
    p: list[float]                # softmax(pmi)
    p_canonical: float
    entropy: float                # normalised
    argmax: str
    lp_ctx: list[float]           # raw sums, reported
    n_tok: list[int]


def readout(blocker_id: str, classes: list[str], lp_ctx: list[float],
            lp_null: list[float], n_tok: list[int]) -> BlockerReadout:
    if not (len(classes) == len(lp_ctx) == len(lp_null) == len(n_tok)):
        raise ValueError("readout: ragged inputs")
    sc = [pmi(a, b) for a, b in zip(lp_ctx, lp_null)]
    p = softmax(sc)
    return BlockerReadout(blocker_id=blocker_id, classes=list(classes), pmi=sc,
                          p=p, p_canonical=p[0], entropy=entropy_norm(p),
                          argmax=classes[max(range(len(p)), key=lambda i: p[i])],
                          lp_ctx=list(lp_ctx), n_tok=list(n_tok))


def length_normalised_p(lp_ctx: list[float], n_tok: list[int]) -> list[float]:
    """Secondary (reported, not gated): softmax of mean per-token log-prob."""
    return softmax([a / max(n, 1) for a, n in zip(lp_ctx, n_tok)])
