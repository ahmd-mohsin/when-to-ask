"""Regime labelling -- OURS.

Decision points fall into three regimes (brief S4), which determine the win region:

  * fork              -- trajectories split on a genuine interpretation choice. Ours wins.
  * confident_wrong   -- all N agree but are all wrong (the divergence blind spot; only a
                         correctness probe catches it).
  * clear             -- unambiguous; correctly silent.

For synthetic tasks the gold regime is known and stored on each record. For real
HiL-Bench tasks we start from the blocker `type` and refine confident-convergent
behaviourally, since "all agree but wrong" is a property of the rollouts, not the blocker:

  ambiguous_requirement / ambiguous*            -> fork
  missing_parameter / missing_info / contradictory -> should-ask (fork unless behaviour
                                                   says confident-convergent)
  (no blocker)                                  -> clear

`infer_behavioral_regime` applies the behavioural refinement and is where the brief's
"light manual annotation on the slice" plugs in.
"""

from __future__ import annotations

from ..harness.tasks import CLEAR, CONFIDENT_WRONG, FORK

_AMBIGUOUS = {"ambiguous_requirement", "ambiguous", "ambiguity", "underspecified"}
_MISSING = {"missing_parameter", "missing_info", "missing_information", "contradictory", "conflicting"}


def regime_from_blocker_type(blocker_type: str | None) -> str:
    """Coarse regime from a HiL-Bench blocker type (pre behavioural refinement)."""
    if blocker_type is None:
        return CLEAR
    t = blocker_type.strip().lower()
    if t in _AMBIGUOUS:
        return FORK
    if t in _MISSING:
        return FORK  # should-ask; may be reclassified confident_wrong by behaviour
    return FORK  # any registered blocker => should-ask


def infer_behavioral_regime(
    *,
    has_blocker: bool,
    output_dispersion: float,
    probe_incorrect: float,
    blocker_type: str | None = None,
    output_eps: float = 1e-6,
    wrong_threshold: float = 0.5,
) -> str:
    """Refine the regime using rollout behaviour (real runs).

    A should-ask decision point where the N agree on output (low output dispersion) yet
    are likely wrong (high probe score) is confident-convergent; otherwise a fork.
    """
    if not has_blocker:
        return CLEAR
    if output_dispersion <= output_eps and probe_incorrect >= wrong_threshold:
        return CONFIDENT_WRONG
    return FORK


def should_ask(regime: str | None) -> bool:
    return regime in (FORK, CONFIDENT_WRONG)
