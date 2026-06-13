"""Execution layer -- produces the per-trajectory OUTPUT SIGNATURE that B1 clusters.

ClarifyGPT's signal (third_party/ClarifyGPT/.../runTests_getTaskID) is: run each candidate
solution on a shared set of test inputs, form an output signature per solution, and group.
B1 (`signals.output_divergence`) consumes those signatures. This module is the source of
the signature for a trajectory at a decision point.

  * `SyntheticExecutor` -- signature parsed from the FakeWhiteBoxModel's action text.
  * `HilBenchExecutor`  -- runs the real SWE/SQL tests via the vendored harbor stack and
                           hashes the test-result vector (ClarifyGPT-style). GPU/Docker only;
                           the interface is identical so B1 is unchanged across sources.

Per brief S5a the SWE/SQL execution layer should be REUSED rather than re-derived; the
HiL path shells out to the harbor runners in third_party/hil-bench.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

from ..backbone.model import Generation

_INTERP_RE = re.compile(r"interp=(\w+)")


class Executor(Protocol):
    def signature(self, generation: Generation, *, task, dp) -> str:
        """A hashable string summarising the trajectory's observable output here."""
        ...


class SyntheticExecutor:
    """Signature = the chosen interpretation in the fake action text.

    Trajectories that committed to the same interpretation get the same signature, so
    B1 sees them as agreeing (exactly ClarifyGPT's identical-output grouping).
    """

    def signature(self, generation: Generation, *, task=None, dp=None) -> str:
        m = _INTERP_RE.search(generation.text or "")
        return m.group(1) if m else "none"


class HilBenchExecutor:
    """Run real tests and hash the output vector (ClarifyGPT-style). GPU/Docker only.

    Calls the harbor SWE/SQL runner on the trajectory's current patch, collects the
    per-test results, and returns a stable hash of the result vector as the signature.
    Two trajectories agree iff their patches produce identical test outputs.
    """

    def __init__(self, hil_root=None):
        from .tasks import HIL_ROOT

        self.hil_root = hil_root or HIL_ROOT

    def signature(self, generation: Generation, *, task, dp) -> str:
        results = self._run_tests(task, generation)
        return hashlib.sha1(repr(results).encode()).hexdigest()[:16]

    def _run_tests(self, task, generation) -> list:  # pragma: no cover - needs Docker/GPU
        raise NotImplementedError(
            "HilBenchExecutor wires into third_party/hil-bench's harbor_swe / harbor_sql "
            "runners (Docker). Implement on the GPU box during the real de-risk run: apply "
            "the trajectory's patch, run the task's tests, return the per-test result vector."
        )


def build_executor(source: str) -> Executor:
    return HilBenchExecutor() if source == "hil_bench" else SyntheticExecutor()
