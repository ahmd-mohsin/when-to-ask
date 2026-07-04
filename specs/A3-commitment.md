# Spec A3 — Commitment (`r` steady + `s` dropped) and conformal `tau`

Commitment is **not** a trained network. It is a function of the lean
trajectory and the ambiguity signal, with one threshold `tau` calibrated by
split conformal prediction (procedure migrated from **LYNX**,
`third_party/LYNX/lynx/utils.py::conformal_quantile`; we borrow the
calibration mechanism, NOT their correctness target — our label is
observable from the trajectory).

## Definitions

- **Instability at read k** (window w, decisions/007):
  `instab_k = max_{j in (k-w, k]} ‖r_j − r_k‖₂` — max displacement of `L(h)`
  over the last w reads (undefined → +inf for k < w−1: a run can't commit
  before w reads exist).
- **s-drop gate:** `s_k ≤ s_ref`, where `s_ref` is fixed offline from A1's
  held-out distributions (crossover of should-ask vs proceed s-histograms).
- **Committed at k** ⇔ `instab_k ≤ tau` AND `s_k ≤ s_ref`.
- **Commitment weight (for Part B):** `w_k = sigmoid(alpha · (tau − instab_k))`
  when the s-gate holds, else 0 — soft, never hard-excludes a run.

## Stabilization-point label (calibration data, per (run, decision))

The earliest read `k*` such that for all later reads j in that decision,
`‖r_j − r_end‖₂ ≤ eps_settle` — the point after which the lean never moved
materially again before the action. Runs that never settle contribute no
calibration point (they are the loop channel's business, not A3's).

## L-scale normalization (amended 2026-07-03, integration finding)

The learned lean space has **arbitrary scale** — nothing in A2 pins it — so
absolute thresholds (eps_settle, tau, Part B's CUSUM reference) are
meaningless in raw units. Measured in the pipeline smoke: 23/24 calibration
sequences skipped as "never settled" and clear decisions fired the trigger.
`calibrate_tau` therefore estimates a **global scale** `l_scale` (RMS
distance of calibration reads from their sequence mean) and works in scaled
units; the detector divides incoming `r` by the same `l_scale`, and Part B
consumes `r / l_scale`. Per-bucket normalization stays the Phase-2 sweep
refinement (decisions/010).

## Calibration (split conformal, LYNX's procedure)

- One calibration point per (run, decision): nonconformity =
  `instab_{k* + w − 1}` — instability at the **first read whose trailing
  window lies wholly in the settled regime**. (Amended 2026-07-03: scoring at
  `k*` itself was wrong — the trailing window at `k*` still spans the settle
  transition, so its instability reflects the transition magnitude, not the
  settled-noise level; measured on fixtures it inflated tau ~3× and would
  admit pre-settle commits.) One point per (run, decision) —
  exchangeability, decisions/010.
- `tau = conformal_quantile(scores, delta)`: the finite-sample
  `ceil((n+1)(1−delta))`-th order statistic (faithful port of LYNX
  utils.py:339).
- Guarantee bought: for a fresh exchangeable stabilization point,
  `P(instab_{k*} > tau) ≤ delta` — i.e. with prob ≥ 1−delta the detector's
  threshold does not reject a true settling. Coverage is approximate under
  distribution shift (documented limitation, decisions/010).

## Interface

```
instability(r_seq: float[R, d_L], window: int) -> float[R]        (+inf-padded head)
stabilization_point(r_seq, eps_settle) -> int | None
conformal_quantile(scores: float[n], delta: float) -> float       (LYNX port)
calibrate_tau(list of r_seq, window, eps_settle, delta) -> tau, report
CommitmentDetector(tau, s_ref, window, alpha).step(r_k, s_k) -> (committed: bool, weight: float)
    (stateful per (run, decision); streaming, no lookahead)
```

## Observable behaviour that verifies this spec (contract, on fixtures)

1. `conformal_quantile` reproduces the finite-sample formula on hand-computed
   cases, and empirical coverage on synthetic scores ≥ 1−delta.
2. On fixture runs, the detector's first committed read lands within
   [settle_idx, settle_idx + w] for ≥ 90% of (run, decision) pairs, and
   **never** fires on loop runs (which must yield weight 0 throughout).
3. Streaming = batch: feeding reads one at a time gives the same decisions as
   computing offline.
4. De-commitment: if `r` starts moving again (blip), committed flips off and
   the weight drops — the mutable-vote retraction path in Part B depends on
   this.
