# 015 — Second (powered) gate run: verdict on the make-or-break

---
status: STOP POINT — owner/supervisor decision required (decisions/011)
date: 2026-07-09
---

**Setup.** Multi-layer collection (160 runs, (R,4,3584), layers [11,14,17,20]
@ 4679cac, sha-verified). Layer sweep + eps/window sweep + 5-fold cross-fit
gates (decisions/014). Confounds from decisions/013 now eliminated: layer
choice (swept 4), A3 settle rate (7%→33% at eps=1.0/window=3), statistical
power (11–12 pooled decisions vs 2–3).

## Results (as-is, not tuned)

- **Layer sweep** (gate5 ratio): L11 0.83 / **L14 1.03** / L17 0.59 / L20 0.89.
  The original mid-layer guess was already the best. "Wrong layer" is ruled out.
- **5-fold gates @ L14, eps 1.0, w3:** gate1 within-decision eta² 0.062±0.032
  (invariance mostly holds); **gate5 ratio 0.726±0.303, silhouette 0.011±0.031
  over 12 pooled decisions — a POWERED negative**; gate7 median K 7.1±2.0,
  frac_pos 1.0 (positive, but built on the same weak L — interpret cautiously).
- **Decisive diagnostic — raw-h cross-run separability** (leave-one-run-out
  nearest-class-centroid on RAW layer-14 activations, 14 forked decisions):
  **mean 0.479 vs chance 0.464.** The gate-5 failure is NOT an A2 artifact —
  the raw representation at our read positions mostly does not carry
  cross-run-readable interpretation identity.
- **But the failure is heterogeneous:** 4/14 decisions separate strongly
  (0.69–0.90 vs 0.50): swe_23/uid_authority (0.90), swe_23/localid_source
  (0.87), swe_12/spotlight_ownership (0.75), swe_14/default_data_path (0.69) —
  all *structural* interpretation differences (which source/API/owner).
  Value-choice blockers (which timeout number, which sentinel constant) sit at
  or below chance. Pattern: **structurally distinct interpretations leave an
  activation footprint; literal-value choices don't** (at cadence read
  positions, this model/scale).

## Honest verdict

The method's central bet — "which interpretation a run is heading toward is
linearly readable from mid-layer states, across runs, for typical blockers" —
**does not hold in general for Qwen2.5-Coder-7B on HiL-Bench SWE tasks. It
holds for a structurally-distinct minority (~30%).** This is a real scientific
finding, not an engineering artifact: every pre-registered confound (layer,
power, commitment calibration, label leaks, GRL treadmill, read-at-boundary)
was eliminated or measured.

## Options for the owner/supervisor (method decisions, not mine to make)

A. **Reframe around the finding:** detector scoped to structural forks
   (where the signal exists) + report the value-fork blindness as a
   characterized limitation; possibly hybrid with output-divergence for
   value forks. Honest, publishable shape (positive capability + negative
   boundary + infrastructure).
B. **One more representational lever:** contrastive per-decision lean
   objective / larger d_lean / read positions at value-mention tokens
   (labeling can find them). Riskier; the raw-h diagnostic caps the upside.
C. **Negative-result paper** with the full pre-registered pipeline as the
   contribution.

Artifacts: models/gate_report_kfold.json, sweep tables in session log,
raw-h diagnostic script inline in the 2026-07-09 session.
