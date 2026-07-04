# Spec A1 — Ambiguity direction `d` and scalar signal `s`

## Purpose

A single fixed direction `d` in residual space such that `s = dot(h, d)` reads
"how under-specified / should-ask does this state look". Jobs of `s`: gate
which junctures count as ambiguous decisions, help time commitment (`s` drops
as a run settles), and gate the off-registry fallback (decisions/009).
**`s` is never the disagreement trigger** — the trigger fires on spread of
`r = L(h)` (Phase 2), not on `s`.

## Construction (migrated mechanism)

Difference-in-means / mass-mean, per CAA (`generate_vectors.py`,
`(pos − neg).mean`) and Geometry of Truth (`probes.py::MMProbe`,
`pos.mean(0) − neg.mean(0)`); provenance in
`third_party/CAA/PROVENANCE.md`, `third_party/geometry-of-truth/PROVENANCE.md`.

```
d_raw = mean(h | label=should_ask) − mean(h | label=proceed)
d     = d_raw / ||d_raw||2
```

No network, no autoencoder, no per-fork sign.

## Interface

- `build_direction(h_should_ask: float[P,H], h_proceed: float[Q,H]) -> float32[H]`
  - errors on: empty class, H mismatch, non-finite input, zero-norm difference.
  - deterministic; unit norm (‖d‖₂ = 1 ± 1e-6).
- `ambiguity_signal(h: float[...,H], d: float32[H]) -> float[...]` — batched dot.
- `auroc(s_pos: float[P], s_neg: float[Q]) -> float` — tie-aware, delegating to
  scikit-learn's `roc_auc_score` (already a project dependency — no reason to
  re-derive it); used by the contract harness and the Phase-1 sanity report.

## Data

- Training labels: HiL-Bench blocker-linked junctures (should-ask) vs
  specified junctures (proceed) — real-data pass happens on AWS after A0
  collection. Until then, the synthetic fixture provides labeled states with a
  planted direction `d*` (fixtures/synthetic.py).
- **Training contrast is MATCHED (spec change 2026-07-02, with reason):**
  `d` is built from should-ask states vs **settled states of the same
  decisions** (post-commitment reads — the doc's "no longer looks
  ambiguous"), not vs different, clear decisions. Contrasting against clear
  decisions leaks decision *identity* into `d` — measured on fixtures:
  cos(d, d*) = 0.53 confounded vs 0.99 matched. This mirrors the contrastive
  recipe of the migrated sources (CAA pairs are matched on content and differ
  only in the target property). On real data the same rule applies: proceed
  states for the contrast come from the same tasks after the juncture is
  resolved/settled, not from unrelated specified junctures. The
  ambiguous-vs-clear confound size on real data is a research number — report
  it, don't tune it away.
- Separation (AUROC) is still **evaluated** against all specified-looking
  states: held-out settled + clear-decision states.
- `d` is built from a **training split only**; every reported separation
  number is computed on held-out states.

## Observable behaviour that verifies this spec (contract, on fixtures)

1. **Recovery:** with ≥ 200 states/class at noise 0.08, `|cos(d, d*)| ≥ 0.95`.
2. **Separation:** AUROC(s on held-out should-ask vs proceed) ≥ 0.9.
3. **Timing shape:** on ambiguous decisions, mean s(pre-commitment reads) >
   mean s(post-commitment reads); on clear decisions s ≈ 0 throughout.
4. Determinism, unit-norm, and all error cases above.

## Research numbers (NOT tuned to pass — reported to owner)

On real A0 data: held-out AUROC of `s`, and the s-drop-at-commitment curve.
These are Phase-1 sanity numbers feeding the A4 gates later; if AUROC on real
data is near chance, stop and report (the A1 label source or read policy is
wrong — do not fix by tuning on eval data).
