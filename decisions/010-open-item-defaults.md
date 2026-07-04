# 010 — Defaults for the doc's "open items that stay empirical"

---
status: assumed (non-blocking; proposed default unobjected)
date: 2026-07-02
---

**Decision.**
- **Conformal exchangeability:** calibrate on ONE read per (run, decision) —
  the stabilization-point read — not every step; treat coverage as
  approximate under shift (per doc).
- **Per-bucket `r`-spread normalization:** deferred to the Phase-2 sweep;
  raw dispersion first, per-bucket z-scaling as the sweep alternative.
- **Oscillating runs** (never stably settle, no repeated env states):
  contribute low-weight votes via the soft commitment weights
  `w = sigmoid(alpha*(commitment - threshold))` — never hard-excluded;
  weight floor swept in Phase 2.
