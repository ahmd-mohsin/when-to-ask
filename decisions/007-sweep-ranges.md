# 007 — Phase-2 sweep ranges (layer, windows, stability metric)

---
status: assumed (non-blocking; proposed default unobjected)
date: 2026-07-02
---

**Decision.** Nothing hardcoded; sweeps in Phase 2 over:
- mid layer ∈ {0.4, 0.5, 0.6, 0.7} × depth (config accepts fraction or index);
- commitment smoothing window w ∈ {3, 5, 8} reads;
- stability metric: **max displacement of `r` over the last w reads**
  (primary), mean pairwise distance (fallback);
- read cadence K ∈ {16, 32, 64} (see 006);
- bucketing hysteresis M and merge threshold: small sweeps per method doc §B.

Defaults used until the sweep runs: layer 0.5·depth, w = 5, K = 32.
