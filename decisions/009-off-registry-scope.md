# 009 — Off-registry fallback: implemented, exercised only in OOD/limitations

---
status: assumed (non-blocking; proposed default unobjected)
date: 2026-07-02
---

**Decision.** The doc's fallback for sub-decisions with no registry class
(observable-action clusters, gated to junctures where the ambiguity signal
`s` is high) is implemented, but core A2 training uses registry-derived
interpretation classes only (see 005). The fallback is exercised in the OOD
transfer gate (A4 check 6) and reported as a limitation path, not mixed into
primary supervision.
