# 008 — N and trajectory diversity

---
status: assumed (non-blocking; proposed default unobjected)
date: 2026-07-02
---

**Decision.**
- **A0 offline collection: N = 8** per task (seeds 0–7; temperature cycled
  {0.7, 0.85, 1.0}; persona nudge OFF by default).
- **Online eval: N = 4**, with all matched-compute baselines run at the same N.
- Diversity is measured and reported, not assumed: per known blocker, the
  number of distinct interpretation classes hit and the entropy of committed
  actions. Escalation rule: if < 30% of fork-tasks show ≥ 2 interpretations,
  raise temperature / enable persona nudge and report the change.

**Consequences.** A0 storage estimate at N=8 over harbor_swe (100 tasks):
~2k reads/run × 3584-dim fp16 ≈ 14 MB/run ≈ 11 GB total.
