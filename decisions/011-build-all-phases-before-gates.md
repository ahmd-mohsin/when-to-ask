# 011 — Implement all phases now; A4 gates become a usage gate, not a build gate

---
status: agreed (owner, 2026-07-03)
date: 2026-07-03
---

**Context.** The build brief ordered: do not implement Part B until the A4
gates pass on real data. Real data needs the owner's AWS GPU instance, which
arrives "some days" out. Owner: "I just want my implementation done for all
phases" — build everything now, test on AWS when it arrives.

**Decision.** Implement Phases 2–4 (A2, A3, A4 runners, Part B, eval) now,
validated end-to-end on synthetic fixtures on the CPU laptop. The A4 gate rule
changes from a *build* gate to a *usage* gate:

- No Part B result is trusted, reported, or iterated on until the A4 gates
  have been run on real held-out data and reviewed by the owner.
- The gates themselves are unchanged: run on held-out real data, numbers
  reported as-is, no tuning to pass (build brief rule 1 stands).
- Accepted risk (owner's call): if gates fail on real data, some Part B
  implementation effort is wasted.

Also: torch (CPU build) is installed on the laptop for A2 training on
fixtures — decisions/004's "no torch on the laptop path" softens to "no
*GPU/torch requirement* for the smoke path"; the same torch code runs on AWS.

**Consequences.** The AWS runbook (Phase 4) defines the real-data order:
prove_hook → A0 collection → A1 sanity numbers → A2 training → A3 calibration
→ **A4 gates → STOP for owner review** → Part B eval.
