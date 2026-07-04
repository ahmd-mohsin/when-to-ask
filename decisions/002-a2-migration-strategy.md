# 002 — A2 without ReDAct code: migrate REAL's machinery, cite "following ReDAct"

---
status: agreed
date: 2026-07-02
---

**Context.** ReDAct (arXiv 2602.19396, "Hiding in Plain Text…", Farzam et al.)
has no public code — verified 2026-07-02 (no arXiv code link, no availability
statement, no matching GitHub repo). The brief's stop-and-ask condition fired.
DEAL and REAL are the same paper (arXiv 2506.08359; v1 titled DEAL, v2 retitled
REAL, ICLR 2026); official repo `liam0949/REAL_ICLR` (MIT) exists but its
mechanism (per-attention-head VQ-AE) differs from ReDAct's two-headed encoder.

**Decision.** Owner: no emailing authors; prefer migrating existing code over
token-expensive reimplementation. Therefore:
- Vendor `liam0949/REAL_ICLR` (MIT) and migrate whatever training machinery
  transfers (activation datasets/loaders, training loop scaffolding, probe
  utilities).
- The A2 architecture itself (shared MLP body → T/L heads, gradient reversal,
  orthogonality, reconstruction) is implemented from the ReDAct paper (PDF in
  `multi trajectory disagreement about same idea/`), since no code exists.
- Paper phrasing: "following ReDAct" / "architecture after ReDAct", NEVER
  "using ReDAct's code". Cite DEAL as arXiv 2506.08359**v1** if the DEAL name
  is used; the repo is named for the REAL title.

**Consequences.** A2 is part-migration, part-faithful-reimplementation; the
PROVENANCE.md for REAL_ICLR records exactly which pieces were taken. The
citation-honesty rule from the brief is preserved by the phrasing change.
