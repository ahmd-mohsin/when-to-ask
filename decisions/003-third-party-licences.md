# 003 — Unlicensed repos: owner has personal permission; vendor as reference

---
status: agreed
date: 2026-07-02
---

**Context.** No licence file in: `farukakgul/LYNX`, `saprmarks/geometry-of-truth`,
and the existing `hil-bench` and `ClarifyGPT` clones (default = all rights
reserved). MIT and fine: `nrimsky/CAA`, `andyzoujm/representation-engineering`,
`liam0949/REAL_ICLR`, `eigenscore`, `OPENIA`, `mini-swe-agent`.

**Decision.** Owner states they have **personal permission from the authors**
to use the unlicensed repos for this research. No emails to be sent. All
five new repos are vendored into `third_party/` (git-ignored except
PROVENANCE.md/VERSIONS.md — reference-use, not redistribution).

**Consequences.** Each unlicensed repo's PROVENANCE.md records
"no licence file; owner reports personal permission from authors
(2026-07-02)". If any of this code is ever to be redistributed with a paper
artifact, revisit then.
