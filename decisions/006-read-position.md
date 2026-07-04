# 006 — Read position: fixed cadence + deliberation cues, never boundary-only

---
status: assumed (non-blocking; proposed default unobjected)
date: 2026-07-02
---

**Decision.** Read the mid-layer residual `h` during generation:
- every **K = 32** generated tokens (K ∈ {16, 32, 64} is a Phase-2 sweep), AND
- at deliberation cues from
  {"hmm", "wait", "let me", "actually", "should i", "alternatively"},
  matched on the **detokenized text at word boundaries** (lowercased,
  whitespace collapsed, trailing punctuation ignored) — NOT per-token string
  equality, since cues span token boundaries and usually arrive with
  punctuation attached ("Wait,"). LYNX's cue-token reading is the precedent;
  exact semantics in specs/A0-data-collection.md.

Every read is logged with its generated-token index and the trigger kind
(`cadence` | `cue`). Aggregating reads to one-vector-per-action (what
`HFWhiteBoxModel.generate` currently does) is forbidden in the wta path —
that is the read-at-boundary pattern the method doc names as the most common
way to kill the project.

**Consequences.** The trajectory log schema is per-read, not per-action.
Lead-time (A4 gate 7) is measured in read positions relative to the action.
