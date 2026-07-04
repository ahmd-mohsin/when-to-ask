# 005 — Interpretation classes: one-time LLM-assisted derivation, frozen + audited

---
status: agreed ("do your best" → proposed option a)
date: 2026-07-02
---

**Context.** HiL-Bench's 200 `blocker_registry.json` files carry one canonical
`resolution` + `example_questions` per blocker — not the enumerated set of
competing interpretation classes that A2's `L`-supervision needs.

**Decision.** Build the class list as a **one-time frozen artifact**:
for each blocker, derive 2–4 interpretation classes from its
`description` + `resolution` + `example_questions` (each example question
already implies the competing alternatives). Output:
`data/interpretation_classes.json`, keyed by (task, blocker_id), versioned
with the prompt + model used for derivation. Hand-audit a random sample of
20 blockers before first use; record the audit in `decisions/`.

**Consequences.** This artifact *is* "the registry's class list" the method
doc refers to. It is built once, offline, before A2 training (Phase 2), and
never edited during training/eval. Deriving it is a data-prep step, not part
of the learned pipeline — no circularity with the runtime detector.
