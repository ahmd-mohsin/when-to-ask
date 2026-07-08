# 013 — First real A4 gate run (20 tasks) — findings for owner review

---
status: STOP POINT — owner/supervisor review required before Part B (decisions/011)
date: 2026-07-08
---

**Setup.** 20 tasks × 8 runs (160 grounded traces), 6561 reads, 60%
decision-labeled, 25% class-labeled, **32 forked blockers** (vs 2 in the
3-task sample). Split: 4 tasks (swe_0/13/18/22) held out ENTIRELY for OOD;
seeds s6/s7 of the other 16 held out for gates 1-5,7; A1/A2/A3 trained on the
rest. Nothing tuned on eval/OOD (build brief rule 1).

**Interpretation-class artifact** (`data/interpretation_classes.json`, 72
blockers): swe_0/1/10 hand-derived; swe_11-26 derived by an Opus per-task
workflow (grounded in registry + traces), 3 LLM-verified, all 17 passed a
deterministic anchor-leak repair (77 anchors that uniquely predicted one class
were demoted from anchors; every blocker kept ≥3 anchors) + owner hand-audit +
empirical labeler validation (forks discriminate correctly — verified on
swe_2's overflow-sentinel fork by hand).

## Gate numbers (held-out real data — reported as-is, NOT tuned)

| gate | number | read |
|---|---|---|
| 1 topic-leakage (WITHIN decision) | acc 0.63 vs chance 0.50; **partial eta² 0.025** (n=2 decisions) | invariance largely holds; underpowered |
| 1 (naive GLOBAL, confounded) | acc 0.41 vs chance 0.034 | **confounded** by decision identity — see note |
| 2 decision-recovery | acc 0.55 vs chance 0.024 (22×) | PASS — topic encodes decision |
| 3 fork-collocation | same 0.83 vs diff 0.66; theta 0.87 (n_same 43) | WEAK PASS — modest but real |
| 4 conflation | 26% of same-task/diff-decision pairs collocate | CONCERN — some label conflation |
| 5 lean-separation | ratio 0.56, silhouette −0.06 (n=3 decisions) | RED but underpowered |
| 6 OOD transfer | bucket purity 0.28 (unseen tasks) | poor transfer — expected limitation |
| 7 lead-time | median K=7, all positive (n=3, proxy ref) | tentatively POSITIVE, weak |

**Gate-1 confound (important, found during the run).** Global class ids are
nested within decisions (each of 220 classes belongs to one of 50 decisions),
so a "predict global class from T" probe scores mostly by recovering the
DECISION — which gate 2 *wants*. The hypothesis is "T blind to the LEAN within
a decision", so gate 1 was corrected to a within-decision probe. Corrected
partial eta² = 0.025 (near zero) says invariance largely held; the naive 0.41
was almost entirely the confound. This was a measurement fix, not tuning; both
numbers are reported.

## Honest bottom line (for the supervisor)

- **Not a failure, not a win — the make-or-break question is still open, and
  the binding constraint is statistical power, not a negative result.**
- What works: topic encoder recovers decision identity (gate 2) and is
  largely lean-invariant within a decision (gate 1 corrected); forking runs
  partially collocate (gate 3); lead-time is positive where measurable (gate 7).
- What's unresolved: gate 5 (does the lean subspace separate interpretations —
  the thing the trigger fires on) is negative on the only 3 held-out decisions
  with enough data. Combined with A3's low settle rate (13/183 = 7%), the
  **lean subspace + commitment are the weak links.**
- Root cause is largely POWER: class-labeled reads are 25%, and holding out by
  seed then requiring ≥2 classes/decision leaves only 2-3 measurable decisions
  for gates 1/5/7. OOD transfer (gate 6) is a genuine limitation.

## Recommended next steps (owner decides — all pre-registered, none is tuning)

1. **Layer sweep** (decisions/007): mid-layer 14 was a guess; the lean may live
   at a different depth. Re-read A0 at layers {0.4,0.5,0.6,0.7}×depth (needs a
   short AWS re-read pass; hidden states weren't saved for other layers).
2. **A3 window/eps sweep** (decisions/007): 7% settle rate is too low — the
   commitment definition needs its window/eps calibrated on real lean scale.
3. **Gate power**: switch gates 1/5/7 from single seed-holdout to k-fold over
   all labeled reads (still A2-held-out) to raise n_decisions; a gate-
   methodology change for owner sign-off, not a tuning of the model.
4. **More class coverage / data**: richer signature lexicons or more tasks/N to
   lift the 25% class-label rate that starves gate 5.

Do NOT build Part B on these numbers (decisions/011). Artifacts:
`models/gate_report.json`, `models/label_audit_full.md`, `models/a2_history.jsonl`.
