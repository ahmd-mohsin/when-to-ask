# 018 — 14B v2 verdict, label fixes, and the scale-run design

---
status: agreed (owner 2026-07-18: "you are finally prepping for the large scale run so do it")
date: 2026-07-18
---

## 1. The 14B v2 collection verdict (wta-a0-v2-14b, 160 runs, 6.1 GPU-h)

Collection VERIFIED (all runs complete, tensors finite, 1807 real actions,
1304 value-triggered reads, 16 forked blockers vs the pilot's 2). Science:

- **Structural forks separate strongly in raw activations**: leave-one-run-out
  centroid 0.71–0.73 vs 0.50 chance (705 reads / 4 decisions; swe_19
  remove-vs-deprecate 1.00). At 7B this was ~chance (0.48) — the footprint
  grows with scale and real trajectories.
- **Real lead-time (first paper-grade number)**: gate 7 measured against
  actual ActionEvents (102/124 committed pairs matched): median K = 10.5 ± 3.9
  reads before the behavioural commit, 100% positive across folds.
- **Value forks stay invisible** even on emission-timed value reads (0.14 vs
  0.42 chance, below chance on all 4 layers) — and the negative SURVIVES the
  action-based label fix (§2), so it is not a labeling artifact for the
  measured decisions. The decisions/015 asymmetry hardens.
- **Topic subspace survives the move to agent transcripts**: gate 2 recovery
  0.44 vs 0.036 chance; gate 3 collocation 0.81/0.60. NEW regression: gate 4
  within-task conflation 0.26 → 0.79 (agent context windows mix blockers) —
  Part B must bucket within-task. Gate 6 OOD purity still ~0.30.
- **Gate 5 (A2-lean, the pre-registered make-or-break) stays underpowered**:
  k-fold no-OOD ratio 0.90 ± 0.77 on 5 pooled decisions, heterogeneous (one
  fold 1.98/0.15). The scale run exists to power this.
- Bug found+fixed: `seq_by_run_decision`/`train_offline` sorted reads by
  `read_token_idx`, which RESTARTS per v2 segment → interleaved turns fed to
  A3/gate7. Fix = generation (row) order, regression-tested.

## 2. Action-based commitment labeling (spec labels.md "v2" section)

Whole-trace signature counting mislabels VALUE commitments on multi-turn
agent traces (verified mislabels: agents mention candidate numbers while
deliberating). Fix: mutating actions (`sed -i`, `>`, `tee `, `patch `,
`git apply`, `perl -i`) are scored FIRST; the commit position is the first
matching mutating action; trace scoring is the fallback; `label_source`
recorded per commitment. On the 14B data 182/244 commitments are now
action-sourced; coverage unchanged (16 forked blockers).

## 3. Class artifact: 60 tasks / 214 blockers + leak enforcement

- 40 new tasks derived (swe_3–9, swe_27–59) from registries ONLY (no traces
  exist yet — signatures to be empirically reviewed against the first
  collection, per the swe_11–26 precedent). Batch sources preserved in
  `data/interpretation_classes_new/`.
- `scripts/audit_class_artifact.py` mechanizes the anchor-leak repair: hard
  errors for anchor==signature, cross-class signature duplication, schema
  violations; containment overlaps are warnings (calibrated to the frozen
  artifact's accepted standard). New batches: 0 errors, near-0 warnings.
- The auditor found **8 anchor-IS-signature leaks in the frozen 20-task
  artifact** (swe_1, swe_10 ×2, swe_13 ×3, swe_14, swe_26). Robustness check
  BEFORE repair: raw-h diagnostic identical with/without them (none of the 5
  blockers ever produced a measured fork) → prior results unaffected →
  anchors deleted, provenance updated, and
  `harness/contract/test_class_artifact.py` now enforces ZERO leak errors on
  every pytest run.
- Known limitation: the matcher is case-insensitive, so case-only class
  distinctions (swe_49 ClickHouse casing) cannot be encoded.

## 4. Scale-run design (owner-approved)

| pool | tasks | classes needed |
|---|---|---|
| TRAIN | 60 = current 20 (burned for testing — we debugged on them) + 40 new | yes (done) |
| TEST (sealed) | ~30 swe from swe_60+ — untouched until the system is frozen | no (hil-bench judge scores) |
| OOD transfer | ~20 sql tasks | no |

Seal rule: once test tasks are run, nothing may be tuned afterwards; those
numbers are final. `collect_v2.py --classes data/interpretation_classes.json`
restricts collection to artifact (train) tasks — necessary because sorted-dir
order interleaves swe_60 before swe_7 and would otherwise touch the test pool.

## 5. Next: AWS runbook step 2c (the scale collection, ~20 GPU-h)
