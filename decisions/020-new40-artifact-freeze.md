# 020: 32B gate-power verdict — data diversity is the bottleneck, not labels

Date: 2026-07-22. Status: DECIDED (owner). Supersedes the artifact-revision
plan drafted earlier the same day (kept below as "rejected plan").

## Verdict

**The 60-task Qwen3-32B collection (`data/a0_v2_32b`) cannot produce a powered
make-or-break gate5 (lean-separation) result. Relabeling cannot fix this. The
binding constraint is cross-run interpretation diversity in the trajectories,
and the remedy is re-collection, not artifact edits.**

## The decisive measurement (labeling-independent ceiling)

Gate5 needs, per forked decision, >=2 interpretation classes each committed by
>=2 DISTINCT runs (so there is within-class cross-run structure to separate).
Counting distinct committing runs per class over all 18 forked blockers in the
first label build (`models/v2_32b/labels_debug.jsonl`):

- **2 of 18** forks have >=2 classes each committed by >=2 runs. This is the
  CEILING that *perfect* labeling could reach — it is a property of the runs,
  not the labeler.
- Of those 2, only **1 is on an editable new-40 task** (swe_36/
  controller_replacement_import_path: keep_shim 6 runs vs stdlib 2 runs). The
  other (swe_18/default_manifest_pack_mode) is in the frozen-20.
- Even those 2 currently yield ~1 class-labeled read in 1 run per class
  (swe_18: product_style_aliases 4 reads/1 run, rival 0; swe_36: keep_shim
  1/1, stdlib 1/1). All other 16 forks have a single-run minority class:
  the model simply never diverged on >=2 runs, so no vocabulary edit creates
  the cross-run structure the gate requires.

Reference: the 14B run (memory / decisions/018) had 16 forked / ~5 measurable
and gate5 still came back negative + heterogeneous. 32B's ceiling of 2
measurable forks is **strictly weaker-powered than the run that already could
not resolve the claim.** Running the gate here would spend laptop time to
reproduce an INSUFFICIENT-DATA / underpowered-negative we can already predict.

## Decisions

1. **Do NOT run the make-or-break gate suite on this collection.** gate5 is
   unpowered by construction; a red/insufficient number adds no information.
2. **Cancel the 40-task artifact revision** (the rejected plan below). Ceiling
   is +1 measurable fork (swe_36), which cannot carry a make-or-break result
   alone. The apply/compare tooling and the per-task evidence lists are kept
   (scratchpad + the analysis transcript) so the revision can be executed
   later, scoped to whatever tasks a re-collection actually targets — done
   once, informed, not speculatively now.
3. **Salvageable now on 32B (coverage-based, fork-independent):** A1 ambiguity
   direction AUROC, gate1 topic-invariance, gate2 decision-recovery, gate7
   lead-time. These need decision-labeled reads (49% here), not forks, and
   would extend the 14B scale-up story to Qwen3-32B. Optional, cheap, CPU-only
   — run if/when the scale-up narrative is wanted. NOT the make-or-break claim.
4. **Escalate the one decision only the human owner can make (GPU cost):**
   a targeted higher-diversity RE-COLLECTION to break the single-run-minority
   ceiling. Recommended shape (a bet, not a guarantee):
   - Concentrate seeds on the ~15 tasks that showed ANY divergence at 8 seeds
     (swe_0, 4, 10, 12, 14, 18, 19, 22, 23, 30, 36, 39, 42, 47, 50), rather
     than 60 tasks broad. Note frozen-20 members here (swe_0,10,12,14,18,19,
     22,23) can be re-collected for data but their artifact stays frozen.
   - Raise seeds to ~24/task (3x) so each interpretation has a real chance of
     >=2 runs. 15 tasks x 24 = 360 runs — comparable GPU to the current
     60 x 8 = 480, just concentrated where forks occur.
   - Cheaper complementary lever: widen the temperature ceiling (e.g. add
     1.15/1.3) to induce more interpretation spread per seed.
   - Optional: cadence 16 (vs 32) for terse Qwen3 to raise reads/run, lifting
     class-labeled reads per committing run (helps the reads-per-class axis;
     does NOT help the runs-per-class axis, which the seed increase targets).
   - Before that re-collection, execute the artifact revision (decision 2)
     scoped to exactly the re-collected tasks, so decision-coverage is high
     when the new runs land.

## Why the collection is still not wasted

98/480 TASK_DONE, 9,401 reads, clean multi-layer activations, verified
protocol. The coverage-based results (decision 3) are real and publishable as
a 32B scale-up of the A1/gate2/lead-time story. Only the novel make-or-break
lean-separation claim is blocked, and only by run diversity.

---

## Rejected plan (drafted 2026-07-22, superseded above)

Original intent: additive trace-grounded vocabulary revision of the new-40
entries (code-term anchors + code-style signature variants), relabel to
`models/v2_32b_r2`, then run gates. Rejected once the ceiling measurement
showed the gate-power limit is run diversity (2-fork ceiling), not label
coverage — so the revision's make-or-break payoff is +1 fork. The revision
still has value as re-collection prep (decision 2/4), not as a rescue of this
collection. Binding rules from that plan (additive only; trace-grounded; no
outcome feedback; auditor 0-errors + zero-leak tests green; frozen-20 never
edited) still govern the revision if/when it is executed.
