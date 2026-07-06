# 012 — A0 collection v1.5: grounded prompts, longer generations, diagnostics

---
status: agreed (owner 2026-07-05: "make all the changes that you think might
help us. include all the loggings…")
date: 2026-07-05
---

**Context.** The 3-task v1 sample exposed: (1) ungrounded traces cannot utter
repo-specific API names, so blockers whose interpretation classes ARE such
names (all of swe_1) never commit; (2) 24-read traces are too short for A3 —
36/39 lean sequences never settle; (3) when a number looks wrong there was no
audit trail from label back to text.

**Decisions.**
1. **Repo grounding, leak-analyzed.** `scripts/extract_task_context.py` reads
   the gold patch ONLY for touched file paths (tests/changelogs/docs excluded
   — test names encode expected behaviour) and copies the **pre-patch**
   contents of those files out of the task's docker image. Pre-patch files
   cannot contain the patch's added lines by construction; residual risk that
   a resolution token already existed pre-patch is accepted and will be
   visible in the audit trail. `full_info/` variants are never touched.
   Fallback ladder, always recorded per task in the collection manifest:
   docker → paths+function-names only → instruction-only. This moves v1
   collection *toward* the benchmark's own setting (hil-bench baseline agents
   see the real repo).
2. **Generations 768 → 1536 tokens** (default) so lean sequences can settle.
3. **Mild deliberation nudge** appended to the prompt ("weigh alternatives
   before committing") — collection design, recorded in the manifest;
   removable with `--no-nudge` for an ablation.
4. **Diagnostics everywhere:** collection manifest (versions, GPU, prompt
   hashes, grounding mode, per-run timing/finish-reason/answer-signature) +
   events.jsonl; labeler debug JSONL (per-read anchor scores + why-unlabeled,
   per-commitment signature scores + snippet) + `scripts/audit_labels.py`
   (human-readable sampled audit — the decisions/005 audit, mechanized);
   A2 loss-history JSONL + GRL-treadmill self-check after every training;
   A3 settle-rate + sequence-length stats + benign-spread reference saved
   with the calibration; gates print explicit INSUFFICIENT-DATA notes instead
   of NaN.
5. Answer signature upgraded to a hash of the last fenced code block
   (whitespace-normalized) — still only an upper bound on interpretation
   diversity; the labeler measures the real thing.

**Consequences.** RunLog schema unchanged — v1 sample data stays compatible.
AWS order becomes: extract_task_context.py → collect_a0.py → (laptop)
train_offline.py + audit_labels.py. AWS_RUNBOOK.md updated.
