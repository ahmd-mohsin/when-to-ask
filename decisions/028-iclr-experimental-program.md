# 028: the ICLR experimental program — one matrix, five supports

Date: 2026-08-19. Status: PROPOSED (owner directive in-session: "a worthy
paper for ICLR in about a month; design experiments + ablations, implement,
then writing takes ~10 days"). Follows 027 Amendments A/B (mechanism dropped;
the measurement paper IS the paper). This entry pre-registers every remaining
experiment, its budget, and its stop rule, so the next working session
executes without redesigning. Paper skeleton: paper/OUTLINE.md.

**Boundary (from 027 Amendment B, restated so nobody drifts):** everything
below is MEASUREMENT of where the fork signal lives. No mechanism claims this
cycle, no trigger tuning, sealed pool stays sealed. If a cell turns out
strong enough to make a mechanism viable, that is FUTURE WORK in this paper
and possibly the next one.

## T1 — THE table: representation x fork-signal matrix (paper centerpiece)

One question per row: can THIS representation of a run's state tell that two
runs resolved the same blocker differently? Columns: (a) same-vs-diff class
separation AUROC on the frozen commitment-pair pool (7,321 same / 1,246 diff
pairs — the pool that scored v1=0.555, v2=0.580), (b) stage-1 detection F1
where applicable, (c) access requirements (single-run? ensemble? registry?).

| row | representation | status |
|---|---|---|
| R1 | raw activations, 8 layers (32B) | DONE — permutation null (026 B) |
| R2 | learned L-space (A2) | DONE — null, n_testable=0 (026 B) |
| R3 | surface behavior hashes | DONE — 0.555 (027 A) |
| R4 | MiniLM embedding of mutating turn | DONE — 0.580 (027 B) |
| R5 | strong embedder (BAAI/bge-large-en-v1.5 via transformers, CPU) | NEW, ~1h |
| R6a | single-run LLM introspection: Fable subagent reads ONE run's prefix (registry-blind), binary ask/no-ask + free-text "what unresolved decision?" — scored as detection F1 vs census | NEW, ~240 judgments |
| R6b | ensemble LLM comparison: Fable subagent sees TWO runs' commitment excerpts (registry-blind), judges same/different resolution — scored as separation AUROC on a 200-pair stratified sample of the pool | NEW, ~200 judgments |

R1-R5 are representations; R6a/b measure whether the signal is recoverable
by full-context LLM JUDGMENT — single-run (R6a, the Ask-or-Assume-style
detector the field builds) vs ensemble-with-comparison (R6b). Expected shape
of the story if the negatives hold and R6b succeeds: the signal exists, is
invisible to every generic representation and to single-run judgment, and
becomes visible only under decision-aware comparison ACROSS runs. Every cell
reported as-run regardless.

R6 protocol (frozen): prompts registry-blind (contract-test greppable);
excerpts = the mutating turn's subgoal + command + error signature + ±400
raw chars of trace context (the labeler's own window rule); pair sample
stratified 100 same / 100 diff, seed 0; R6a prefix points = the round before
each run's first commitment on a forked blocker + matched rounds on
non-forked tasks; transport = Fable subagents in-session (025 Amendment A
precedent), chunked 20/agent, resumable via per-chunk result files; ~460
judgments ≈ well under one session budget. STOP rule: if session limits bite,
report the completed fraction — no silent truncation.

## F2 — the rollout-budget ablation (the "how many runs buy the signal" figure)

Census vs N: for N in {2, 4, 8, 12, 16, 24} seeds (20 resamples each, seed 0),
recompute: fraction of tasks with >=1 fork, forked-decision count, and the
per-decision run-level TESTABILITY floor distribution (026's exact-test
machinery). Data exists; pure CPU analysis (~1h implement + minutes to run).
This figure turns the "2/3 of tasks fork at N=24" census fact and the
"n_testable=0 at ~6 runs" floor finding into one design-guidance plot —
reviewers' "so what should I do" answer.

## T3 — probe-robustness appendix (closes the last gate-2 escape)

Full-dim (5120) linear probe + a 2-layer MLP probe (hidden 512, the one
nonlinear family, pre-declared here) on the FIXED labels, gate-2 task, same
s6,s7 split as the text control, vs the causal-masked text baseline 0.730.
Box, ~2-4h. Whatever the number, it goes in: if the MLP closes the gap the
internals story gets a caveat; if not, the negative is escape-proof.

## T5 — cross-scale replication (generality row)

R3/R4 AUROC + fork census on the 14B collection (data/a0_v2, local) and the
v1 7B sample. CPU, ~1 day implement+run. Activations rows at those scales
already exist historically (015/018) — cite with 026's corrected-baseline
caveats, do not rerun.

## F4 — lead-time and fork anatomy (descriptive, all data exists)

Commitment-dating distributions; fork onset vs commitment (gate7 machinery);
splits by fork type (structural/value, data/fork_type_annotations.json) and
by temperature arm. Feeds the census section; ~half a day.

## T6 — ground-truth error bounds (framing, no new labels)

Bound census error with what exists: audit_signature_discrimination (3
strict suspects / 214), the 46-disagreement adjudication record, judge
validation numbers (025 A). One paragraph + one appendix table. Optional
(owner time permitting, NOT blocking): ~50-item owner hand-label of fork/
no-fork tasks for a human-agreement point.

## Execution order (dependencies, ~2 weeks of experiments)

1. F2 + F4 (local CPU, no new deps) — start immediately, they de-risk nothing
   and feed the census section.
2. R5 (local, ~1h) then T5 (local, ~1 day).
3. R6a/R6b (Fable subagents; the only session-budget item) — after R5 so the
   embedder rows are frozen first.
4. T3 on the box (parallel to any of the above; owner runs or pastes).
5. Analysis freeze -> writing (paper/OUTLINE.md section order), target ~10
   writing days, ICLR deadline buffer >= 5 days.

## Amendment A (2026-08-20 — as-implemented instantiation, frozen BEFORE first runs)

Recorded before any 028 number exists. Nothing above is edited; these pin
the implementation choices the entry left open, each settled pre-run:

1. **F2 floor universe (pre-run adversarial review finding, confirmed on
   real data).** "026's exact-test machinery" computes floors over the
   GATE5 universe — per decision, runs carrying >=1 class-labeled READ
   (debug-trail phase==1), eligibility >=4 class-labeled reads and >=2
   classes — which on the full fixed labels has 37 eligible decisions at
   median 6 runs vs 63/10 in the commitment trail. F2's quantity (c) is
   therefore computed in the gate5 universe (execution-checked to reproduce
   exactly 37 eligible / median 6 / 13 testable before the first run);
   quantities (a)/(b) stay census (commitment-trail). Commitment-universe
   floors are emitted as clearly-labeled supplementary data.
2. **F4 frozen definitions**: commitment round per offline_ask_headtohead.
   commit_rounds; fork onset = min datable commitment round in the fork;
   fork completion = gate7 formula per blocker; ask window = completion -
   onset; modal class = largest committed class (ties by name sort).
3. **R6 instantiation** (scripts/r6_build_items.py IS the frozen
   instantiation, committed before any judgment): R6a = 4 judged runs per
   task x 60 tasks (equal per-task fire chances; fewer where fewer eligible
   runs), forked-task cut = the run's first commitment round on a forked
   blocker, non-forked cuts cycled from the sorted forked-k pool; prefix =
   turn-by-turn ActionEvent summaries (subgoal/command<=400ch/error) + the
   verbatim baseline/instruction.md; R6b stratification = round-robin over
   (task, blocker) groups, rng seed 0; judge-visible payload keys pinned by
   contract test (registry-blind, 4 new tests; suite 240). R6b AUROC score
   = confidence signed by the different/same answer, AUROC computed as the
   exact Mann-Whitney U with ties counted 0.5 (scores take 10 discrete
   values; a non-tie-aware rank formula was caught pre-run and replaced).
   Transport: items shuffled deterministically (seed 0) before 20-item
   chunking so no judge sees a truth-homogeneous chunk; payload item_ids
   re-keyed to opaque salted hashes (build-order ids monotonically encoded
   ground truth — an id-threshold rule scored 100% with zero excerpt use;
   caught pre-run, private map results/r6_items/blind_id_map.json
   translates back at scoring); judges instructed to read ONLY their
   payload file (transcripts audited for stray reads after the run);
   chunks stored via scripts/r6_store_chunks.py only when judgment ids
   exactly match the payload's id set (resumable: unstored chunks re-run).
4. **T5 data fact**: the v1 7B collection recorded ZERO ActionEvents
   (predates composite logging), so its R3/R4 pair pool is structurally
   empty; the 7B row reports census only, with this reason. 14B labels
   regenerated with the fixed labeler (models/t5_v2_14b_fixed: 244
   commitments, 182 action-sourced, 16 forked blockers — matches the
   historical 14B numbers); 7B likewise (models/t5_a0_7b_fixed: 354
   commitments, all trace-sourced, 32 forked blockers).
5. **R5 embedder**: BgeEmbedder = CLS pooling + l2 norm (model-card
   recipe), truncation 256 to match the MiniLM row, batch 32; driver
   recomputes v1/v2 alongside as consistency checks (no gate — 027 B's
   fallback stands; the row is reported as-run).
6. **T3**: split code replicated verbatim from gate2_text_control.py;
   MLP = sklearn MLPClassifier hidden 512, relu, adam, alpha 1e-4, batch
   256, lr 1e-3, max_iter 100, early stopping, random_state 0; every
   unpinned hyperparameter stays at the sklearn default (a stricter
   patience that crept into the draft script was caught pre-run and
   removed; the as-run config including patience and validation fraction
   is recorded in the output JSON); a 256-d JL linear probe runs first as
   a consistency check vs the recorded 0.2745; input-identity guards abort
   on any npz that is not the fixed-label set (787,281 reads; s6,s7 test
   split 9,180 / 110 classes); results persist after each probe.

## Stop rules and reporting

Every cell lands in the paper as-run. No cell is rerun with variations
unless this entry is amended BEFORE the rerun. The T1 pool, splits, seeds,
prompts, and sample sizes above are frozen by this entry once accepted.
Suite must stay green; every experiment ships with its driver script in
scripts/ and its output in results/ (fresh files, never overwriting R2-era
or 026/027 artifacts).
